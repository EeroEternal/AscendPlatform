"""The ``gdn-moe`` operator family: Qwen3.6's GDN and MoE takeovers, as data.

This module is the *table* that used to live inside :mod:`ascend_platform.ops.install`
(``_build_bindings``).  The install mechanism - planning against the live module tree, wrapping
with the per-call family switch, registering a ``Patch`` per attribute write - stays in
``install.py``; this file only says **which vLLM names this model family needs** and what each one
is replaced with.  Adding a model family is therefore a new file in this package, not an edit to
``install.py``.

``required_names()``
    The op names this family promises, in table order.  ``install.py`` checks its plan against this
    list, and ``ascend-platform doctor`` can print it, so a table that silently loses an entry is visible.
``bind()``
    Build the :class:`~ascend_platform.ops.install.OpBinding` records.  It is a function (not a
    module-level constant) so importing this package stays cheap: the operator implementations
    (and torch) are imported on first use, exactly as ``install._build_bindings`` did.

Candidates: one slot per ABI name, baseline first
------------------------------------------------
Every binding carries a :class:`~ascend_platform.ops.candidates.CandidateSet` whose **baseline** is
today's torch implementation (``BASELINE_NAME = "torch"``, ``enabled_by_default=True``, always
present).  ``install.py`` binds the set's thin dispatcher instead of the implementation, so a second
implementation can be added **here, as data** - predicate over the call's shape/dtype, a recorded
``measured_ratio``, a ``note`` - with no change to the installer and no change to the ABI.  For the
target model nothing changes: with no eligible candidate the dispatcher calls the baseline, so the
behaviour is byte-identical to the pre-candidate wiring (the operator repo's
``tests/test_op_candidates.py`` pins that; ``tests/test_ops_rebinding.py`` pins the gate here).
The one exception is ``unquantized_fused_moe`` (``kind="custom_op_oot"``): its ``impl`` is a class
factory, so there is no call shape to dispatch on and it carries no set.

The one second candidate registered today is ``moe_reference_oracle`` on ``fused_experts``: the
repo's pure-torch MoE oracle (:mod:`tileinfer.testing.moe_reference`), which is *equivalent* and
importable without a device - and slow, so ``enabled_by_default=False``.  It is registered because
the seam should be exercised by something real, not because it should ever run a serving batch.  The
TileLang kernels (which live in the operator repo) are deliberately **not** referenced here: they import
``tilelang`` at module level, so a binding to them would make this file unimportable on a CPU box.
Nor is the kernel route forgotten: the MoE grouped-GEMM kernel measured **0.86x** of the aclnn
baseline on device after three optimisation rounds (``docs/status.md`` E19/H7), which is below the
``1.0x`` contract line - so it is not enabled, and a candidate that records ``measured_ratio=0.86``
stays ineligible even if someone flips its flag (see ``candidates.DEFAULT_THRESHOLD``).

Import policy: no ``vllm`` and no ``torch_npu`` at module level (the operator impls are imported
inside :func:`bind`), so the CPU launcher/doctor paths can import the family registry.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Tuple

__all__ = ["NAME", "BASELINE_NAME", "required_names", "bind"]

#: Registry name; a ``ModelProfile.op_family`` carries exactly this string.
NAME = "gdn-moe"

#: The baseline candidate's name inside every set: the torch port, i.e. the behaviour this family
#: had before the candidate seam existed.
BASELINE_NAME = "torch"


def set_engine_factories(factories: Mapping[str, Any]) -> None:
    """Publish engine-side implementations (see :func:`_Impl.impl`).

    Delegates to the registry in the package ``__init__`` so there is one storage point; the
    resolver reads :func:`engine_factories` on every lookup.  No cache to invalidate: ``bind()``
    builds a fresh resolver per call, so the next table sees the published factory (a module-level
    resolver cache would have made this publish-before-bind ordering dependent).
    """
    from ascend_platform.ops.families import set_engine_factories as publish

    publish(factories)


def _wrong_kind(op: str, served: Any) -> bool:
    """Does the provider's entry for ``op`` have the wrong shape for the binding that asks for it?

    Two independent sources must agree before this answers anything:

    * the **vocabulary** (``contracts/vocabulary.py``) declares which names are factory slots;
    * the **descriptor** declares what the provider actually published (``kind``).

    Either side being silent means "no objection" - an operator package may serve names the platform
    has not learned yet, and an untyped entry keeps working exactly as before.  Only a declared
    factory slot served as compute (the real bug: ``unquantized_fused_moe`` aliased to the compute
    function) is refused.
    """
    from ascend_platform.contracts.vocabulary import OpKind, requires_factory

    if served is None:
        return False
    if requires_factory(op) and served.kind is not OpKind.CLASS_FACTORY:
        return True
    if not requires_factory(op) and served.kind is OpKind.CLASS_FACTORY:
        return True
    return False


class _Impl:
    """Resolve ``(impl, note)`` pairs from the operator layer, one name at a time.

    The operator layer is imported through ``contracts.ports`` so the platform keeps a single
    dependency direction.  Any failure - the module is absent (the migration's own split, where
    ``fused_moe_oot`` lives only in the source repo), the attribute is gone, the module needs a
    device - is recorded against the one name that needed it.  Resolution is cached because the
    table is rebuilt on every ``install_ops`` after ``_reset`` and the misses are deterministic.
    """

    def __init__(self) -> None:
        self._cache: dict = {}

    def impl(self, op: str, module: str, attribute: str, *, ask_provider: bool = True):
        """Resolve one op: the **provider** first, then the legacy module layout.

        ``op`` is the *semantic* name when the two sides share one (``moe.routing_topk_softmax``,
        ``moe.experts``, ...) and the vLLM-facing name when they do not - the provider is asked for
        that key, so a name the provider does not serve is reported as "the provider does not serve
        it" instead of silently falling through.  The vLLM-facing name stays the fallback key for
        the layout path, because that is what ``ports._LAYOUTS`` addresses.

        Three sources, in this order:

        1. an **engine-published** implementation (``install_ops(engine_factories=...)``) - the only
           one that can supply a ``class_factory``, because only ``engine/abi`` knows vLLM's class
           extension point;
        2. the **provider** (``contracts/provider.py``): a package says which op names it serves and
           where they live, so the platform does not have to know its directory structure;
        3. the **legacy layout** (``ports._LAYOUTS``), kept for layers that publish no descriptor.
        The layout path stays as the fallback for a layer that installs no descriptor - which is
        every layer today, including ``ascend_attention``, whose descriptor is the next step on that
        side.  Both paths are tolerant: a miss is a fact about this op.
        """
        key = (op, module, attribute)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        from ascend_platform.contracts.provider import provider as _provider

        from ascend_platform.ops.families import engine_factories

        published = engine_factories().get(op)
        if published is not None:
            self._cache[key] = (published, "")
            return self._cache[key]

        provider = _provider() if ask_provider else None
        if provider is not None:
            value, provider_note = provider.load(op)
            if value is not None and _wrong_kind(op, provider.op(op)):
                # The provider serves this name, but with the wrong *shape*: a plain compute
                # function where the record needs ``base_cls -> subclass`` (or vice versa).  Binding
                # it would look like success and fail at model instantiation, so it is refused here
                # with the vocabulary as the authority, and the layout path is not consulted either
                # (it would hand back the same function under another name).
                self._cache[key] = (None, provider_note)
                return self._cache[key]
            if value is not None:
                self._cache[key] = (value, "")
                return self._cache[key]
        else:
            provider_note = ""

        from ascend_platform.contracts.ports import ops_module

        try:
            value = getattr(ops_module(module), attribute)
        except Exception as exc:  # noqa: BLE001 - "this op is missing" is data, not a crash
            note = (
                f"the operator layer has no {module}.{attribute} ({type(exc).__name__}: {exc}); "
                "this op stays on vLLM's own implementation"
            )
            if provider_note:
                note = f"{provider_note}; {note}"
            result = (None, note)
        else:
            result = (value, "")
        self._cache[key] = result
        return result


#: Module paths of the two vLLM files the GDN layer builds its call sites out of.
_GDN = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
_FLA = "vllm.model_executor.layers.fla.ops"
_MAMBA = "vllm.model_executor.layers.mamba.ops"

#: The op names, in the order the table binds them (pinned here by ``tests/test_ops_rebinding.py``;
#: the operator repo pins the wiring itself in ``tests/test_ops_install.py``).
REQUIRED_NAMES: Tuple[str, ...] = (
    "chunk_gated_delta_rule",
    "l2norm_fwd",
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "fused_sigmoid_gating_delta_rule_update",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "chunk_local_cumsum",
    "fused_post_conv_prep",
    "topk_softmax",
    "moe_align_block_size",
    "fused_experts",
    "unquantized_fused_moe",
    "silu_and_mul",
)


def required_names() -> Tuple[str, ...]:
    """The op names this family must bind (or report as declined/disabled)."""
    return REQUIRED_NAMES


# =============================================================================================
# The candidate seam: what may run instead of the torch port, and under which gate
# =============================================================================================

#: The one recorded reason a kernel candidate is not enabled here.  It quotes the repo's own
#: measurement instead of a guess, because the next person to touch this file will ask "why not?".
_MOE_REFERENCE_NOTE = (
    "certification oracle, not a speed-up: tileinfer/testing/moe_reference.reference_moe is the "
    "pure-torch fp32 MoE oracle the grouped-GEMM kernels are certified against (per-token Python "
    "loop). It is equivalent to the torch port - which is the only reason it can be registered - "
    "and slower, which is why it is off by default. The kernel route it certifies measured 0.86x "
    "of the aclnn baseline on device after three optimisation rounds (docs/status.md E19/H7), "
    "below the 1.0x contract line, so nothing in this family is enabled. A candidate may only be "
    "enabled after a device run above that line, with its ratio recorded in measured_ratio."
)


def _fp32_tiny_experts(call_spec: Any) -> bool:
    """The oracle's shape predicate: 2-D, fp32, contiguous, at most 2 tokens.

    A predicate reads the *call's* arguments and nothing else: ``.shape``/``.dtype`` of a real
    tensor and ``is_contiguous()``.  It never calls ``.item()``/``.max()`` or does tensor arithmetic
    (that would be a host sync per call), and it never calls the implementation - so it is safe to
    run inside a forward and safe to run on a fake (the operator repo's ``tests/test_op_candidates.py``).
    """
    hidden_states = call_spec.arg(0, "hidden_states")
    shape = getattr(hidden_states, "shape", ())
    if len(shape) != 2 or shape[0] > 2:
        return False
    return (
        str(getattr(hidden_states, "dtype", None)) == "torch.float32"
        and bool(hidden_states.is_contiguous())
    )


def _reference_experts(
    hidden_states: Any,
    w1: Any,
    w2: Any,
    topk_weights: Any,
    topk_ids: Any,
    activation: Any = "silu",
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    expert_map: Any = None,
    quant_config: Any = None,
    **extras: Any,
) -> Any:
    """``fused_experts`` served by the repo's oracle - the second, *equivalent* implementation.

    The ABI is :func:`ascend_platform.ops.moe.fused_experts`' (same positional order); what differs
    is only the body, which is :func:`tileinfer.testing.moe_reference.reference_moe` (fp32, one
    Python loop per token, no slot layout).  It implements the same unquantised SwiGLU expert MLP,
    and it refuses everything the oracle cannot express rather than silently computing the wrong
    math: a quantised config, a shared expert, an expert map, or a router weight applied on the
    input.  Both the oracle and the refusal helper are resolved through the operator port *on call*,
    so this module stays importable on a box with no operator layer and no torch.
    """
    from ascend_platform.contracts.ports import operator_layer, reference_moe

    _check_unquantized = operator_layer().get("ops", "_check_unquantized")

    # The same refusal the torch port makes: an unquantised config (or None) only.
    _check_unquantized(quant_config)
    activation_name = str(getattr(activation, "value", activation)).strip().lower()
    if activation_name != "silu":
        raise NotImplementedError(
            f"moe_reference_oracle: activation={activation!r} is not implemented; the oracle is "
            "SwiGLU only (the same restriction the torch port documents)"
        )
    if expert_map is not None or apply_router_weight_on_input or extras:
        raise NotImplementedError(
            "moe_reference_oracle cannot honour expert_map / apply_router_weight_on_input / the "
            "shared-expert extensions: reference_moe is the plain routed SwiGLU oracle. Refusing "
            "is the point - a candidate must never silently compute a different function."
        )
    out = reference_moe(hidden_states, w1, w2, topk_ids, topk_weights)
    return out.to(hidden_states.dtype)


def _with_candidates(binding: Any) -> Any:
    """Attach ``binding``'s :class:`~ascend_platform.ops.candidates.CandidateSet`.

    ``binding.impl`` stays the baseline (nothing downstream changes what it reports), and the set is
    what ``install.py`` binds the dispatcher for.  A binding with no implementation (a reported
    miss, e.g. ``chunk_local_cumsum``) gets no set: there is nothing to dispatch between.

    A ``kind="custom_op_oot"`` binding gets no set either, for a different reason: its ``impl`` is a
    *class factory* (``impl(base_cls) -> subclass``), which the candidate seam cannot dispatch on
    (a predicate reads tensors from a call's arguments).  The implementation that *does* have a call
    shape is the ``fused_experts`` record, and that is where the set lives.
    """
    if binding.impl is None:  # pragma: no cover - every entry in the table has an impl today
        return binding
    if binding.kind == "custom_op_oot":
        return binding
    from ascend_platform.ops.candidates import Candidate, CandidateSet, always

    impl = binding.impl
    baseline = Candidate(
        name=BASELINE_NAME,
        impl=impl,
        predicate=always,  # the fallback is never consulted; the shape keeps the set uniform
        enabled_by_default=True,
        note=(
            f"the torch port {getattr(impl, '__module__', '?')}."
            f"{getattr(impl, '__qualname__', getattr(impl, '__name__', '?'))}; "
            "the pre-candidate behaviour, byte for byte"
        ),
    )
    chosen = CandidateSet(op=binding.op, baseline=baseline)
    if binding.op == "fused_experts":
        chosen.add(
            Candidate(
                name="moe_reference_oracle",
                impl=_reference_experts,
                predicate=_fp32_tiny_experts,
                enabled_by_default=False,
                note=_MOE_REFERENCE_NOTE,
            )
        )
    return replace(binding, candidates=chosen)


def bind() -> List[Any]:
    """Build this family's :class:`OpBinding` records; imports the torch impls lazily."""
    # Imported here, not at module level: ``install`` imports this module, so the data class has to
    # come from the already-loaded install module (a top-level import would be circular), and the
    # operator modules import torch.
    from ascend_platform.ops.install import OpBinding

    # The implementations belong to the operator layer; this table only names them.  Resolution is
    # per name, not per module: a layer that ships eleven of these and not the twelfth used to raise
    # on the first missing module and take the whole table with it, so thirteen required ops ended
    # up "0 bound" with one warning that blamed the model profile.  Now the miss is a property of
    # the one record that needs it (``impl=None`` + ``impl_note``) and ``install._plan`` reports it.
    resolver = _Impl()

    moe_topk = resolver.impl("moe.routing_topk_softmax", "moe", "topk_softmax")
    moe_align = resolver.impl("moe.align_block_size", "moe", "moe_align_block_size")
    moe_experts = resolver.impl("moe.experts", "moe", "fused_experts")
    moe_unquantized = resolver.impl("unquantized_fused_moe", "moe", "unquantized_fused_moe")
    moe_silu = resolver.impl("moe.swiglu_gate", "moe", "silu_and_mul")
    # ``engine.moe_method`` is a **factory slot** in the vocabulary, so the provider may serve it
    # only with ``kind="class_factory"``; the alias ``unquantized_fused_moe`` (plain compute) is
    # refused by ``_wrong_kind`` instead of being bound as a factory and failing at model-load time.
    # Turning a compute function into vLLM's method object is the engine-ABI layer's job
    # (``engine/abi/``), so until that factory is published this op stays unbound - with the reason.
    oot_method = resolver.impl("engine.moe_method", "fused_moe_oot", "make_oot_method")
    chunk_rule = resolver.impl("chunk_gated_delta_rule", "fla.chunk", "chunk_gated_delta_rule")
    conv_fn = resolver.impl("causal_conv1d_fn", "fla.causal_conv1d", "causal_conv1d_fn")
    conv_update = resolver.impl("causal_conv1d_update", "fla.causal_conv1d", "causal_conv1d_update")
    gating_update = resolver.impl(
        "fused_sigmoid_gating_delta_rule_update", "fla.gating", "fused_sigmoid_gating_delta_rule_update"
    )
    recurrent_decode = resolver.impl(
        "fused_recurrent_gated_delta_rule_packed_decode",
        "fla.recurrent",
        "fused_recurrent_gated_delta_rule_packed_decode",
    )
    post_conv_prep = resolver.impl("fused_post_conv_prep", "fla.post_conv", "fused_post_conv_prep")
    l2norm = resolver.impl("l2norm_fwd", "gdn", "l2norm_fwd")
    cumsum = resolver.impl("chunk_local_cumsum", "gdn", "chunk_local_cumsum")

    op_consumer = "module-global call in _forward_core"
    records = [
        OpBinding(
            op="chunk_gated_delta_rule",
            platform_name="gdn.chunk_gated_delta_rule",
            family="gdn",
            targets=(f"{_GDN}.ChunkGatedDeltaRule",),
            kind="custom_op",
            impl=chunk_rule[0],
            impl_note=chunk_rule[1],
            why=(
                "GDN prefill runs through the CustomOp ChunkGatedDeltaRule "
                "(qwen_gdn_linear_attn.py:290-311); on a non-CUDA platform its __init__ pins "
                "self._forward_method = self.forward_native (:176-177, :211), which is upstream's "
                "Triton FLA chain (fla/ops/chunk.py:137-245) - unavailable on Ascend. Isolated "
                "prefill needs this op, so without the takeover GDN prefill cannot run."
            ),
            remove_when=(
                "when upstream's ChunkGatedDeltaRule gains an NPU/torch branch, or when we switch "
                "to CustomOp.register_oot instead of replacing methods on the in-tree class."
            ),
            upstream_ref=(
                "vllm__model_executor__layers__mamba__gdn__qwen_gdn_linear_attn.py:290-416; "
                "vllm__model_executor__custom_op.py:169-207"
            ),
        ),
        OpBinding(
            op="l2norm_fwd",
            platform_name="gdn.l2norm",
            family="gdn",
            targets=(f"{_FLA}.l2norm.l2norm_fwd",),
            aliases=(f"{_FLA}.chunk.l2norm_fwd", f"{_GDN}.l2norm_fwd"),
            impl=l2norm[0],
            impl_note=l2norm[1],
            why=(
                "q/k normalisation inside the GDN chain; upstream's body is a Triton kernel "
                "(fla_l2norm.py:114-149) and the signature is identical (x, eps=1e-6, "
                "output_dtype=None, :95-97), so this is a true drop-in. eps sits inside the "
                "squared sum, the convention our port documents from l2norm_fwd_kernel1."
            ),
            remove_when="when vLLM's fla ops ship a non-Triton l2norm (or we drop ops/gdn.py).",
            upstream_ref="fla_l2norm.py:28-151",
        ),
        OpBinding(
            op="causal_conv1d_fn",
            platform_name="gdn.conv1d_prefill",
            family="gdn",
            targets=(f"{_MAMBA}.causal_conv1d.causal_conv1d_fn",),
            aliases=(f"{_GDN}.causal_conv1d_fn",),
            impl=conv_fn[0],
            impl_note=conv_fn[1],
            why=(
                "GDN prefill conv over the packed batch; upstream is the Triton "
                "_causal_conv1d_fwd_kernel (vllm ...mamba/ops/causal_conv1d.py:16-746). Our "
                f"signature is that wrapper's verbatim (:468-486), see mamba/ops/causal_conv1d.py:353-371. "
                f"Also rebound in the consumer module ({op_consumer}, qwen_gdn_linear_attn.py:1365), "
                "because a name imported by value cannot be patched at its definition alone."
            ),
            remove_when="when vllm's mamba ops gain a native/torch causal_conv1d.",
            upstream_ref="vllm__model_executor__layers__mamba__ops__causal_conv1d.py:468-746",
        ),
        OpBinding(
            op="causal_conv1d_update",
            platform_name="gdn.conv1d_update",
            family="gdn",
            targets=(f"{_MAMBA}.causal_conv1d.causal_conv1d_update",),
            aliases=(f"{_GDN}.causal_conv1d_update",),
            impl=conv_update[0],
            impl_note=conv_update[1],
            why=(
                "GDN decode/decode-spec conv state update; upstream is "
                "_causal_conv1d_update_kernel (:749-1067). Our signature is the wrapper's verbatim "
                f"(:1069-1081), see mamba/ops/causal_conv1d.py:505. {op_consumer}: "
                "qwen_gdn_linear_attn.py:1344, :1378, :1674. Isolated decode needs this op."
            ),
            remove_when="when vllm's mamba ops gain a native/torch causal_conv1d_update.",
            upstream_ref="vllm__model_executor__layers__mamba__ops__causal_conv1d.py:1069-1190",
        ),
        OpBinding(
            op="fused_sigmoid_gating_delta_rule_update",
            platform_name="gdn.fused_sigmoid_gating",
            family="gdn",
            targets=(f"{_FLA}.fused_sigmoid_gating.fused_sigmoid_gating_delta_rule_update",),
            aliases=(f"{_GDN}.fused_sigmoid_gating_delta_rule_update",),
            impl=gating_update[0],
            impl_note=gating_update[1],
            why=(
                "Per-decode-step gate + state update (sigmoid gating, unavoidable on the decode "
                "path); upstream is a Triton kernel (fla_fused_sigmoid_gating.py:24-179) with the "
                "signature our port reproduces verbatim (tileinfer/vllm/ops/fla/gating.py:216-234). Called at "
                f"qwen_gdn_linear_attn.py:1457, :1484, :1542 ({op_consumer})."
            ),
            remove_when="when the gating kernel has a non-Triton branch upstream.",
            upstream_ref="fla_fused_sigmoid_gating.py:181-199",
        ),
        OpBinding(
            op="fused_recurrent_gated_delta_rule_packed_decode",
            platform_name="gdn.recurrent_decode",
            family="gdn",
            targets=(
                f"{_FLA}.fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode",
            ),
            aliases=(f"{_GDN}.fused_recurrent_gated_delta_rule_packed_decode",),
            impl=recurrent_decode[0],
            impl_note=recurrent_decode[1],
            why=(
                "Packed recurrent decode (one token per sequence) used when "
                "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE is on; upstream is a Triton kernel "
                "(fla_fused_recurrent.py:256-480), our signature is the wrapper's verbatim "
                f"(:339-350) - tileinfer/vllm/ops/fla/recurrent.py:511-521. Called at qwen_gdn_linear_attn.py:1684 "
                f"({op_consumer})."
            ),
            remove_when="when the packed-decode kernel has a non-Triton branch upstream.",
            upstream_ref="fla_fused_recurrent.py:339-480",
        ),
        OpBinding(
            op="chunk_local_cumsum",
            platform_name="gdn.chunk_local_cumsum",
            family="gdn",
            targets=(f"{_FLA}.cumsum.chunk_local_cumsum",),
            aliases=(f"{_GDN}.chunk_local_cumsum",),
            impl=cumsum[0],  # the real port; abi_ok=False is about the call contract
            impl_note=cumsum[1],
            abi_ok=False,            abi_note=(
                "declined - not a drop-in: upstream takes (g, chunk_size, reverse, cu_seqlens, "
                "chunk_indices, head_first, output_dtype) and handles 3-D/4-D layouts "
                "(fla_cumsum.py:243-252), while ops/gdn.py's port is token-major [T, ...] with "
                "(g, chunk_size, reverse, cu_seqlens) only. Our chunk chain already calls our own "
                "cumsum internally, so nothing needs the vLLM symbol replaced."
            ),
            why="Reported, not patched: see abi_note.",
            remove_when=(
                "when ops/gdn.py grows the head_first/chunk_indices/output_dtype ABI, at which "
                "point this entry becomes a normal rebinding."
            ),
            upstream_ref="fla_cumsum.py:243-252",
        ),
        OpBinding(
            op="fused_post_conv_prep",
            platform_name="gdn.post_conv_prep",
            family="gdn",
            targets=(f"{_FLA}.fused_gdn_prefill_post_conv.fused_post_conv_prep",),
            aliases=(f"{_GDN}.fused_post_conv_prep",),
            impl=post_conv_prep[0],
            impl_note=post_conv_prep[1],
            why=(
                "GDN prefill's post-conv prep: splits the packed q|k|v conv output, L2-normalises "
                "q/k, and builds the log gate g and the delta-rule beta. The layer calls it before "
                "the chunk kernel (qwen_gdn_linear_attn.py:1428) and again in the warmup (:1113), and "
                "the prefill chunk call passes use_qk_l2norm_in_kernel=False (:1529), so the "
                "normalisation only happens here. Upstream's body is one Triton kernel plus a "
                "launcher (fla/ops/fused_gdn_prefill_post_conv.py:19-149 and :152-248, both local), "
                "and the ten-argument signature our port reproduces is that launcher's (:152-163) - "
                "identical to the call site's (:1428-1439) - including its dtype contract: q/k/v in "
                "the conv dtype, g and beta always float32 (:199-203). Without it prefill cannot "
                "even build its inputs on Ascend."
            ),
            remove_when=(
                "when vLLM's fused_gdn_prefill_post_conv gains a non-Triton branch, or when the "
                "V1 prefill path stops going through this fused prep."
            ),
            upstream_ref=(
                "fla/ops/fused_gdn_prefill_post_conv.py:19-149, :152-248; "
                "qwen_gdn_linear_attn.py:27-31, :1113-1124, :1428-1439, :1529; "
                "model_executor/layers/fla/ops/__init__.py:10"
            ),
        ),
        OpBinding(
            op="topk_softmax",
            platform_name="moe.routing_topk_softmax",
            family="moe",
            targets=("vllm._custom_ops.topk_softmax",),
            impl=moe_topk[0],
            impl_note=moe_topk[1],
            why=(
                "MoE routing. vLLM 0.23 calls the wrapper in vllm/_custom_ops.py:2494-2509, i.e. "
                "torch.ops._moe_C.topk_softmax (:2502) - a compiled CUDA kernel that does not exist "
                "on Ascend. Our signature is that wrapper's verbatim: (topk_weights, topk_ids, "
                "token_expert_indices, gating_output, renormalize, e_score_correction_bias). The one "
                "knob this port cannot honour is e_score_correction_bias (the DeepSeek-style "
                "selection bias; its placement inside the kernel is not in the local subset) and it "
                "raises NotImplementedError rather than being ignored, so an un-biased router gets "
                "the exact behaviour and a biased one fails loudly."
            ),
            remove_when=(
                "when torch.ops._moe_C.topk_softmax has a torch/NPU fallback upstream, or this port "
                "implements e_score_correction_bias."
            ),
            upstream_ref="vllm___custom_ops.py:2494-2509 (-> torch.ops._moe_C.topk_softmax :2502)",
        ),
        OpBinding(
            op="moe_align_block_size",
            platform_name="moe.align_block_size",
            family="moe",
            targets=("vllm._custom_ops.moe_align_block_size",),
            impl=moe_align[0],
            impl_note=moe_align[1],
            why=(
                "Builds the padded per-expert slot layout the grouped GEMM consumes. The call site "
                "is vllm/_custom_ops.py:2346-2363 (torch.ops._moe_C.moe_align_block_size at :2355), "
                "and the positional order is settled by it: (topk_ids, num_experts, block_size, "
                "sorted_token_ids, experts_ids, num_tokens_post_pad, expert_map) - num_experts "
                "BEFORE block_size. ops/moe.py's port matches that order exactly (it was a coin "
                "flip before this file was available). expert_map (the expert-parallel remap) is "
                "refused, not ignored: its kernel body is not local and v0 has no EP. The other "
                "vLLM helper with a similar name is a different function and is deliberately not "
                "bound - see the module docstring, point 4."
            ),
            remove_when=(
                "when torch.ops._moe_C.moe_align_block_size has a torch/NPU fallback upstream, or "
                "this port grows the expert_map (EP) semantics."
            ),
            upstream_ref=(
                "vllm___custom_ops.py:2346-2363 (-> torch.ops._moe_C.moe_align_block_size :2355)"
            ),
        ),
        OpBinding(
            op="fused_experts",
            platform_name="moe.experts",
            family="moe",
            targets=("vllm.model_executor.layers.fused_moe.fused_moe.fused_experts",),
            impl=moe_experts[0],
            impl_note=moe_experts[1],
            why=(
                "The billed MoE expert MLP (SwiGLU + weighted top-k combine). The Python call site "
                "is .../fused_moe/fused_moe.py:1474-1515, which forwards every argument to "
                "torch.ops.vllm.fused_experts (:1490) - a torch.library registration, not a "
                "rebindable attribute, so the takeover point is the Python wrapper. Our signature "
                "is that wrapper's positional order: (hidden_states, w1, w2, topk_weights, "
                "topk_ids, activation, apply_router_weight_on_input, global_num_experts, "
                "expert_map, quant_config), with the TileInfer extras keyword-only after it. "
                "activation is accepted in both spellings because the wrapper passes the enum and "
                "the inner op the string (:1480 vs :1496). The quantisation switches are refused "
                "(this port is unquantised bf16/fp16 SwiGLU), so a quantised call raises instead of "
                "computing in the wrong precision - that is the only branch that cannot be honoured. "
                "REACHABILITY (0.23): nothing calls this wrapper on the modular path any more - "
                "MoERunner goes router.select_experts -> quant_method.apply -> moe_kernel.apply "
                "(runner/moe_runner.py:524-539), and grep over the local tree finds no other "
                "consumer. The binding therefore stays for the vLLM series that does call it, and "
                "its *real* consumer on 0.23 is the 'unquantized_fused_moe' record below: the OOT "
                "method's stand-in moe_kernel calls this very function (same object, "
                "tileinfer/vllm/ops/fused_moe_oot.py: TorchMoEKernel.apply) through the moe_kernel "
                "slot, at unquantized_fused_moe_method.py:311-324. It is reached through "
                "moe_kernel.apply now, not directly - which is why it is listed as bound but is "
                "not, on its own, coverage."
            ),
            remove_when=(
                "when vLLM's FusedMoE path has a portable torch/NPU expert implementation, or this "
                "port grows the quantised branches."
            ),
            upstream_ref=(
                ".../fused_moe/fused_moe.py:1335-1422, :1474-1515 "
                "(-> torch.ops.vllm.fused_experts :1490); reached via "
                ".../fused_moe/unquantized_fused_moe_method.py:311-324 on 0.23"
            ),
        ),
        OpBinding(
            op="unquantized_fused_moe",
            platform_name="engine.moe_method",
            family="moe",
            kind="custom_op_oot",
            targets=(
                "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method"
                ".UnquantizedFusedMoEMethod",
                "vllm.model_executor.layers.fused_moe.UnquantizedFusedMoEMethod",
            ),
            impl=oot_method[0],
            impl_note=oot_method[1],
            why=(
                "The routed-expert MoE kernel on an OOT platform. vLLM 0.23 selects "
                "UnquantizedMoeBackend.OOT and returns experts_cls=None (oracle/unquantized.py:168-"
                "169), process_weights_after_loading returns before building any kernel "
                "(unquantized_fused_moe_method.py:217-222), and FusedMoEMethodBase.is_monolithic "
                "then calls None.is_monolithic() (fused_moe_method_base.py:156-158) - the "
                "AttributeError raised at MODEL-LOAD time via layer.py:457 "
                "(FusedMoE.maybe_init_modular_kernel). 'OOT handles internally' means the platform "
                "registers a subclass, and vLLM's own docstring names this exact class "
                "(custom_op.py:326-329). Registering one is a class-level act, not an attribute "
                "rebind, which is why this record uses kind='custom_op_oot': install.py calls "
                "impl(base_cls) and then base_cls.register_oot(subclass) (custom_op.py:331-353), so "
                "CustomOp.__new__ substitutes it for the UnquantizedFusedMoEMethod(...) that "
                "layer.py:359 constructs. The subclass supplies an experts_cls whose "
                "is_monolithic() is False (the modular runner must take the "
                "router.select_experts -> apply branch, runner/moe_runner.py:516-539), a stand-in "
                "moe_kernel for the kernel slot the OOT branch never fills, and the expert compute "
                "itself: ascend_platform.ops.moe.fused_experts, the same object the fused_experts "
                "record above wraps. Quantised/bias/LoRA/non-gated/DP-EP-PCP configs are refused "
                "loudly at construction (fused_moe_oot.check_unquantized_config)."
            ),
            remove_when="upstream builds an OOT moe_kernel",
            upstream_ref=(
                ".../fused_moe/unquantized_fused_moe_method.py:48-78, :157-202, :204-222, :302-324; "
                ".../fused_moe/fused_moe_method_base.py:35-49, :135-161; "
                ".../fused_moe/oracle/unquantized.py:153-175; "
                ".../fused_moe/layer.py:350-365, :450-474; "
                ".../fused_moe/runner/moe_runner.py:369-374, :498-539, :682-706; "
                ".../fused_moe/runner/shared_experts.py:92-112; "
                "model_executor/custom_op.py:109-128, :204-205, :322-353"
            ),
        ),
        OpBinding(
            op="silu_and_mul",
            platform_name="moe.swiglu_gate",
            family="dense",
            targets=(
                "vllm._custom_ops.silu_and_mul",
                "vllm.model_executor.layers.activation.silu_and_mul",
            ),
            impl=moe_silu[0],
            impl_note=moe_silu[1],
            why=(
                "SwiGLU activation for the dense/shared MLP and the expert path; upstream is "
                "torch.ops._C.silu_and_mul. Family 'dense' is OFF by default (switch.py): the "
                "target model runs without it, and [unverified] the call name - enabling it by "
                "default would only add unbound warnings to every boot."
            ),
            remove_when="when SiluAndMul's vLLM call site is verified.",
            upstream_ref="[unverified] - no activation module in reference/vllm-0.23",
        ),
    ]
    # Every record gets its candidate seam last, so the table above stays what it always was: one
    # row per vLLM name with the torch implementation documented.  ``_with_candidates`` keeps that
    # impl as the baseline (``OpBinding.impl`` is unchanged) and attaches the set install.py binds.
    return [_with_candidates(binding) for binding in records]
