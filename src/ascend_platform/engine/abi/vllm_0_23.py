"""vLLM 0.23's symbol table, and the one extension point that is not a function.

Two things live here, and they are here for the same reason - they are facts about a *vLLM release*,
not about an operator:

``ENGINE_SYMBOL``
    semantic name (``contracts/vocabulary.py``) -> the dotted path vLLM 0.23 actually ships.  A
    family table asks for ``moe.routing_topk_softmax``; this module is what turns that into
    ``vllm._custom_ops.topk_softmax``.  When vLLM moves the symbol, one line changes here and every
    family keeps its names.

``moe_method_factory``
    the routed-expert MoE method class.  On an out-of-tree platform vLLM 0.23 selects
    ``UnquantizedMoeBackend.OOT`` and hands back ``experts_cls=None``
    (``oracle/unquantized.py:168-169``); ``process_weights_after_loading`` then returns *before*
    building any kernel (``unquantized_fused_moe_method.py:217-222``), and
    ``FusedMoEMethodBase.is_monolithic`` calls ``None.is_monolithic()``
    (``fused_moe_method_base.py:156-158``) **while the model loads**.  vLLM's documented answer is
    the ``CustomOp.register_oot`` subclass (its own example is
    ``@UnquantizedFusedMoEMethod.register_oot class HPUUnquantizedFusedMoEMethod``,
    ``custom_op.py:322-353``).  This factory builds that subclass.

The factory takes the **compute function as an argument** - it never imports an operator package.
That is the whole point of the split: the engine side knows the engine, the operator side knows the
math, and this module is the only place the two meet.  Its three stand-in objects answer exactly the
members the local vLLM source reads, with the two different calling conventions kept apart:

=========================================  =========================================  ===============
member                                     read at                                    answer
=========================================  =========================================  ===============
``experts_cls.is_monolithic()``            ``fused_moe_method_base.py:156-158``        ``False`` (method)
``moe_kernel.is_monolithic``               ``fused_moe_method_base.py:161``            ``False`` (attr)
``moe_kernel.can_overlap_shared_experts``  ``fused_moe_method_base.py:43-49``          ``False``
``moe_kernel.output_is_reduced()``         ``moe_runner.py:369-374``                   ``False``
``moe_kernel.apply(**kwargs)``             ``unquantized_fused_moe_method.py:311-324`` our compute
=========================================  =========================================  ===============

``experts_cls.is_monolithic`` being a *method* and ``moe_kernel.is_monolithic`` being an *attribute*
is the trap worth naming: a stand-in that answers the first with a plain truthy attribute is a
silent wrong branch, so ``tests/test_engine_abi.py`` pins both shapes.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Mapping, Tuple

logger = logging.getLogger("ascend_platform.engine.abi.vllm_0_23")

__all__ = [
    "ENGINE_SYMBOL",
    "NO_ENGINE_SYMBOL",
    "MOE_METHOD_TARGETS",
    "OOT_CLASS_NAME",
    "target_for",
    "moe_method_factory",
    "moe_method_targets",
]


#: Semantic name -> the path vLLM 0.23 ships for it.  ``None`` means "the engine has no symbol for
#: this operation" (it is a platform-internal step, or the engine's own code already does it).
ENGINE_SYMBOL: Mapping[str, str] = {
    # MoE
    "moe.routing_topk_softmax": "vllm._custom_ops.topk_softmax",
    "moe.align_block_size": "vllm._custom_ops.moe_align_block_size",
    "moe.swiglu_gate": "vllm._custom_ops.silu_and_mul",
    "moe.experts": "vllm.model_executor.layers.fused_moe.fused_moe.fused_experts",
    # GDN / linear attention (the FLA set vLLM 0.23 vendors)
    "gdn.chunk_gated_delta_rule": (
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn.ChunkGatedDeltaRule"
    ),
    "gdn.chunk_local_cumsum": "vllm.model_executor.layers.fla.ops.cumsum.chunk_local_cumsum",
    "gdn.l2norm": "vllm.model_executor.layers.fla.ops.l2norm.l2norm_fwd",
    "gdn.fused_sigmoid_gating": (
        "vllm.model_executor.layers.fla.ops.fused_sigmoid_gating"
        ".fused_sigmoid_gating_delta_rule_update"
    ),
    "gdn.recurrent_decode": (
        "vllm.model_executor.layers.fla.ops.fused_recurrent"
        ".fused_recurrent_gated_delta_rule_packed_decode"
    ),
    "gdn.post_conv_prep": (
        "vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv.fused_post_conv_prep"
    ),
    "gdn.conv1d_prefill": "vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn",
    "gdn.conv1d_update": "vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update",
}

#: Semantic names vLLM 0.23 has **no symbol** to rebind.  Either the engine reaches the operation
#: through a platform-owned hook (attention, sampling) or it is a platform entry point with no
#: in-tree counterpart (``moe.experts_unquantized``).  Listed, not omitted: "absent from
#: ``ENGINE_SYMBOL``" and "we forgot it" must not look the same.
NO_ENGINE_SYMBOL: Tuple[str, ...] = (
    # The operator layer's own GDN decomposition: vLLM 0.23 reaches gating/conv-update/WY through
    # its FLA set (mapped above), not through these three steps, so there is nothing to rebind.
    "gdn.gating",
    "gdn.conv_update",
    "gdn.chunk_wu",
    "moe.experts_unquantized",
    "moe.slot_layout",
    "moe.prenz_pack",
    "attention.paged_prefill",
    "sampling.topk_threshold",
    "sampling.greedy",
    "sampling.top_p",
)

#: The class ``CustomOp.register_oot`` is called on, in the order vLLM 0.23 exposes it (the method
#: module first, then the package re-export that ``layer.py`` imports from).
MOE_METHOD_TARGETS = (
    "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method.UnquantizedFusedMoEMethod",
    "vllm.model_executor.layers.fused_moe.UnquantizedFusedMoEMethod",
)

#: What the subclass is called once registered.  One name, so a traceback says where it came from.
OOT_CLASS_NAME = "AscendUnquantizedFusedMoEMethod"


def target_for(semantic: str) -> str:
    """The engine path for a semantic name, or ``KeyError`` naming both sides."""
    try:
        return ENGINE_SYMBOL[semantic]
    except KeyError:
        raise KeyError(
            f"ascend-platform ABI (vLLM 0.23): no engine symbol for {semantic!r}. If vLLM ships "
            "one, add it to ENGINE_SYMBOL; if it does not, add the name to NO_ENGINE_SYMBOL so the "
            "two cases stay distinguishable."
        ) from None


def moe_method_targets() -> tuple:
    """The base-class paths, as data (install.py resolves them against the live module tree)."""
    return MOE_METHOD_TARGETS


class _Experts:
    """Stands in for the ``experts_cls`` the OOT oracle returns as ``None``."""

    def is_monolithic(self) -> bool:  # noqa: D401 - upstream's name, called with parentheses
        """``fused_moe_method_base.py:158``.  The runner computes top-k itself, so ``False``."""
        return False

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "_Experts(is_monolithic=False)"


class _PrepareFinalize:
    """The one member of the prepare/finalize object the method reads."""

    def topk_indices_dtype(self) -> Any:
        """``fused_moe_method_base.py:137-138``: ``None`` means "do not recast topk_ids"."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "_PrepareFinalize(topk_indices_dtype=None)"


class _MoEKernel:
    """Stands in for the kernel the OOT branch never builds; computes with the injected callable."""

    #: Read WITHOUT parentheses (``fused_moe_method_base.py:161``).
    is_monolithic = False
    #: ``fused_moe_method_base.py:43-49``: the runner applies the shared expert itself, outside us.
    can_overlap_shared_experts = False

    def __init__(self, compute: Callable[..., Any]) -> None:
        self._compute = compute
        self.prepare_finalize = _PrepareFinalize()

    def output_is_reduced(self) -> bool:
        """``moe_runner.py:369-374``: output is not all-reduced; the runner does that."""
        return False

    def apply(
        self,
        hidden_states: Any,
        w1: Any,
        w2: Any,
        topk_weights: Any,
        topk_ids: Any,
        activation: Any = "silu",
        apply_router_weight_on_input: bool = False,
        global_num_experts: int = -1,
        expert_map: Any = None,
        *,
        shared_experts: Any = None,
        shared_experts_input: Any = None,
    ) -> Any:
        """The routed expert MLP: the operator layer's compute, through the kernel slot.

        ``shared_experts``/``shared_experts_input`` are accepted and ignored **on purpose**: the
        runner always passes them and applies them itself in ``NO_OVERLAP`` order *before* this call
        (``moe_runner.py:512-514``), then adds the output.  Adding them here would double-count the
        dense expert.
        """
        return self._compute(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            quant_config=None,
        )

    def apply_monolithic(self, *args: Any, **kwargs: Any) -> Any:
        """Never reached: :attr:`is_monolithic` is ``False`` (``moe_runner.py:516-522``)."""
        raise NotImplementedError(
            "ascend-platform: apply_monolithic is not implemented, and cannot be reached while "
            "is_monolithic is False. Reaching it means the runner's branch changed - read "
            "fused_moe_method_base.py:154-161 before implementing it."
        )

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"_MoEKernel(compute={getattr(self._compute, '__name__', self._compute)!r})"


def moe_method_factory(compute: Callable[..., Any]) -> Callable[[type], type]:
    """Build the ``impl`` for the ``engine.moe_method`` slot: ``(base_cls) -> subclass``.

    ``compute`` is the operator layer's routed-expert MLP (``moe.experts`` /
    ``moe.experts_unquantized``).  It is a parameter, not an import: this module must import on a box
    with no operator package installed (``doctor``, the CPU suite).
    """
    if not callable(compute):
        raise TypeError(
            f"ascend-platform MoE method factory: compute must be callable, got {compute!r}"
        )

    def factory(base_cls: type) -> type:
        """The ``custom_op_oot`` contract: one class in, one subclass out."""
        if not isinstance(base_cls, type):
            raise TypeError(
                "ascend-platform MoE method factory: expected vLLM's UnquantizedFusedMoEMethod "
                f"class, got {base_cls!r}; install.py resolves it from the binding's target"
            )

        class AscendUnquantizedFusedMoEMethod(base_cls):  # noqa: N801 - named for its replacement
            """The OOT method: vLLM's class, with the kernel slot pointing at our compute."""

            def __init__(self, moe: Any) -> None:
                super().__init__(moe)
                backend = getattr(self, "unquantized_backend", None)
                backend_name = str(getattr(backend, "name", backend))
                if backend_name != "OOT":
                    raise RuntimeError(
                        "ascend-platform OOT MoE: this class only implements the OOT backend, but "
                        f"select_unquantized_moe_backend chose {backend_name!r}. Registering it "
                        "there would run our compute on top of that backend's weight-format "
                        "conversion - wrong numerics instead of a crash. Turn the moe family off "
                        "for this platform (ASCEND_PLATFORM_OPS) or drop the engine.moe_method "
                        "binding."
                    )
                # The two members the base class and the runner read.
                self.experts_cls = _Experts()
                self.moe_kernel = _MoEKernel(compute)
                logger.info(
                    "ascend-platform OOT MoE: using %r for the routed experts (backend OOT)",
                    getattr(compute, "__module__", "?"),
                )

            def maybe_make_prepare_finalize(self, routing_tables: Any = None) -> None:
                """No all2all prepare/finalize on OOT (the base returns ``None`` in that case)."""
                return None

            @property
            def supports_eplb(self) -> bool:
                """No EPLB: the stand-in kernel has no expert-load-balancing interface."""
                return False

            def process_weights_after_loading(self, layer: Any) -> None:
                """Upstream's OOT early return, then prove nobody replaced our kernel.

                ``unquantized_fused_moe_method.py:217-222`` returns for ``OOT`` *before*
                ``_setup_kernel``, so no weight shuffle happens.  If a future vLLM drops that branch
                the inherited body would build a real kernel over converted weights while our
                compute still expects the checkpoint layout - the error below names that cause.
                """
                super().process_weights_after_loading(layer)
                if not isinstance(self.moe_kernel, _MoEKernel):
                    raise RuntimeError(
                        "ascend-platform OOT MoE: process_weights_after_loading replaced the "
                        f"stand-in kernel with {type(self.moe_kernel).__name__!r}; upstream no "
                        "longer takes the OOT early return "
                        "(unquantized_fused_moe_method.py:217-222), so the weights are now in a "
                        "converted layout our compute does not read."
                    )

        AscendUnquantizedFusedMoEMethod.__name__ = OOT_CLASS_NAME
        AscendUnquantizedFusedMoEMethod.__qualname__ = OOT_CLASS_NAME
        return AscendUnquantizedFusedMoEMethod

    factory.__doc__ = (
        "The ``engine.moe_method`` factory for vLLM 0.23, wrapping "
        f"{getattr(compute, '__module__', '?')}.{getattr(compute, '__name__', '?')}."
    )
    return factory
