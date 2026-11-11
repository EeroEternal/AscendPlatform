"""The engine ABI: vLLM 0.23's symbol table, and the class extension point that is not a function.

Two things are pinned here, and they fail in different ways if they rot:

* the **symbol table** - every semantic name either has a vLLM 0.23 path or is *listed* as having
  none; and the paths still match what the family table binds today.  A silent rename on the vLLM
  side then fails here, with both names in the message, instead of at model load;
* the **MoE method class** - the shapes the base class actually reads.  ``experts_cls.is_monolithic``
  is called *with* parentheses and ``moe_kernel.is_monolithic`` is read *without* them; answering
  either wrongly is a silent wrong branch, so both are asserted, and so is ``apply`` reaching the
  injected compute rather than a hardcoded import.

No vLLM is imported by this file: the base class is a stand-in with the same surface, which is why
the whole thing is testable on a CPU box.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
import textwrap
import types

import pytest

from ascend_platform.contracts.vocabulary import BY_NAME, OpKind, SEMANTIC_OPS
from ascend_platform.engine.abi import vllm_0_23 as abi
from ascend_platform.ops.families import gdn_moe


# ---------------------------------------------------------------------------------------------
# The symbol table: complete, non-overlapping, and still true
# ---------------------------------------------------------------------------------------------


def test_every_declared_name_is_classified_exactly_once():
    named = set(abi.ENGINE_SYMBOL) | set(abi.NO_ENGINE_SYMBOL)
    declared = {op.name for op in SEMANTIC_OPS if op.kind is OpKind.COMPUTE}
    assert not (set(abi.ENGINE_SYMBOL) & set(abi.NO_ENGINE_SYMBOL)), (
        "a name cannot both have an engine symbol and be listed as having none"
    )
    assert declared - named == set(), (
        "these vocabulary names are neither mapped to a vLLM 0.23 symbol nor listed as having none: "
        f"{sorted(declared - named)}"
    )
    assert named <= set(BY_NAME), (
        f"the ABI table names semantic ops the vocabulary does not declare: {sorted(named - set(BY_NAME))}"
    )


def test_every_record_declares_a_semantic_name_from_the_vocabulary():
    unknown = [b.op for b in gdn_moe.bind() if b.platform_name not in BY_NAME]
    assert not unknown, (
        "these records carry no (or an undeclared) platform_name, so nothing ties them to the "
        f"engine ABI table: {unknown}"
    )


def test_the_paths_match_what_the_family_table_binds_today():
    """The ABI table is the authority; the record's ``targets`` must agree with it, not drift."""
    mismatches, unchecked = [], []
    for record in gdn_moe.bind():
        published = abi.ENGINE_SYMBOL.get(record.platform_name)
        if published is None:
            unchecked.append(record.platform_name)
            continue
        if published != record.targets[0]:
            mismatches.append(
                f"{record.op}: abi={published!r} binding={record.targets[0]!r}"
            )
    assert not mismatches, mismatches
    assert sorted(unchecked) == ["engine.moe_method", "moe.swiglu_gate"], (
        "two records have no engine symbol on this release: the class-factory slot, and the "
        "activation whose only vLLM 0.23 entry point is a C++ op the OOT path never calls; a "
        f"different set means the table or the records moved: {sorted(unchecked)}"
    )


def test_a_published_symbol_table_is_authoritative_and_a_disagreement_is_drift():
    """Two sources exist during the move; they must agree, and the engine's table wins."""
    from ascend_platform.ops import install as install_mod

    record = next(b for b in gdn_moe.bind() if b.op == "topk_softmax")
    wanted = record.platform_name

    # Publishing the *correct* path changes nothing: it is the same string the record carries.
    try:
        install_mod.set_engine_symbols({wanted: record.targets[0]})
        assert install_mod._targets_of(record) == (record.targets, "")

        # Publishing a different path is reported as drift, and the record is not bound.
        install_mod.set_engine_symbols({wanted: "vllm._custom_ops.moved_elsewhere"})
        targets, drift = install_mod._targets_of(record)
        assert targets == record.targets and "engine ABI drift" in drift
        assert wanted in drift and "moved_elsewhere" in drift
    finally:
        install_mod.set_engine_symbols(None)
    assert install_mod.engine_symbols() == {}


def test_an_unknown_name_says_which_side_to_fix():
    with pytest.raises(KeyError) as excinfo:
        abi.target_for("moe.not_a_thing")
    message = str(excinfo.value)
    assert "ENGINE_SYMBOL" in message and "NO_ENGINE_SYMBOL" in message


def test_importing_the_abi_module_does_not_import_vllm():
    """``doctor`` and the CPU suite import this module; pulling vLLM in would break both."""
    code = textwrap.dedent(
        """
        import sys
        import ascend_platform.engine.abi.vllm_0_23  # noqa: F401
        print("vllm" in sys.modules)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
    )
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip() == "False", "the ABI module must not import vLLM at import time"


# ---------------------------------------------------------------------------------------------
# The MoE method class: a fake base with vLLM's surface
# ---------------------------------------------------------------------------------------------


class FakeBackend:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeUnquantizedFusedMoEMethod:
    """The part of ``UnquantizedFusedMoEMethod`` the subclass touches, in vLLM's shapes."""

    def __init__(self, moe) -> None:
        self.moe = moe
        self.unquantized_backend = FakeBackend("OOT")
        self.experts_cls = None
        self.moe_kernel = None
        self.weight_shuffled = False

    def maybe_make_prepare_finalize(self, routing_tables=None):
        raise AssertionError("the subclass must override this: the inherited body raises")

    @property
    def supports_eplb(self) -> bool:
        return True  # the in-tree default the subclass must refuse

    def process_weights_after_loading(self, layer) -> None:
        if self.unquantized_backend.name == "OOT":
            return  # upstream's early return: no kernel is ever built
        self.weight_shuffled = True
        self.moe_kernel = object()


def _compute(record=None):
    """A routed-expert stand-in; ``record`` (a dict) captures how the kernel called it."""

    def compute(hidden, w1, w2, topk_weights, topk_ids, **kwargs):
        if record is not None:
            record["args"] = (hidden, w1, w2, topk_weights, topk_ids)
            record["kwargs"] = kwargs
        return "mo-out"

    return compute


def test_the_factory_returns_a_subclass_of_the_base_it_is_given():
    cls = abi.moe_method_factory(_compute())(FakeUnquantizedFusedMoEMethod)
    assert issubclass(cls, FakeUnquantizedFusedMoEMethod)
    assert cls.__name__ == abi.OOT_CLASS_NAME


def test_the_two_is_monolithic_shapes_are_both_right():
    cls = abi.moe_method_factory(_compute())(FakeUnquantizedFusedMoEMethod)
    method = cls(moe=object())
    assert method.experts_cls.is_monolithic() is False, (
        "experts_cls.is_monolithic is CALLED (fused_moe_method_base.py:157-158): an attribute that "
        "is merely falsy would raise 'bool is not callable'"
    )
    assert method.moe_kernel.is_monolithic is False, (
        "moe_kernel.is_monolithic is READ (fused_moe_method_base.py:161): a method would be a "
        "truthy bound method and take the monolithic branch"
    )
    assert method.moe_kernel.can_overlap_shared_experts is False
    assert method.moe_kernel.output_is_reduced() is False
    assert method.moe_kernel.prepare_finalize.topk_indices_dtype() is None


def test_apply_reaches_the_injected_compute_and_ignores_the_shared_expert():
    calls: dict = {}
    cls = abi.moe_method_factory(_compute(record=calls))(FakeUnquantizedFusedMoEMethod)
    kernel = cls(moe=object()).moe_kernel
    out = kernel.apply(
        "hidden",
        "w1",
        "w2",
        "topk_weights",
        "topk_ids",
        activation="silu",
        shared_experts=object(),
        shared_experts_input=object(),
    )
    assert out == "mo-out"
    assert calls["args"] == ("hidden", "w1", "w2", "topk_weights", "topk_ids")
    assert calls["kwargs"]["quant_config"] is None
    assert "shared_experts" not in calls["kwargs"], (
        "the runner applies the shared expert itself before this call; passing it through would "
        "double-count the dense expert"
    )


def test_monolithic_and_eplb_and_prepare_finalize_are_refused_explicitly():
    cls = abi.moe_method_factory(_compute())(FakeUnquantizedFusedMoEMethod)
    method = cls(moe=object())
    with pytest.raises(NotImplementedError, match="is_monolithic is False"):
        method.moe_kernel.apply_monolithic()
    assert method.supports_eplb is False, "the stand-in kernel has no EPLB interface"
    assert method.maybe_make_prepare_finalize() is None


def test_a_non_oot_backend_is_refused_at_construction():
    class NonOOT(FakeUnquantizedFusedMoEMethod):
        def __init__(self, moe) -> None:
            super().__init__(moe)
            self.unquantized_backend = FakeBackend("DEEPGEMM")

    cls = abi.moe_method_factory(_compute())(NonOOT)
    with pytest.raises(RuntimeError, match="only implements the OOT backend"):
        cls(moe=object())


def test_process_weights_keeps_the_stand_in_kernel_and_notices_if_upstream_replaces_it():
    cls = abi.moe_method_factory(_compute())(FakeUnquantizedFusedMoEMethod)
    method = cls(moe=object())
    method.process_weights_after_loading(layer=object())  # the OOT early return: nothing happens
    assert method.weight_shuffled is False

    class UpstreamBuildsAKernel(FakeUnquantizedFusedMoEMethod):
        def process_weights_after_loading(self, layer) -> None:
            self.moe_kernel = object()  # a future vLLM that builds a real kernel for OOT

    broken = abi.moe_method_factory(_compute())(UpstreamBuildsAKernel)(moe=object())
    with pytest.raises(RuntimeError, match="replaced the"):
        broken.process_weights_after_loading(layer=object())


def test_the_factory_validates_its_arguments():
    with pytest.raises(TypeError, match="compute must be callable"):
        abi.moe_method_factory("not a function")
    factory = abi.moe_method_factory(_compute())
    with pytest.raises(TypeError, match="UnquantizedFusedMoEMethod class"):
        factory("vllm.path.to.Class")


# ---------------------------------------------------------------------------------------------
# The injection seam: the engine publishes, the family table consumes
# ---------------------------------------------------------------------------------------------


def test_a_published_factory_binds_the_moe_method_slot_and_is_visible_immediately():
    factory = abi.moe_method_factory(_compute())
    # Without a published factory the slot resolves through the other two sources, so this test
    # compares against whatever the environment provides (unbound on a bare box, the operator
    # layer's own ``make_oot_method`` where the migration source is installed) instead of
    # assuming ``None``.
    before = {b.op: b.impl for b in gdn_moe.bind()}["unquantized_fused_moe"]
    try:
        gdn_moe.set_engine_factories({"engine.moe_method": factory})
        published = {b.op: b.impl for b in gdn_moe.bind()}["unquantized_fused_moe"]
        assert published is factory, (
            "an engine-published class factory must win over the layout path: it is the only "
            "source that knows the engine's class shape"
        )
    finally:
        gdn_moe.set_engine_factories({})
    restored = {b.op: b.impl for b in gdn_moe.bind()}["unquantized_fused_moe"]
    assert restored is before, "clearing the registry must restore the previous resolution"


def test_the_vocabulary_declares_the_moe_method_slot_as_a_factory():
    """The kind check that refuses a compute function here depends on this one declaration."""
    assert BY_NAME["engine.moe_method"].kind is OpKind.CLASS_FACTORY
    assert types.FunctionType is not OpKind  # keep ``types`` imported for the fake-module helpers


# ---------------------------------------------------------------------------------------------
# In-place adapters: vLLM's protocol over a functional operator
# ---------------------------------------------------------------------------------------------


class _FakeOps:
    """The operator layer's functional shape, with a log of what it was asked for."""

    def __init__(self):
        self.calls = []

    def topk_softmax(self, scores, top_k, *, renormalize=True):
        self.calls.append(("topk", top_k, renormalize))
        values, indices = torch.softmax(scores.float(), -1).topk(top_k, -1)
        if renormalize:
            values = values / values.sum(-1, keepdim=True)
        return values, indices.to(torch.int32)

    def moe_align_block_size(self, topk_ids, num_experts, block_size, *, num_tokens=None):
        self.calls.append(("align", num_experts, block_size, num_tokens))
        pairs = topk_ids.numel()
        # A deliberately *wider* buffer than vLLM allocates, so the adapter's prefix copy is exercised.
        width = pairs + num_experts * block_size
        sorted_ids = torch.full((width,), pairs, dtype=torch.int32)  # pairs == vLLM's pad sentinel
        sorted_ids[:pairs] = torch.arange(pairs, dtype=torch.int32)
        expert_ids = torch.arange(pairs // block_size + num_experts, dtype=torch.int32)
        return sorted_ids, expert_ids, torch.tensor(num_experts * block_size, dtype=torch.int32)

    def silu_and_mul(self, x):
        self.calls.append(("silu", tuple(x.shape)))
        half = x.shape[-1] // 2
        return torch.nn.functional.silu(x[..., :half]) * x[..., half:]


def test_topk_softmax_adapter_writes_vllms_buffers_in_place():
    """vLLM's router allocates float32 weights and int32 ids, then reads them back (returns None)."""
    from ascend_platform.engine.abi import vllm_0_23

    ops = _FakeOps()
    adapter = vllm_0_23.topk_softmax_inplace(ops.topk_softmax)
    logits = torch.randn(5, 8)
    weights = torch.empty(5, 2, dtype=torch.float32)
    ids = torch.empty(5, 2, dtype=torch.int32)
    token_expert = torch.empty(5, 2, dtype=torch.int32)

    assert adapter(weights, ids, token_expert, logits, True) is None
    values, indices = torch.softmax(logits, -1).topk(2, -1)
    expected = values / values.sum(-1, keepdim=True)
    assert torch.allclose(weights, expected)
    assert torch.equal(ids, indices.to(torch.int32))
    assert torch.equal(token_expert, ids), "the third buffer is filled, not left undefined"
    assert ops.calls == [("topk", 2, True)], "top_k comes from the buffer width and renormalize passes through"


def test_topk_softmax_adapter_refuses_grouped_routing():
    """Ignoring the correction bias would pick the wrong experts - a silent accuracy bug."""
    from ascend_platform.engine.abi import vllm_0_23

    adapter = vllm_0_23.topk_softmax_inplace(_FakeOps().topk_softmax)
    with pytest.raises(NotImplementedError, match="e_score_correction_bias"):
        adapter(
            torch.empty(2, 2, dtype=torch.float32),
            torch.empty(2, 2, dtype=torch.int32),
            torch.empty(2, 2, dtype=torch.int32),
            torch.randn(2, 8),
            False,
            torch.zeros(8),
        )


def test_align_adapter_copies_the_prefix_and_keeps_the_sentinel_tail():
    """vLLM sizes for the maximum and reads only up to ``num_tokens_post_pad``."""
    from ascend_platform.engine.abi import vllm_0_23

    ops = _FakeOps()
    adapter = vllm_0_23.moe_align_block_size_inplace(ops.moe_align_block_size)
    topk_ids = torch.randint(0, 8, (6, 2), dtype=torch.int32)
    sorted_ids = torch.full((80,), -7, dtype=torch.int32)
    expert_ids = torch.full((20,), -7, dtype=torch.int32)
    post_pad = torch.empty(1, dtype=torch.int32)

    assert adapter(topk_ids, 8, 4, sorted_ids, expert_ids, post_pad) is None
    assert int(post_pad) == 32
    width = 6 * 2 + 8 * 4
    assert torch.equal(sorted_ids[:width], torch.tensor(
        list(range(12)) + [12] * (width - 12), dtype=torch.int32
    )), "the operator's whole output is copied, including its own pad sentinel"
    assert torch.equal(sorted_ids[width:], torch.full((80 - width,), -7, dtype=torch.int32)), (
        "beyond the operator's output the caller's buffer is untouched"
    )
    produced = 6 * 2 // 4 + 8  # what the fake returns: pairs // block_size + num_experts
    assert torch.equal(expert_ids[:produced], torch.arange(produced, dtype=torch.int32))
    assert torch.equal(expert_ids[produced:], torch.full((20 - produced,), -1, dtype=torch.int32)), (
        "blocks the operator did not produce must be the -1 sentinel, never stale memory"
    )
    assert ops.calls == [("align", 8, 4, 6)], "num_tokens is the token count, not the pair count"


def test_align_adapter_refuses_expert_parallel():
    from ascend_platform.engine.abi import vllm_0_23

    adapter = vllm_0_23.moe_align_block_size_inplace(_FakeOps().moe_align_block_size)
    with pytest.raises(NotImplementedError, match="expert_map"):
        adapter(
            torch.zeros(2, 2, dtype=torch.int32), 8, 4,
            torch.empty(20, dtype=torch.int32), torch.empty(10, dtype=torch.int32),
            torch.empty(1, dtype=torch.int32), torch.zeros(8, dtype=torch.int32),
        )


def test_silu_and_mul_adapter_writes_the_out_buffer():
    from ascend_platform.engine.abi import vllm_0_23

    adapter = vllm_0_23.silu_and_mul_inplace(_FakeOps().silu_and_mul)
    x = torch.randn(3, 8)
    out = torch.empty(3, 4)
    assert adapter(out, x) is None
    assert torch.allclose(out, torch.nn.functional.silu(x[:, :4]) * x[:, 4:])


def test_inplace_adapters_only_cover_the_symbols_that_exist():
    """A semantic the operator does not serve stays unbound instead of becoming a raising proxy."""
    from ascend_platform.engine.abi import vllm_0_23

    ops = _FakeOps()
    adapters = vllm_0_23.inplace_adapters(
        {"moe.routing_topk_softmax": ops.topk_softmax, "moe.align_block_size": ops.moe_align_block_size}
    )
    assert set(adapters) == {"moe.routing_topk_softmax", "moe.align_block_size"}
    assert set(vllm_0_23.INPLACE_ENGINE_SYMBOL) == {"moe.routing_topk_softmax", "moe.align_block_size"}, (
        "the in-place table lists only symbols this release actually ships; moe.swiglu_gate is "
        "excluded because vLLM 0.23 has no _custom_ops.silu_and_mul wrapper to bind it to"
    )
    assert "moe.swiglu_gate" not in vllm_0_23.ENGINE_SYMBOL
    assert "moe.swiglu_gate" in vllm_0_23.NO_ENGINE_SYMBOL
