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


def test_the_paths_match_what_the_family_table_binds_today():
    """The table is a *view* of the binding, so a drift between them is caught here first."""
    engine_name_of = {
        "topk_softmax": "moe.routing_topk_softmax",
        "moe_align_block_size": "moe.align_block_size",
        "silu_and_mul": "moe.swiglu_gate",
        "fused_experts": "moe.experts",
        "chunk_gated_delta_rule": "gdn.chunk_gated_delta_rule",
        "chunk_local_cumsum": "gdn.chunk_local_cumsum",
        "l2norm_fwd": "gdn.l2norm",
        "fused_sigmoid_gating_delta_rule_update": "gdn.fused_sigmoid_gating",
        "fused_recurrent_gated_delta_rule_packed_decode": "gdn.recurrent_decode",
        "fused_post_conv_prep": "gdn.post_conv_prep",
        "causal_conv1d_fn": "gdn.conv1d_prefill",
        "causal_conv1d_update": "gdn.conv1d_update",
    }
    mismatches = []
    for record in gdn_moe.bind():
        semantic = engine_name_of.get(record.op)
        if semantic is None:
            continue
        bound = record.targets[0]
        if abi.ENGINE_SYMBOL[semantic] != bound:
            mismatches.append(f"{record.op}: abi={abi.ENGINE_SYMBOL[semantic]!r} table={bound!r}")
    assert not mismatches, mismatches


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
