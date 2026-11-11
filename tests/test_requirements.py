"""What a family requires, in semantic terms, with the mandatory/optional split and its consequence.

The property that matters is ``OpsReport.servable``: a family can be missing an *optional* operator
(``chunk_local_cumsum`` is declined on purpose) and still run, or be missing a *mandatory* one and
fail on the first forward.  Before this file, both printed as "10 bound" and nothing in the platform
could tell them apart, which is the same class of error as the original "0 bound" bug: a number
without a verdict.

These tests are pure data plus a fake report - no vLLM, no operators, no device.
"""

from __future__ import annotations

import pytest

from ascend_platform.contracts.vocabulary import validate
from ascend_platform.ops import requirements as req
from ascend_platform.ops.install import OpsReport


def test_every_requirement_is_a_declared_semantic_name():
    declared = [r.semantic for family in req.FAMILY_REQUIREMENTS for r in req.requirements(family)]
    assert req.unknown_requirements(declared) == (), (
        "a requirement outside contracts/vocabulary.py cannot be checked against a provider"
    )
    assert validate(declared) == ()


def test_the_gdn_family_cannot_run_without_the_recurrence_and_runs_without_the_fused_cumsum():
    assert "gdn.chunk_gated_delta_rule" in req.mandatory("gdn")
    assert "gdn.chunk_local_cumsum" in req.optional("gdn"), (
        "the ported chunked rule computes the in-chunk cumsum itself; making it mandatory would "
        "refuse to serve a model that runs correctly today"
    )
    assert not set(req.mandatory("gdn")) & set(req.optional("gdn"))


def test_the_moe_family_requires_the_class_vLLM_instantiates():
    assert "engine.moe_method" in req.mandatory("moe"), (
        "without the OOT method class vLLM has no routed-expert implementation at all"
    )
    assert "moe.swiglu_gate" in req.optional("moe"), "only the opt-in dense family uses it"


def test_classify_ignores_switched_off_families():
    missing_mandatory, missing_optional = req.classify(
        bound=[], families={"gdn": False, "moe": False}
    )
    assert (missing_mandatory, missing_optional) == ([], []), (
        "a family that is switched off is not a gap: reporting it would make the default "
        "configuration look broken"
    )


def test_classify_separates_mandatory_from_optional():
    bound = sorted(set(req.mandatory("gdn")) - {"gdn.l2norm"})
    missing_mandatory, missing_optional = req.classify(bound=bound, families={"gdn": True})
    assert missing_mandatory == ["gdn.l2norm"]
    assert missing_optional == ["gdn.chunk_local_cumsum"]


def test_a_report_with_a_missing_mandatory_op_says_it_cannot_serve():
    report = OpsReport(families={"gdn": True})
    report.missing_mandatory = ["gdn.chunk_gated_delta_rule"]
    assert report.servable is False
    assert "CANNOT SERVE" in report.summary()
    assert "gdn.chunk_gated_delta_rule" in report.summary()


def test_a_report_that_only_misses_an_optional_op_is_servable():
    report = OpsReport(families={"gdn": True})
    report.missing_optional = ["gdn.chunk_local_cumsum"]
    assert report.servable is True, (
        "a declined-by-design operator must not make the platform refuse a model it can serve"
    )
    assert "CANNOT SERVE" not in report.summary()


@pytest.mark.parametrize("family", sorted(req.FAMILY_REQUIREMENTS))
def test_each_declared_requirement_carries_a_reason(family):
    without_reason = [r.semantic for r in req.requirements(family) if not r.why]
    assert not without_reason, (
        "a requirement without a 'why' is a number nobody can argue with; the next person to move "
        f"it needs the sentence: {without_reason}"
    )


# ---------------------------------------------------------------------------------------------
# The verdict turns into a boot decision (engine/plugin.py)
# ---------------------------------------------------------------------------------------------


def test_a_complete_install_does_not_stop_the_boot():
    from ascend_platform.engine.plugin import _enforce_requirements

    _enforce_requirements({"servable": True, "missing_mandatory": [], "layer_present": True})


def test_a_missing_mandatory_op_stops_the_boot_and_names_the_ways_out():
    from ascend_platform.engine.plugin import OperatorRequirementsUnmet, _enforce_requirements

    with pytest.raises(OperatorRequirementsUnmet) as excinfo:
        _enforce_requirements(
            {"servable": False, "missing_mandatory": ["moe.experts"], "layer_present": True}
        )
    message = str(excinfo.value)
    for fragment in ("moe.experts", "ASCEND_PLATFORM_OPS", "doctor", "cannot run on Ascend"):
        assert fragment in message, f"{fragment!r} missing from the refusal message"


def test_no_operator_layer_at_all_boots_in_diagnostic_mode(caplog):
    """``ASCEND_PLATFORM_OPS`` unset and nothing installed: boot so ``doctor`` can explain."""
    from ascend_platform.engine.plugin import _enforce_requirements

    with caplog.at_level("WARNING", logger="ascend_platform.engine.plugin"):
        _enforce_requirements(
            {"servable": False, "missing_mandatory": ["moe.experts"], "layer_present": False}
        )
    messages = [record.getMessage() for record in caplog.records]
    assert any("without an operator layer" in message for message in messages), messages


def test_the_refusal_is_not_swallowed_by_the_blanket_handler(monkeypatch):
    """``install()`` catches ``Exception`` for every other step - this one must escape it.

    Without this test the ordering is invisible: the same code with the raise inside the ``try``
    logs "install() failed; continuing" and returns ``False``, i.e. the platform boots and dies on
    the first forward, which is the exact failure mode the verdict exists to prevent.
    """
    from ascend_platform.engine import plugin

    class _Report:
        def summary(self) -> str:
            return "patch registry: 1 applied"

    monkeypatch.setattr(plugin, "_STATE", {"installed": False, "reason": "", "context": None})
    monkeypatch.setattr(
        plugin,
        "_vllm_probe",
        lambda: {"available": True, "version": "0.23.0", "supported": True, "detail": "vllm 0.23.0"},
    )
    monkeypatch.setattr(plugin, "_install_patches", lambda strict: _Report())
    monkeypatch.setattr(plugin, "_install_backend", lambda: {"detail": "backend ok"})
    monkeypatch.setattr(plugin, "_install_graph_bridge", lambda: {"detail": "graph ok"})
    monkeypatch.setattr(
        plugin,
        "_install_ops",
        lambda: {
            "detail": "ops: 0 bound",
            "servable": False,
            "missing_mandatory": ["gdn.chunk_gated_delta_rule"],
            "layer_present": True,
            "bound": [],
            "unbound": [],
            "disabled": [],
        },
    )
    monkeypatch.setattr(plugin, "_install_models", lambda: {"detail": "models ok"})

    with pytest.raises(plugin.OperatorRequirementsUnmet):
        plugin.install()
    assert plugin._STATE["installed"] is False, (
        "a process that refused to boot must not cache itself as installed, or a retry after "
        "fixing the layer becomes a no-op"
    )
