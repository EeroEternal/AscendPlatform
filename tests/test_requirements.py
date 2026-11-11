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
