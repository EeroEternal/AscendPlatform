"""Name rebinding, as the platform owns it: the switch, the candidate gate, the binding table.

What this file deliberately does **not** test is any operator's arithmetic.  Two things are platform
semantics and are tested here:

* *when* a rebound name is allowed to run (:mod:`ascend_platform.ops.switch` — per call, never at
  import time) and *which* candidate wins (:mod:`ascend_platform.ops.candidates` — the recorded
  ratio is part of the gate, not a comment);
* *what* the binding table says (names, targets, aliases, and the reason each takeover exists).

The end-to-end mechanism — actually writing the names into vLLM's modules and rolling back — needs a
real operator implementation to have something to write, and is covered by
``tests/test_boot_rehearsal.py::test_the_gdn_state_update_is_our_ported_packed_decode_not_the_ssu_backend``
when one is installed.
"""

from __future__ import annotations

import pytest

from ascend_platform.contracts.ports import operator_available
from ascend_platform.ops import candidates as candidates_mod
from ascend_platform.ops import families, install as install_mod, switch

def BH(*args, **kwargs):
    """A distinguishable callable: never called, only compared by identity."""
    return None


# ---------------------------------------------------------------------------------------------
# The switch: per call, never at import
# ---------------------------------------------------------------------------------------------


def test_the_switch_is_family_level_and_defaults_to_the_two_the_model_needs(monkeypatch):
    monkeypatch.delenv(switch.ENV_OPS, raising=False)
    enabled, unknown = switch.parse_spec()
    assert enabled == set(switch.OPS_DEFAULT_ON) == {"gdn", "moe"}
    assert unknown == (), (
        "unlike the legacy all-on switch, the default is the two families the model cannot run "
        "without; the third (dense) is opt-in"
    )
    assert "dense" not in enabled


def test_all_selects_every_family_and_a_list_selects_a_subset(monkeypatch):
    monkeypatch.setenv(switch.ENV_OPS, ",".join(switch.OPS_ALL))
    assert switch.parse_spec()[0] == set(switch.OPS_ALL)
    assert switch.ops_enabled("dense")

    monkeypatch.setenv(switch.ENV_OPS, "gdn")
    assert switch.ops_enabled("gdn") and not switch.ops_enabled("moe")


def test_the_switch_is_family_level_not_operator_level():
    """``TILEINFER_OPS`` names families; an operator name is a configuration error, not a no-op."""
    with pytest.raises(ValueError, match="not a takeover family"):
        switch.ops_enabled("l2norm_fwd", spec="gdn")


def test_an_unknown_name_is_fatal_for_the_strict_reader_and_safe_for_the_tolerant_one(
    monkeypatch, caplog
):
    """A typo must be visible, but must not take a runnable engine down."""
    with pytest.raises(ValueError, match="unknown ops"):
        switch.ops_enabled("gdn", spec="not_a_family")

    monkeypatch.setenv(switch.ENV_OPS, "not_a_family")
    assert switch.enabled_families() == set(switch.OPS_DEFAULT_ON), (
        "falling back to the default set keeps the engine runnable; an empty set would disable "
        "every takeover and the model would fail on the first decode"
    )
    assert any("unknown ops" in record.message for record in caplog.records)


def test_the_switch_is_read_per_call_so_a_restart_is_not_needed(monkeypatch):
    monkeypatch.setenv(switch.ENV_OPS, "gdn")
    assert switch.ops_enabled("gdn") is True
    monkeypatch.setenv(switch.ENV_OPS, "dense")
    assert switch.ops_enabled("gdn") is False, (
        "the switch is consulted per call; binding it at import would make this a restart"
    )


def test_describe_reports_the_raw_and_the_effective_decision(monkeypatch):
    monkeypatch.setenv(switch.ENV_OPS, "gdn")
    described = switch.describe()
    assert described["enabled"] == ["gdn"]
    assert described["raw"] == "gdn"


# ---------------------------------------------------------------------------------------------
# The candidate gate: the recorded ratio is part of the decision
# ---------------------------------------------------------------------------------------------


def _baseline():
    return candidates_mod.Candidate("baseline", BH, candidates_mod.always, enabled_by_default=True)


def _candidate(name="cand", ratio=None, enabled=True, predicate=None):
    return candidates_mod.Candidate(
        name,
        lambda *a, **k: "candidate",
        predicate or (lambda call: True),
        enabled_by_default=enabled,
        measured_ratio=ratio,
    )


def test_a_candidate_is_eligible_only_if_enabled_and_not_below_the_threshold():
    assert _candidate(ratio=None).eligible() is True
    assert _candidate(ratio=1.2).eligible() is True
    assert _candidate(ratio=0.86).eligible() is False, "a recorded 0.86x may not run by default"
    assert _candidate(ratio=1.2, enabled=False).eligible() is False
    assert _candidate(ratio=0.5).eligible(threshold=0.4) is True, "the threshold is a parameter"


def test_the_baseline_runs_when_no_candidate_qualifies(monkeypatch):
    monkeypatch.setenv(candidates_mod.ENV_CANDIDATES, "on")
    chosen = candidates_mod.CandidateSet("op", _baseline())
    chosen.add(_candidate(ratio=0.86))
    assert chosen.select(candidates_mod.CallSpec((), {}, "op")) is chosen.baseline.impl


def test_the_first_eligible_candidate_wins_and_order_is_priority(monkeypatch):
    monkeypatch.setenv(candidates_mod.ENV_CANDIDATES, "on")
    chosen = candidates_mod.CandidateSet("op", _baseline())
    first = chosen.add(_candidate("first", ratio=1.1))
    chosen.add(_candidate("second", ratio=1.5))
    assert chosen.select_candidate(candidates_mod.CallSpec((), {}, "op")) is first


def test_the_global_kill_switch_falls_back_to_the_baseline(monkeypatch):
    monkeypatch.setenv(candidates_mod.ENV_CANDIDATES, "off")
    chosen = candidates_mod.CandidateSet("op", _baseline())
    chosen.add(_candidate(ratio=2.0))
    assert chosen.select(candidates_mod.CallSpec((), {}, "op")) is chosen.baseline.impl


def test_a_set_with_no_candidates_never_consults_the_environment(monkeypatch):
    monkeypatch.setenv(candidates_mod.ENV_CANDIDATES, "off")
    chosen = candidates_mod.CandidateSet("op", _baseline())
    assert chosen.select(candidates_mod.CallSpec((), {}, "op")) is chosen.baseline.impl
    assert len(chosen) == 1


def test_a_raising_predicate_is_treated_as_no_match_and_warns_once(monkeypatch, caplog):
    monkeypatch.setenv(candidates_mod.ENV_CANDIDATES, "on")

    def boom(call):
        raise RuntimeError("predicate bug")

    chosen = candidates_mod.CandidateSet("op", _baseline())
    chosen.add(_candidate("broken", ratio=1.1, predicate=boom))
    spec = candidates_mod.CallSpec((), {}, "op")
    assert chosen.select(spec) is chosen.baseline.impl
    assert chosen.select(spec) is chosen.baseline.impl
    assert sum("predicate raised" in record.message for record in caplog.records) == 1


def test_a_duplicate_name_is_refused_because_names_identify_the_report():
    chosen = candidates_mod.CandidateSet("op", _baseline())
    chosen.add(_candidate("same"))
    with pytest.raises(ValueError, match="duplicate candidate name"):
        chosen.add(_candidate("same"))
    with pytest.raises(ValueError, match="already the baseline"):
        chosen.add(_candidate("baseline"))


def test_the_dispatched_wrapper_forwards_the_original_call_unchanged(monkeypatch):
    monkeypatch.setenv(candidates_mod.ENV_CANDIDATES, "off")
    seen = {}

    def baseline(x, *, y=None):
        seen["call"] = (x, y)
        return "baseline"

    chosen = candidates_mod.CandidateSet(
        "op", candidates_mod.Candidate("base", baseline, candidates_mod.always, True)
    )
    dispatched = chosen.dispatched()
    assert dispatched(3, y=4) == "baseline"
    assert seen["call"] == (3, 4), "the wrapper must not rewrite the vLLM ABI"
    assert getattr(dispatched, "__tileinfer_baseline__", None) is baseline


# ---------------------------------------------------------------------------------------------
# The table's data: names, targets, and the reason each takeover exists
# ---------------------------------------------------------------------------------------------


@pytest.mark.skipif(not operator_available(), reason="the binding table needs an operator layer")
def test_every_binding_names_a_vllm_target_and_justifies_itself():
    records = install_mod.bindings()
    assert records, "the family table is empty"
    names = [record.op for record in records]
    assert len(names) == len(set(names)), f"duplicate op names in the table: {names}"
    for record in records:
        assert record.targets, f"{record.op}: no vLLM target"
        for target in record.targets:
            assert target.startswith("vllm."), f"{record.op}: target {target!r} is not a vLLM path"
        for alias in getattr(record, "aliases", ()):
            assert alias.startswith("vllm."), f"{record.op}: alias {alias!r} is not a vLLM path"
            assert alias not in record.targets, f"{record.op}: alias repeats a target"
        assert record.why.strip(), f"{record.op}: no reason recorded"
        if getattr(record, "impl", None) is None:
            continue  # a reported miss (declined on purpose) has nothing to remove later
        assert record.remove_when, f"{record.op}: no removal condition recorded"


@pytest.mark.skipif(operator_available(), reason="an operator layer is installed here")
def test_without_an_operator_layer_the_report_names_the_candidates_and_the_fix():
    """The failure mode matters: a missing layer must name the candidates and the fix, not traceback.

    ``required_names()`` is *data* now and answers without a layer - the requirement list is what
    the platform asks for, not something it learns by importing.  The canonical sentence moved to
    the install report, which is where a reader meets it (banner, ``doctor``), and the per-op
    reasons stay per-op: 13 unbound entries that each say "the operator layer has no ..." are a
    worse answer than one sentence naming the candidates, and counting them cannot tell you
    whether the layer is absent or half-installed.
    """
    from ascend_platform.contracts.ports import OperatorLayerUnavailable

    assert len(install_mod.required_names()) == 13, "the requirement list must not need a layer"

    with pytest.raises(OperatorLayerUnavailable) as excinfo:
        from ascend_platform.contracts.ports import operator_layer

        operator_layer(required=True)
    message = str(excinfo.value)
    assert "ascend_attention" in message and "ASCEND_PLATFORM_OPS" in message

    report = install_mod.install_ops()
    try:
        assert report.layer_error, "a missing layer must be recorded, not inferred from 13 unbound"
        assert "ascend_attention" in report.layer_error
        assert report.bound == []
    finally:
        install_mod._reset()


@pytest.mark.skipif(not operator_available(), reason="the binding table needs an operator layer")
def test_the_required_names_are_the_documented_set():
    names = install_mod.required_names()
    assert len(names) == len(set(names))
    assert "l2norm_fwd" in names
    assert "fused_recurrent_gated_delta_rule_packed_decode" in names


def test_a_version_mismatch_binds_nothing_and_says_why(monkeypatch):
    """The gate that turns "wrong numbers three hours in" into "a refusal at start-up".

    ``install_ops`` binds vLLM's dispatch to the operator layer's callables; if that layer speaks a
    different contract, every binding is a guess.  So the version is checked *before* the first one,
    and a mismatch must leave ``bound`` empty with the reason in ``layer_error`` - the same field the
    missing-layer case uses, so ``doctor`` and the log line already report it.
    """
    from ascend_platform.contracts.version import IncompatibleVersion

    if not operator_available():
        pytest.skip("the gate only runs when a layer resolved; nothing to gate here")

    install_mod._reset()

    def refuse(*args, **kwargs):
        raise IncompatibleVersion(
            "operator layer 'tileinfer': version 9.9.9 is outside the supported range >=0.1,<0.2"
        )

    monkeypatch.setattr(install_mod, "check_operator_version", refuse)
    report = install_mod.install_ops()
    install_mod._reset()

    assert report.bound == [], "a mismatched layer must not receive a single binding"
    assert report.operator_version == ""
    assert "outside the supported range" in report.layer_error
    assert "9.9.9" in report.layer_error
    assert "unavailable" in report.summary() and "9.9.9" in report.summary()


def test_a_matching_version_is_recorded_in_the_report():
    """The positive half: a healthy install records the version it verified."""
    if not operator_available():
        pytest.skip("needs an operator layer")
    install_mod._reset()
    report = install_mod.install_ops()
    install_mod._reset()
    assert report.layer_error == ""
    assert report.operator_version, "the version verified at start-up belongs in the report"
    assert report.bound, "a healthy layer binds something"
