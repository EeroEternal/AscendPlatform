"""Every knob the code reads is in ``docs/env-vars.md``, and every knob in it is read.

The failure this prevents is quiet in both directions:

* a documented variable that nothing reads (``TILEINFER_METRICS_PATH`` today) — the operator
  exports it, sees no effect, and concludes the feature is broken;
* a variable the code reads but nobody documented (``TILEINFER_PHASE_TIMING``, ...) — the person
  holding the card during the hardware gate has no way to find the instrument that would answer
  their question, because it only appears in a module docstring.

The allowlist is the honest third case: strings that are *provenance* (the source repo's environment
at measurement time, recorded in ``docs/hardware-facts.md``) or vLLM's own switches.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "ascend_platform"
REFERENCE = ROOT / "docs" / "env-vars.md"

#: ``TILEINFER_*`` / ``ASCEND_PLATFORM_*`` / ``VLLM_*`` / ``TILELANG_*`` / ``ASCEND_*``, as a whole word.
ENV_NAME = re.compile(
    r"\b((?:TILEINFER|ASCEND_PLATFORM|ASCEND_RT|ASCEND_OPP|ASCEND_TOOLKIT|VLLM|TILELANG)_[A-Z0-9_]+)\b"
)

#: Not knobs of this package: recorded from the source repo's environment, or owned by vLLM.
PROVENANCE = {
    "TILEINFER_ASCEND_TARGET",
    "TILEINFER_SIMD_SOFTMAX",
    "TILEINFER_DENSE_NN",
    "TILEINFER_ALLOW_SPLIT_KV",
    "TILEINFER_SKIP_SPLIT_COMPILE_HANG",
    "TILEINFER_METRICS_PATH",
    "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
}


def _code_names() -> dict[str, str]:
    """``name -> first file:line`` for every env-var-shaped string literal under ``src/``."""
    found: dict[str, str] = {}
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if ENV_NAME.fullmatch(node.value):
                    found.setdefault(node.value, f"{path.relative_to(ROOT)}:{node.lineno}")
    return found


def _documented_names() -> set[str]:
    return set(ENV_NAME.findall(REFERENCE.read_text(encoding="utf-8")))


def test_the_scan_is_not_vacuous() -> None:
    assert len(_code_names()) >= 20, "the AST scan stopped finding variables; fix the scanner"
    assert len(_documented_names()) >= 20, "the reference table stopped listing variables"


def test_every_variable_the_code_reads_is_in_the_reference() -> None:
    missing = {name: where for name, where in _code_names().items() if name not in _documented_names()}
    assert not missing, (
        "these variables are read by the code but missing from docs/env-vars.md "
        "(if one is provenance, add it to PROVENANCE and to §3 of the reference):\n  "
        + "\n  ".join(f"{name}  ({where})" for name, where in sorted(missing.items()))
    )


def test_every_documented_variable_is_read_or_declared_provenance() -> None:
    code = set(_code_names())
    orphans = sorted(_documented_names() - code - PROVENANCE)
    assert not orphans, (
        "docs/env-vars.md documents variables that nothing in src/ reads and that are not declared "
        f"provenance: {orphans}"
    )


def test_a_provenance_entry_still_appears_in_the_reference() -> None:
    """The allowlist is a claim about the doc: it says "this name is explained in §3"."""
    documented = _documented_names()
    missing = sorted(PROVENANCE - documented)
    assert not missing, f"PROVENANCE names no longer mentioned in docs/env-vars.md: {missing}"


@pytest.mark.parametrize(
    "name",
    ["TILEINFER_OPS", "ASCEND_PLATFORM_OPS", "VLLM_DSV41", "TILEINFER_PHASE_TIMING"],
)
def test_the_names_the_hardware_gate_uses_are_documented(name: str) -> None:
    assert name in _documented_names()
