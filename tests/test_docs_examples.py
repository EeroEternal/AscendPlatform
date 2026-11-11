"""Examples in the docs are executed, not admired.

Two kinds of drift are silent by construction:

* a **version range** written in three places (the validator, ``pyproject.toml`` and
  ``docs/versioning.md``) — bumping one and forgetting the others means the package refuses a vLLM it
  documents as supported, or claims to support one it rejects;
* a **schema example** in a design doc — the doc is what an operator-layer author copies, so if our
  own validator rejects our own example, the first manifest anyone writes fails.

Both are cheap to check and expensive to discover on the box.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from ascend_platform.contracts import manifest as manifest_mod
from ascend_platform.contracts.version import SUPPORTED_VLLM_RANGE

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
VERSIONING = ROOT / "docs" / "versioning.md"
OP_MANIFEST = ROOT / "docs" / "op-manifest.md"


# ---------------------------------------------------------------------------------------------
# the version range lives in one place, quoted in two
# ---------------------------------------------------------------------------------------------


def test_the_vllm_range_is_the_same_in_code_pyproject_and_docs() -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    assert f"vllm{SUPPORTED_VLLM_RANGE}" in pyproject, (
        f"pyproject.toml does not pin vllm{SUPPORTED_VLLM_RANGE}; the extra and the start-up check "
        "would disagree"
    )
    assert SUPPORTED_VLLM_RANGE in VERSIONING.read_text(encoding="utf-8"), (
        f"docs/versioning.md does not quote {SUPPORTED_VLLM_RANGE!r}"
    )


def test_the_manifest_schema_version_is_the_one_the_docs_promise() -> None:
    text = OP_MANIFEST.read_text(encoding="utf-8")
    assert f'"schema": {manifest_mod.MANIFEST_SCHEMA_VERSION}' in text, (
        "docs/op-manifest.md shows a manifest whose schema number is not "
        f"{manifest_mod.MANIFEST_SCHEMA_VERSION}"
    )


# ---------------------------------------------------------------------------------------------
# the documented manifest is a manifest this package accepts
# ---------------------------------------------------------------------------------------------

JSON_BLOCK = re.compile(r"```json\n(?P<body>.*?)\n```", re.S)


def _documented_manifests() -> list[tuple[int, dict]]:
    text = OP_MANIFEST.read_text(encoding="utf-8")
    found = []
    for match in JSON_BLOCK.finditer(text):
        lineno = text[: match.start()].count("\n") + 1
        try:
            found.append((lineno, json.loads(match.group("body"))))
        except json.JSONDecodeError as exc:  # pragma: no cover - a broken example is the failure
            pytest.fail(f"docs/op-manifest.md:{lineno} is not valid JSON: {exc}")
    return found


DOCUMENTED = _documented_manifests()


def test_the_doc_has_a_manifest_example_to_check() -> None:
    assert DOCUMENTED, "docs/op-manifest.md no longer contains a json example"


@pytest.mark.parametrize("lineno,payload", DOCUMENTED, ids=[f"line{n}" for n, _ in DOCUMENTED])
def test_the_documented_manifest_is_accepted_by_our_own_validator(lineno: int, payload: dict) -> None:
    parsed = manifest_mod.parse_manifest(payload)
    assert parsed.library, "the example must name the library that ships the ops"
    assert parsed.version, "the example must carry the library version"
    assert parsed.names(), "the example declares no op"
    for entry in parsed.ops:
        assert entry.bench, f"docs/op-manifest.md:{lineno}: {entry.name} has no bench"
        assert entry.measured_ratio and entry.measured_ratio > 0


@pytest.mark.parametrize("lineno,payload", DOCUMENTED, ids=[f"line{n}" for n, _ in DOCUMENTED])
def test_the_documented_manifest_passes_the_compatibility_check(lineno: int, payload: dict) -> None:
    parsed = manifest_mod.parse_manifest(payload)
    # Raises ManifestError on refusal; returning is passing.
    manifest_mod.check_compatibility(parsed, platform_version="0.1.0")
    manifest_mod.check_compatibility(parsed, platform_version="0.1.99")
