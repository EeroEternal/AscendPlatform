"""The two contracts as data: version ranges, and the operator manifest.

The promise in README §"三个契约" is that a mismatch is a *start-up failure with a readable reason*,
not a wrong number discovered three hours into a run.  These tests are that promise: every rejected
input asserts on the message, because the message is the product.
"""

from __future__ import annotations

import json

import pytest

from ascend_platform.contracts.manifest import (
    MANIFEST_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    check_compatibility,
    load_manifest,
    load_manifest_text,
    parse_manifest,
)
from ascend_platform.contracts.version import (
    SUPPORTED_OPERATOR_RANGE,
    SUPPORTED_VLLM_RANGE,
    IncompatibleVersion,
    InvalidVersion,
    VersionRange,
    check_range,
)
from ascend_platform.device import DeviceProfile

BH = {"name": "hbm", "expected": "1"}


# ---------------------------------------------------------------------------------------------
# Version ranges
# ---------------------------------------------------------------------------------------------


def test_a_range_admits_the_versions_inside_it():
    rng = VersionRange.parse(">=0.23,<0.24")
    assert rng.contains("0.23.0")
    assert rng.contains("0.23.9")
    assert not rng.contains("0.24.0")
    assert not rng.contains("0.22.9")


def test_a_suffix_does_not_change_the_ordering():
    rng = VersionRange.parse(">=1.2.3")
    assert rng.contains("1.2.3rc1")
    assert rng.contains("1.2.3.post4")
    assert rng.contains("1.2.3.dev0")
    assert not rng.contains("1.2.2")


def test_the_exact_and_not_equal_clauses_work():
    assert VersionRange.parse("==0.23.0").contains("0.23.0")
    assert not VersionRange.parse("==0.23.0").contains("0.23.1")
    assert VersionRange.parse("!=0.23.1").contains("0.23.0")
    assert not VersionRange.parse("!=0.23.1").contains("0.23.1")


@pytest.mark.parametrize("spec", ["", "   ", "not-a-range", "~=0.23", ">=0.23,foo"])
def test_a_malformed_range_is_rejected(spec):
    with pytest.raises(InvalidVersion):
        VersionRange.parse(spec)


def test_a_trailing_comma_in_a_range_is_tolerated():
    """Hand-written ranges pick up trailing commas; tolerating one is cheaper than a failed boot."""
    assert VersionRange.parse(">=0.23,<0.24,").contains("0.23.0")


@pytest.mark.parametrize("version", ["", "abc", "0.23.0.1", "1.2.3.beta4"])
def test_a_malformed_version_is_rejected(version):
    with pytest.raises(InvalidVersion):
        VersionRange.parse(">=0.23").contains(version)


def test_check_range_passes_a_compatible_version_through():
    assert check_range("0.23.0", ">=0.23,<0.24", what="vLLM").contains("0.23.0")


def test_check_range_names_the_mismatch_and_the_pinned_range():
    with pytest.raises(IncompatibleVersion) as excinfo:
        check_range("0.24.1", SUPPORTED_VLLM_RANGE, what="vLLM")
    message = str(excinfo.value)
    assert "0.24.1" in message and SUPPORTED_VLLM_RANGE in message
    assert "refusing to start" in message


def test_an_undeclared_version_is_a_refusal_not_a_benefit_of_the_doubt():
    with pytest.raises(IncompatibleVersion, match="no version declared"):
        check_range(None, ">=0.1", what="operator layer")


def test_the_declared_engine_range_is_the_023_line():
    assert SUPPORTED_VLLM_RANGE == ">=0.23,<0.24"


# ---------------------------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------------------------


def _op(**overrides):
    op = {
        "name": "paged_decode_attention",
        "shapes": {"head_dim": [128, 256], "group": ">=2"},
        "dtypes": ["bfloat16", "float16"],
        "kv_layouts": ["ND"],
        "graph_capturable": True,
        "measured_ratio": 1.42,
        "bench": "benchmarks/bench_attention.py",
    }
    op.update(overrides)
    return op


def _manifest(**overrides):
    data = {
        "schema": MANIFEST_SCHEMA_VERSION,
        "library": {"name": "ascend-attention", "version": "0.3.0",
                    "requires_platform": ">=0.1,<0.2"},
        "ops": [_op()],
    }
    data.update(overrides)
    return data


def test_a_valid_manifest_parses_into_typed_entries():
    manifest = parse_manifest(_manifest())
    assert manifest.library == "ascend-attention"
    assert manifest.version == "0.3.0"
    assert manifest.requires_platform == ">=0.1,<0.2"
    assert manifest.names() == ("paged_decode_attention",)
    entry = manifest.op("paged_decode_attention")
    assert entry.graph_capturable is True
    assert entry.measured_ratio == 1.42
    assert manifest.capturable() == (entry,)
    assert entry.as_dict()["shapes"] == {"head_dim": [128, 256], "group": ">=2"}


def test_the_shipped_json_schema_describes_the_same_shape():
    """The schema is a file other repositories can validate against; it must not drift."""
    assert MANIFEST_SCHEMA["properties"]["schema"]["const"] == MANIFEST_SCHEMA_VERSION
    required = set(MANIFEST_SCHEMA["properties"]["ops"]["items"]["required"])
    assert required == {"name", "dtypes", "kv_layouts", "graph_capturable"}
    assert set(MANIFEST_SCHEMA["required"]) == {"schema", "library", "ops"}


@pytest.mark.parametrize(
    "data, message",
    [
        ({}, "missing required field 'schema'"),
        ({"schema": 2}, "schema 2 is not supported"),
        (_manifest(library="not-an-object"), "library must be an object"),
        (_manifest(library={"name": "x"}), "missing required field 'version'"),
        (_manifest(ops="nope"), "ops must be an array"),
        (_manifest(ops=[{}]), "missing required field 'name'"),
        (_manifest(ops=[_op(), _op()]), "duplicate op name"),
        (_manifest(ops=[_op(dtypes=[])]), "dtypes must not be empty"),
        (_manifest(ops=[_op(kv_layouts=[])]), "kv_layouts must not be empty"),
        (_manifest(ops=[_op(graph_capturable="yes")]), "graph_capturable must be a bool"),
        (_manifest(ops=[_op(measured_ratio=None)]), "needs both measured_ratio and bench"),
        (_manifest(ops=[_op(bench=None)]), "needs both measured_ratio and bench"),
        (_manifest(ops=[_op(measured_ratio=0)]), "measured_ratio must be > 0"),
        (_manifest(ops=[_op(shapes=[])]), "shapes must be an object"),
    ],
)
def test_every_violation_is_rejected_with_the_op_named(data, message):
    with pytest.raises(ManifestError, match=message):
        parse_manifest(data)


def test_an_unmeasured_op_may_not_be_advertised():
    """Repo rule: no ``measured_ratio`` and no bench ⇒ it may not appear in the manifest at all."""
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(_manifest(ops=[_op(measured_ratio=None, bench=None)]))
    assert "no ratio, no mainline" in str(excinfo.value)


def test_the_schema_is_json_serialisable_so_it_can_be_shipped_as_a_file():
    assert json.loads(json.dumps(MANIFEST_SCHEMA)) == MANIFEST_SCHEMA


def test_a_manifest_loads_from_text_and_from_a_file(tmp_path):
    text = json.dumps(_manifest())
    assert load_manifest_text(text).library == "ascend-attention"
    path = tmp_path / "manifest.json"
    path.write_text(text, encoding="utf-8")
    assert load_manifest(str(path)).library == "ascend-attention"


def test_invalid_json_is_reported_as_a_manifest_error():
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest_text("{nope")


# ---------------------------------------------------------------------------------------------
# Compatibility at start-up
# ---------------------------------------------------------------------------------------------


def _device(**overrides):
    base = dict(
        name="Ascend950PR",
        kernel_page_size=128,
        engine_page_multiple=2048,
        dtypes=("bfloat16", "float16"),
        kv_cache_dtypes=("auto",),
        layouts=("ND", "NZ"),
    )
    base.update(overrides)
    return DeviceProfile(**base)


def test_a_compatible_manifest_is_accepted():
    check_compatibility(parse_manifest(_manifest()), platform_version="0.1.0.dev0",
                        device=_device())


def test_a_manifest_that_wants_a_newer_platform_is_refused():
    with pytest.raises(IncompatibleVersion, match="outside the supported range"):
        check_compatibility(parse_manifest(_manifest()), platform_version="0.2.0",
                            device=_device())


def test_an_op_whose_dtypes_the_device_lacks_is_refused():
    manifest = parse_manifest(_manifest(ops=[_op(dtypes=["float8_e4m3"])]))
    with pytest.raises(ManifestError) as excinfo:
        check_compatibility(manifest, platform_version="0.1.0.dev0", device=_device())
    message = str(excinfo.value)
    assert "paged_decode_attention" in message and "float8_e4m3" in message


def test_an_op_whose_kv_layout_the_device_lacks_is_refused():
    manifest = parse_manifest(_manifest(ops=[_op(kv_layouts=["FRACTAL"])]))
    with pytest.raises(ManifestError, match="KV layouts"):
        check_compatibility(manifest, platform_version="0.1.0.dev0", device=_device())


def test_compatibility_can_be_checked_without_a_device():
    check_compatibility(parse_manifest(_manifest()), platform_version="0.1.0.dev0")


# ---------------------------------------------------------------------------------------------
# Real-world version strings: the gate must not fail over a cosmetic suffix
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version",
    ["0.23.0", "0.23.0rc1", "0.23.0.dev0", "0.23.0+cu123", "v0.23.0", " 0.23.0 "],
)
def test_a_version_that_really_occurs_on_a_box_is_admitted(version):
    """``torch``/``vLLM`` report ``+cpu``/``+cu123`` builds and a git install adds ``+git1234``.

    The suffix is not a different release line; refusing to start over it would be a false negative
    at the moment the box is being brought up.
    """
    assert check_range(version, SUPPORTED_VLLM_RANGE, what="vLLM").contains(version)


def test_an_operator_layer_installed_from_git_is_admitted():
    """The source repo installs as ``0.1.0.dev0``; a checkout adds ``+git<hash>``."""
    assert check_range("0.1.0.dev0+git1234", SUPPORTED_OPERATOR_RANGE, what="operator layer")


@pytest.mark.parametrize("version", ["0.23.x", "twenty-three", "0.23.0-1"])
def test_an_unreadable_installed_version_is_a_mismatch_not_a_crash(version):
    """``install_ops`` and ``doctor`` catch ``IncompatibleVersion``; anything else escapes as a bug.

    A ``ValueError`` here would crash start-up with "cannot parse version" and be reported by
    ``doctor`` as a defect in this package, for something the operator can actually fix.
    """
    with pytest.raises(IncompatibleVersion) as excinfo:
        check_range(version, SUPPORTED_VLLM_RANGE, what="vLLM")
    message = str(excinfo.value)
    assert version in message and "cannot read version" in message
    assert "refusing to start" in message


def test_a_malformed_requirement_is_still_our_bug():
    """The range is a constant in this repo; a typo in it may raise ``InvalidVersion`` (a ValueError)."""
    with pytest.raises(InvalidVersion):
        check_range("0.23.0", ">=0.23,~=0.24", what="vLLM")


def test_a_manifest_with_an_unreadable_platform_range_is_a_manifest_error():
    """The range is the operator repo's data; a bad one must not surface as a ValueError crash.

    ``install_ops`` and ``doctor`` catch ``ManifestError``/``IncompatibleVersion``; letting
    ``InvalidVersion`` escape would turn another repository's typo into a traceback here.
    """
    from ascend_platform.contracts.manifest import ManifestError

    body = _manifest()
    body["library"] = dict(body["library"], requires_platform="~=0.1")
    with pytest.raises(ManifestError) as excinfo:
        parse_manifest(body)
    assert "requires_platform" in str(excinfo.value)
    assert "docs/op-manifest.md" in str(excinfo.value)
