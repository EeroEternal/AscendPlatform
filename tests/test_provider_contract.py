"""The provider contract: a package announces what it serves, and the platform stops guessing paths.

What this file pins, and why each one is a *failure mode* rather than a nicety:

* a descriptor is **data** - name, version, platform range, per-op targets - and a malformed one is
  refused with a message about the offending field, not a ``KeyError`` three frames down;
* discovery has exactly two sources (the pinned root, then the entry-point group) and the pinned
  root wins, because "I pointed the platform at this package" is an explicit instruction;
* a broken provider degrades the **boot** to "no provider" and never to a traceback - the same
  discipline ``install_ops`` follows for a missing operator layer;
* the family table asks the provider **first** and the legacy module layout second, so a package
  with a descriptor is used without the platform learning its directory structure, and a package
  without one keeps working.

The last test is the one that would have caught the original bug: a provider that serves twelve of
thirteen names must leave the other twelve bound.
"""

from __future__ import annotations

import sys
import types

import pytest

from ascend_platform.contracts import provider as provider_mod
from ascend_platform.contracts.manifest import ManifestError
from ascend_platform.contracts.provider import (
    PROVIDER_GROUP,
    ProviderDescriptor,
    ProviderError,
    ProviderOp,
    discover,
    parse_descriptor,
    provider,
    reset_provider,
)
from ascend_platform.contracts.version import IncompatibleVersion

DESCRIPTOR = {
    "name": "fake-ops",
    "version": "0.1.0",
    "requires_platform": ">=0.1,<0.2",
    "ops": {
        "topk_softmax": "fake_ops.moe:topk_softmax",
        "fused_experts": {
            "target": "fake_ops.moe:fused_experts",
            "dtypes": ["bfloat16"],
            "graph": "eager_only",
            "note": "torch baseline",
        },
    },
}


@pytest.fixture(autouse=True)
def _clean_discovery(monkeypatch):
    """No provider leaks between tests: the cache and the pinned variable are both reset."""
    monkeypatch.delenv(provider_mod.__dict__.get("ENV_OPS_LAYER", "ASCEND_PLATFORM_OPS"), raising=False)
    reset_provider()
    from ascend_platform.contracts.ports import reset_operator_layer

    reset_operator_layer()
    yield
    reset_provider()


def _install_fake_module(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


# ---------------------------------------------------------------------------------------------
# Parsing: the descriptor is data, and bad data has a name
# ---------------------------------------------------------------------------------------------


def test_a_well_formed_descriptor_parses_both_shapes():
    as_mapping = parse_descriptor(DESCRIPTOR)
    as_list = parse_descriptor(
        {
            "name": "fake-ops",
            "version": "0.1.0",
            "requires_platform": ">=0.1,<0.2",
            "ops": [
                {"name": "topk_softmax", "target": "fake_ops.moe:topk_softmax"},
                {"name": "fused_experts", "target": "fake_ops.moe:fused_experts"},
            ],
        }
    )
    assert set(as_mapping.op_names()) == {"topk_softmax", "fused_experts"}
    assert set(as_list.op_names()) == set(as_mapping.op_names())
    assert as_mapping.ops["fused_experts"].graph == "eager_only"
    assert as_mapping.summary().startswith("provider fake-ops 0.1.0")


def test_a_descriptor_without_a_name_or_version_is_refused_by_field():
    with pytest.raises(ProviderError, match="name"):
        parse_descriptor({"version": "0.1.0"})
    with pytest.raises(ProviderError, match="version"):
        parse_descriptor({"name": "x"})


def test_a_target_that_is_not_module_colon_attribute_is_refused():
    with pytest.raises(ProviderError, match="module:attribute"):
        parse_descriptor({"name": "x", "version": "0.1.0", "ops": {"op": "nowhere"}})


def test_a_provider_outside_the_platform_range_is_refused_with_both_numbers():
    with pytest.raises(IncompatibleVersion) as excinfo:
        parse_descriptor({"name": "x", "version": "9.9.9", "requires_platform": ">=0.1,<0.2"})
    message = str(excinfo.value)
    assert "9.9.9" in message and "0.2" in message


def test_a_provider_error_is_a_manifest_error_so_the_taxonomy_is_one_thing():
    assert issubclass(ProviderError, ManifestError)


# ---------------------------------------------------------------------------------------------
# Discovery: pinned root first, entry point second
# ---------------------------------------------------------------------------------------------


def test_the_pinned_root_is_asked_first_and_wins(monkeypatch):
    _install_fake_module("fake_ops_pkg", describe=lambda: DESCRIPTOR)
    monkeypatch.setenv("ASCEND_PLATFORM_OPS", "fake_ops_pkg")

    found = discover()
    assert found is not None and found.name == "fake-ops"
    assert found.source == "ASCEND_PLATFORM_OPS=fake_ops_pkg"


def test_the_resolved_layer_is_asked_even_without_the_pinned_variable(monkeypatch):
    """``ascend_attention`` wins ``OPERATOR_ROOTS`` with nothing set - its descriptor must be found."""
    _install_fake_module("auto_layer", describe=lambda: DESCRIPTOR)
    monkeypatch.setattr(
        "ascend_platform.contracts.ports.operator_layer",
        lambda required=True: types.SimpleNamespace(root="auto_layer"),
    )
    found = discover()
    assert found is not None and found.source == "operator layer auto_layer"


def test_a_layer_without_a_descriptor_is_not_a_provider(monkeypatch):
    """``tileinfer`` and ``ascend_attention`` today: the layout path still serves them."""
    _install_fake_module("plain_layer")
    monkeypatch.setenv("ASCEND_PLATFORM_OPS", "plain_layer")
    assert discover() is None


def test_a_broken_pinned_descriptor_degrades_to_no_provider(monkeypatch):
    _install_fake_module("broken_ops_pkg", describe=lambda: {"name": "broken"})
    monkeypatch.setenv("ASCEND_PLATFORM_OPS", "broken_ops_pkg")
    assert discover() is None, "a bad descriptor must not stop the boot"


def test_the_entry_point_group_is_the_second_source(monkeypatch):
    class _Entry:
        name = "from-metadata"

        def load(self):
            return lambda: DESCRIPTOR

    def fake_entry_points(group=None, **kwargs):
        assert group == PROVIDER_GROUP
        return [_Entry()]

    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)
    found = discover()
    assert found is not None and found.source == "entry point from-metadata"


def test_a_broken_entry_point_does_not_hide_a_working_one(monkeypatch):
    class _Broken:
        name = "broken"

        def load(self):
            raise ImportError("libmissing.so")

    class _Good:
        name = "good"

        def load(self):
            return lambda: DESCRIPTOR

    monkeypatch.setattr(
        "importlib.metadata.entry_points", lambda group=None, **kw: [_Broken(), _Good()]
    )
    found = discover()
    assert found is not None and found.name == "fake-ops"


# ---------------------------------------------------------------------------------------------
# Loading: one op at a time, and the miss is a sentence
# ---------------------------------------------------------------------------------------------


def test_load_returns_the_callable_and_names_the_miss(monkeypatch):
    def topk_softmax(*args, **kwargs):  # pragma: no cover - identity only
        return None

    _install_fake_module("fake_ops.moe", topk_softmax=topk_softmax)
    sys.modules["fake_ops"] = _install_fake_module("fake_ops")
    _install_fake_module("fake_ops_pkg", describe=lambda: DESCRIPTOR)
    monkeypatch.setenv("ASCEND_PLATFORM_OPS", "fake_ops_pkg")

    loaded = provider()
    assert loaded is not None
    impl, reason = loaded.load("topk_softmax")
    assert impl is topk_softmax and reason == ""

    missing, why = loaded.load("silu_and_mul")
    assert missing is None
    assert "does not serve 'silu_and_mul'" in why


def test_a_target_that_does_not_import_is_reported_against_that_op(monkeypatch):
    _install_fake_module(
        "half_ops_pkg",
        describe=lambda: {
            "name": "half",
            "version": "0.1.0",
            "ops": {"present": "actually_missing.module:fn", "absent": "nope:fn"},
        },
    )
    monkeypatch.setenv("ASCEND_PLATFORM_OPS", "half_ops_pkg")
    loaded = provider()
    assert loaded is not None
    impl, why = loaded.load("present")
    assert impl is None and "cannot import actually_missing.module" in why


# ---------------------------------------------------------------------------------------------
# The wiring that would have caught the original bug
# ---------------------------------------------------------------------------------------------


def test_the_family_table_prefers_the_provider_over_the_operator_layout(monkeypatch):
    """A provider's callable wins by identity; names it does not serve fall back to the layout."""
    sentinel = lambda *a, **k: None  # noqa: E731 - identity only

    _install_fake_module("prov_layer.ops.moe", topk_softmax=sentinel)
    _install_fake_module("prov_layer.ops", )
    _install_fake_module(
        "prov_layer",
        describe=lambda: {
            "name": "prov",
            "version": "0.1.0",
            "ops": {"moe.routing_topk_softmax": "prov_layer.ops.moe:topk_softmax"},
        },
    )
    monkeypatch.setenv("ASCEND_PLATFORM_OPS", "prov_layer")

    from ascend_platform.contracts.ports import reset_operator_layer
    from ascend_platform.ops.families import gdn_moe

    reset_provider()
    reset_operator_layer()
    try:
        resolver = gdn_moe._Impl()  # noqa: SLF001 - the resolver is the unit under test
        impl, note = resolver.impl("moe.routing_topk_softmax", "moe", "topk_softmax")
        assert impl is sentinel and note == "", "the provider's target must win over the layout path"

        # ``moe_align_block_size`` is not served by this provider: the layout path still answers,
        # and a genuinely missing name is a note rather than an exception.
        missing, why = resolver.impl("moe.align_block_size", "moe", "moe_align_block_size")
        assert missing is None
        assert "does not serve 'moe.align_block_size'" in why
    finally:
        reset_provider()
        reset_operator_layer()


def test_a_provider_descriptor_is_json_able_for_the_banner():
    from ascend_platform.contracts.provider import describe_provider

    assert describe_provider()["available"] is False
    assert ProviderDescriptor(name="x", version="0.1.0").ops == {}
    assert ProviderOp(name="x", target="m:t").resolve.__name__ == "resolve"


def test_the_moe_class_takeover_never_takes_a_plain_compute_function(monkeypatch):
    """A descriptor alias may point ``unquantized_fused_moe`` at plain compute - that is not a factory.

    ``kind="custom_op_oot"`` binds ``impl(base_cls) -> subclass``; a function that expects tensors
    would be *bound* and fail at the first model instantiation, far from the cause.  So that record
    is resolved with ``ask_provider=False`` and stays unbound until the engine-ABI wrapper exists.
    """
    _install_fake_module("alias_ops", describe=lambda: {
        "name": "alias",
        "version": "0.1.0",
        "ops": {"unquantized_fused_moe": "alias_ops.moe:unquantized_fused_moe"},
    })
    _install_fake_module("alias_ops.moe", unquantized_fused_moe=lambda *a, **k: None)
    monkeypatch.setenv("ASCEND_PLATFORM_OPS", "alias_ops")

    from ascend_platform.contracts.ports import reset_operator_layer
    from ascend_platform.ops.families import gdn_moe

    reset_provider()
    reset_operator_layer()
    try:
        record = next(b for b in gdn_moe.bind() if b.op == "unquantized_fused_moe")
        assert record.impl is None, "a compute function must not be bound as a class factory"
        assert "fused_moe_oot" in record.impl_note
    finally:
        reset_provider()
        reset_operator_layer()


# ---------------------------------------------------------------------------------------------
# The protocol form: resolve(name) instead of a table
# ---------------------------------------------------------------------------------------------


def _fake_layer(tmp_path, monkeypatch, *, describe: str = "", resolve: str = "") -> str:
    """An importable package with the parts asked for, and nothing else.  Returns its name."""
    root_name = f"userops_{abs(hash((describe, resolve))) % 100000}"
    root = tmp_path / root_name
    root.mkdir()
    body = [
        '"""A user package: no platform import, no engine import."""',
        "",
        "def handle(x):",
        "    return x",
        "",
        "__all__ = ['handle']",
    ]
    if describe:
        body += ["", "def describe():", f"    {describe}"]
    if resolve:
        body += ["", "def resolve(name):", f"    {resolve}"]
    (root / "__init__.py").write_text("\n".join(body) + "\n", encoding="utf-8")

    import sys

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(root_name, None)
    from ascend_platform.contracts import ports, provider

    monkeypatch.setattr(ports, "OPERATOR_ROOTS", (root_name,))
    monkeypatch.delenv(ports.ENV_OPS_LAYER, raising=False)
    provider.reset_provider()
    return root_name


def test_a_package_can_answer_per_name_without_enumerating(tmp_path, monkeypatch):
    """``resolve(name)`` is the whole contract: one function, no vocabulary to copy."""
    name = _fake_layer(
        tmp_path,
        monkeypatch,
        resolve="return handle if name.startswith('moe.') else None",
    )
    from ascend_platform.contracts.provider import discover, provider

    descriptor = discover()
    assert descriptor is not None
    assert descriptor.name == name
    assert descriptor.ops == {}, "a protocol-only package enumerates nothing"
    assert "no ops" in descriptor.summary()

    serving = provider()
    impl, reason = serving.load("moe.experts")
    assert callable(impl) and reason == ""
    assert serving.load("gdn.l2norm") == (
        None,
        f"provider {name!r}: resolve('gdn.l2norm') returned None",
    )


def test_describe_wins_per_name_and_resolve_fills_the_gaps(tmp_path, monkeypatch):
    """Both forms at once: the declared table is authoritative, the hook answers the rest."""
    name = _fake_layer(
        tmp_path,
        monkeypatch,
        describe='return {"name": "userops", "version": "1.0", "ops": {"moe.experts": f"{__name__}:handle"}}',
        resolve="return handle if name == 'moe.align_block_size' else None",
    )
    from ascend_platform.contracts.provider import discover, provider

    descriptor = discover()
    assert descriptor.ops["moe.experts"].target == f"{name}:handle"
    serving = provider()
    assert serving.load("moe.experts")[0] is not None
    assert callable(serving.load("moe.align_block_size")[0])


def test_a_resolver_that_raises_is_a_reason_not_a_crash(tmp_path, monkeypatch):
    name = _fake_layer(tmp_path, monkeypatch, resolve="raise RuntimeError('boom')")
    from ascend_platform.contracts.provider import provider

    impl, reason = provider().load("moe.experts")
    assert impl is None
    assert "resolve('moe.experts') raised RuntimeError('boom')" in reason


def test_a_package_with_neither_form_is_not_a_provider(tmp_path, monkeypatch):
    _fake_layer(tmp_path, monkeypatch)
    from ascend_platform.contracts.provider import discover

    assert discover() is None
