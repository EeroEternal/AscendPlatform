"""The hook an out-of-tree model package registers itself through — and the promise it never raises.

This repository is a *platform plugin*, not a model zoo (README, ``docs/architecture.md`` §3): the
model implementations live in another repository, and vLLM only learns about them through the
``vllm.general_plugins`` entry point, which runs in **every** process, including the one that will
later fail to load a checkpoint.  Two properties follow, and both are asserted here:

* a provider that is not installed is a silent skip, not a boot failure — a general plugin that
  raises takes the whole engine down before any log line explains why;
* the parser is forgiving about the environment variable's shape (spaces, trailing comma, omitted
  entry point) because it is typed by a human into a shell profile.

The module also owns the ``register_config`` shim lookup.  vLLM reads ``config.json`` through
``transformers.AutoConfig`` *before* it looks at ``architectures``, so a checkpoint whose
``model_type`` is newer than the installed transformers needs its config class taught first.  The
shim path is derived from the provider's module name, which is convention-heavy and worth pinning.
"""

from __future__ import annotations

import sys
import types

import pytest

from ascend_platform.engine import model_providers as mp


def test_the_environment_variable_is_the_documented_knob():
    assert mp.MODEL_PROVIDERS_ENV == "ASCEND_PLATFORM_MODELS"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("pkg.models.dsv41:register_models", (("pkg.models.dsv41", "register_models"),)),
        ("a.b:c, d.e", (("a.b", "c"), ("d.e", "register_models"))),
        ("  a.b:c  ,  ,d.e:f,", (("a.b", "c"), ("d.e", "f"))),
        ("", ()),
        ("   ", ()),
    ],
)
def test_the_parser_accepts_what_a_human_types(raw, expected):
    assert mp._parse(raw) == expected


def test_an_uninstalled_provider_is_dropped_not_raised(monkeypatch):
    monkeypatch.setenv(mp.MODEL_PROVIDERS_ENV, "json:register_models,definitely_not_a_module:go")
    assert mp.model_providers() == (("json", "register_models"),)


def test_the_default_provider_is_used_only_when_the_variable_is_unset(monkeypatch):
    """Unset means "the migration-era default"; set-but-empty means the same, by the ``raw`` check."""
    monkeypatch.delenv(mp.MODEL_PROVIDERS_ENV, raising=False)
    resolved = mp.model_providers()
    for module, entry in resolved:
        assert (module, entry) in mp.DEFAULT_MODEL_PROVIDERS
    for module, entry in mp.DEFAULT_MODEL_PROVIDERS:
        if (module, entry) in resolved:
            continue
        with pytest.raises(ImportError):
            __import__(module)  # it is absent because it is not installed, not because of a filter


def _install_fake_provider(monkeypatch, root: str, *, config_body=None):
    """A provider ``root.models.x`` plus an optional ``root.config`` shim, both in ``sys.modules``."""
    package_parts = root.split(".")
    for depth in range(1, len(package_parts) + 1):
        name = ".".join(package_parts[:depth])
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    models = types.ModuleType(root + ".models")
    models.register_models = lambda: {"registered": ["Fake"]}
    monkeypatch.setitem(sys.modules, root + ".models", models)
    provider = types.ModuleType(root + ".models.fake")
    monkeypatch.setitem(sys.modules, root + ".models.fake", provider)
    if config_body is not None:
        config = types.ModuleType(root + ".config")
        config.register_config = config_body
        monkeypatch.setitem(sys.modules, root + ".config", config)


def test_the_shim_is_looked_up_at_the_package_root(monkeypatch):
    """``pkg.models.fake`` -> shim ``pkg.config``: the convention vLLM's own model packages follow."""
    called = []

    def register_config():
        called.append(True)
        return {"config": "faketype"}

    _install_fake_provider(monkeypatch, "pkg", config_body=register_config)
    monkeypatch.setenv(mp.MODEL_PROVIDERS_ENV, "pkg.models.fake:register_models")
    assert mp.register_configs() == "faketype"
    assert called == [True]


def test_a_missing_shim_is_a_skip_and_the_status_says_so(monkeypatch):
    """The provider is installed but ships no ``config`` module: one status, no exception."""

    _install_fake_provider(monkeypatch, "pkg2", config_body=None)
    monkeypatch.setenv(mp.MODEL_PROVIDERS_ENV, "pkg2.models.fake:register_models")
    status = mp.register_configs()
    assert status.startswith("skipped (")
    assert "ModuleNotFoundError" in status


def test_a_raising_shim_is_named_not_propagated(monkeypatch):
    """A broken model package must not take down the engine's boot path."""
    def register_config():
        raise RuntimeError("boom")

    _install_fake_provider(monkeypatch, "pkg3", config_body=register_config)
    monkeypatch.setenv(mp.MODEL_PROVIDERS_ENV, "pkg3.models.fake:register_models")
    status = mp.register_configs()
    assert status.startswith("skipped (")
    assert "RuntimeError" in status


def test_no_providers_is_a_status_not_an_error(monkeypatch):
    monkeypatch.setenv(mp.MODEL_PROVIDERS_ENV, "definitely_not_a_module:go")
    assert mp.model_providers() == ()
    assert mp.register_configs() == "no provider"
