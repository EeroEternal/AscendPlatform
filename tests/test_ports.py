"""The operator port: lazy, swappable, and loud when absent.

The single most important property is at the bottom of this file: ``import ascend_platform`` must
succeed on a machine with no operator layer at all.  Everything above it is the machinery that makes
the absence *explainable* instead of a ``ModuleNotFoundError`` from deep inside an import chain.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from ascend_platform.contracts import ports
from ascend_platform.contracts.ports import (
    ENV_OPS_LAYER,
    OPERATOR_ROOTS,
    OperatorLayerUnavailable,
    operator_available,
    operator_info,
    operator_layer,
    operator_root,
    register_operator_layout,
    reset_operator_layer,
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_port():
    """Every test starts from "nothing resolved" and leaves the process as it found it."""
    reset_operator_layer()
    yield
    reset_operator_layer()


@pytest.fixture
def fake_layer(tmp_path, monkeypatch):
    """A minimal operator layer on ``sys.path`` with the two symbols the port proxies."""
    package = tmp_path / "fake_operator_layer"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "9.9.9"\n', encoding="utf-8")
    (package / "api.py").write_text(
        textwrap.dedent(
            """
            RECORD = {}

            class BatchAttention:
                def __init__(self, *args, **kwargs):
                    RECORD["args"] = args
                    RECORD["kwargs"] = kwargs
                    self.built = True
            """
        ),
        encoding="utf-8",
    )
    for name, body in (
        ("metadata", "class RaggedMetadata:\n    pass\n"),
        ("plan", "class TileSchedule:\n    pass\n\ndef plan_query_tiles_from_host(*a, **k):\n    return 'planned'\n"),
        ("testing", "def reference_moe(*a, **k):\n    return 'oracle'\n"),
    ):
        (package / f"{name}.py").write_text(body, encoding="utf-8")
    kernels = package / "kernels" / "attention"
    kernels.mkdir(parents=True)
    (package / "kernels" / "__init__.py").write_text("", encoding="utf-8")
    (kernels / "__init__.py").write_text("", encoding="utf-8")
    (kernels / "paged_decode_ascend950.py").write_text(
        "def forward_prefill(*a, **k):\n    return 'prefill'\n\n"
        "def _pack_indices(*a, **k):\n    return 'packed'\n",
        encoding="utf-8",
    )
    ops = package / "ops"
    ops.mkdir()
    (ops / "__init__.py").write_text("", encoding="utf-8")
    (ops / "moe.py").write_text("def fused_moe():\n    return 'moe'\n", encoding="utf-8")
    ops.mkdir(exist_ok=True)

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(ENV_OPS_LAYER, "fake_operator_layer")
    register_operator_layout(
        "fake_operator_layer",
        {
            "attention": "api",
            "metadata": "metadata",
            "plan": "plan",
            "decode_kernel": "kernels.attention.paged_decode_ascend950",
            "testing": "testing",
            "ops": "ops",
        },
    )
    reset_operator_layer()
    yield package
    ports._LAYOUTS.pop("fake_operator_layer", None)


def test_nothing_is_resolved_until_something_asks(monkeypatch):
    monkeypatch.delenv(ENV_OPS_LAYER, raising=False)
    assert ports._LAYER is None
    ports.operator_available()
    assert True  # the point is that reaching this line did not raise


def test_the_pinned_layer_is_the_only_candidate(monkeypatch):
    monkeypatch.setenv(ENV_OPS_LAYER, "definitely_not_installed")
    assert operator_available() is False
    with pytest.raises(OperatorLayerUnavailable) as excinfo:
        operator_layer()
    message = str(excinfo.value)
    assert "definitely_not_installed" in message
    assert ENV_OPS_LAYER in message


def test_the_default_candidates_prefer_the_target_layer():
    assert OPERATOR_ROOTS == ("ascend_attention", "tileinfer")


def test_an_absent_layer_raises_with_the_install_hint(monkeypatch):
    monkeypatch.setenv(ENV_OPS_LAYER, "definitely_not_installed")
    with pytest.raises(OperatorLayerUnavailable) as excinfo:
        operator_layer()
    message = str(excinfo.value)
    assert "no operator layer is importable" in message
    assert "ascend_attention" in message
    assert "Last error" in message, "the reason each candidate failed must be reported"


def test_operator_info_is_a_status_dict_not_an_exception(monkeypatch):
    monkeypatch.setenv(ENV_OPS_LAYER, "definitely_not_installed")
    info = operator_info()
    assert info["available"] is False and info["root"] is None
    assert info["tried"] == ["definitely_not_installed"]


def test_a_registered_layout_makes_a_third_party_layer_usable(fake_layer):
    assert operator_available() is True
    assert operator_root() == "fake_operator_layer"
    assert operator_info() == {
        "available": True,
        "root": "fake_operator_layer",
        "version": "9.9.9",
        "tried": ["fake_operator_layer"],
    }


def test_the_proxies_return_the_layer_objects_not_copies(fake_layer):
    assert ports.BatchAttention(1, b=2).built is True
    record = sys.modules["fake_operator_layer.api"].RECORD
    assert record["args"] == (1,)
    assert record["kwargs"] == {"b": 2}
    assert ports.plan_query_tiles_from_host() == "planned"
    assert ports.reference_moe() == "oracle"
    assert ports.forward_prefill() == "prefill"
    assert ports.pack_indices() == "packed"
    assert type(ports.RaggedMetadata()).__name__ == "RaggedMetadata"
    assert type(ports.TileSchedule()).__name__ == "TileSchedule"


def test_a_missing_symbol_names_both_sides(fake_layer):
    with pytest.raises(OperatorLayerUnavailable) as excinfo:
        ports.PageTable()
    message = str(excinfo.value)
    assert "fake_operator_layer" in message and "PageTable" in message
    assert "out of step" in message


def test_ops_module_resolves_a_submodule_of_the_operator_package(fake_layer):
    assert ports.ops_module("moe").fused_moe() == "moe"


def test_a_partial_layout_is_rejected_with_the_missing_names():
    with pytest.raises(ValueError, match="missing"):
        register_operator_layout("another-layer", {"attention": "api"})


def test_a_layout_with_an_unknown_entry_is_rejected():
    with pytest.raises(ValueError, match="unknown entries"):
        register_operator_layout("another-layer", {name: "x" for name in ports._REQUIRED_LOGICAL}
                                 | {"gpu": "x"})


def test_a_broken_layer_import_is_reported_not_swallowed(tmp_path, monkeypatch):
    package = tmp_path / "broken_operator_layer"
    package.mkdir()
    (package / "__init__.py").write_text("raise ImportError('kaboom')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(ENV_OPS_LAYER, "broken_operator_layer")
    with pytest.raises(OperatorLayerUnavailable) as excinfo:
        operator_layer()
    assert "kaboom" in str(excinfo.value)


# ---------------------------------------------------------------------------------------------
# The property the whole design exists for
# ---------------------------------------------------------------------------------------------


def test_the_package_imports_with_no_operator_layer_and_no_vllm():
    """A fresh interpreter with only ``src`` on the path: this is the CI guard on the seam."""
    program = textwrap.dedent(
        """
        import sys
        import ascend_platform
        import ascend_platform.device
        import ascend_platform.contracts.ports as ports

        assert ports.operator_available() is False, "an operator layer is importable here"
        assert ascend_platform.__version__
        # Nothing engine-side may have been imported as a side effect.
        assert "vllm" not in sys.modules, "importing the package pulled in vLLM"
        assert "torch_npu" not in sys.modules, "importing the package pulled in torch_npu"
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin",
             "ASCEND_PLATFORM_OPS": "definitely_not_installed"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# ---------------------------------------------------------------------------------------
# the version contract: two number ranges, checked at start-up
# ---------------------------------------------------------------------------------------


def test_the_layer_version_is_read_from_the_package(fake_layer):
    assert ports.operator_version() == "9.9.9"


def test_an_in_range_layer_passes_the_check(tmp_path, monkeypatch):
    package = tmp_path / "goodop"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "0.1.4"\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(ENV_OPS_LAYER, "goodop")
    register_operator_layout("goodop", {name: "api" for name in ports._REQUIRED_LOGICAL})
    reset_operator_layer()
    try:
        assert ports.check_operator_version() == "0.1.4"
    finally:
        ports._LAYOUTS.pop("goodop", None)


def test_an_out_of_range_layer_is_refused_with_both_numbers(fake_layer):
    """``fake_layer`` declares 9.9.9 on purpose: out of range must refuse, not warn."""
    from ascend_platform.contracts.version import IncompatibleVersion

    with pytest.raises(IncompatibleVersion) as excinfo:
        ports.check_operator_version()
    message = str(excinfo.value)
    assert "9.9.9" in message, "the installed version must be in the message"
    assert ">=0.1,<0.2" in message, "and so must the range this build supports"


def test_a_layer_that_declares_no_version_is_refused_too(tmp_path, monkeypatch):
    """Silence is the one answer that cannot be verified, so it is treated as a mismatch."""
    package = tmp_path / "silentop"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(ENV_OPS_LAYER, "silentop")
    register_operator_layout("silentop", {name: "api" for name in ports._REQUIRED_LOGICAL})
    reset_operator_layer()
    try:
        from ascend_platform.contracts.version import IncompatibleVersion

        with pytest.raises(IncompatibleVersion) as excinfo:
            ports.check_operator_version()
        assert "no version declared" in str(excinfo.value)
    finally:
        ports._LAYOUTS.pop("silentop", None)


def test_no_layer_at_all_is_reported_as_unavailable_not_as_a_version_mismatch(monkeypatch):
    monkeypatch.setenv(ENV_OPS_LAYER, "definitely_not_installed_anywhere")
    reset_operator_layer()
    with pytest.raises(OperatorLayerUnavailable):
        ports.check_operator_version()
    assert ports.operator_version() == "unknown"


def test_a_git_installed_layer_with_a_local_version_is_admitted(tmp_path, monkeypatch):
    """Real installs report ``0.1.0.dev0+git1234``; that is the same line, not a mismatch.

    Covered at the range level too, but this is the call ``install_ops`` makes at start-up: a false
    refusal here stops the box before the first forward, with a message about versions.
    """
    package = tmp_path / "gitop"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "0.1.0.dev0+git1234"\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(ENV_OPS_LAYER, "gitop")
    register_operator_layout("gitop", {name: "api" for name in ports._REQUIRED_LOGICAL})
    reset_operator_layer()
    try:
        assert ports.check_operator_version() == "0.1.0.dev0+git1234"
    finally:
        ports._LAYOUTS.pop("gitop", None)


def test_an_unreadable_layer_version_is_a_mismatch_not_a_crash(tmp_path, monkeypatch):
    """``ValueError`` would escape both call sites, which catch ``IncompatibleVersion`` only."""
    from ascend_platform.contracts.version import IncompatibleVersion

    package = tmp_path / "weirdop"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "banana"\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(ENV_OPS_LAYER, "weirdop")
    register_operator_layout("weirdop", {name: "api" for name in ports._REQUIRED_LOGICAL})
    reset_operator_layer()
    try:
        with pytest.raises(IncompatibleVersion) as excinfo:
            ports.check_operator_version()
        assert "banana" in str(excinfo.value)
    finally:
        ports._LAYOUTS.pop("weirdop", None)
