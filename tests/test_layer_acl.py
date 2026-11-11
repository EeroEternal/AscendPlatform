"""The directory ACL, enforced: dependency direction, engine-freedom, and the operator seam.

Migrated from the source repo's ``tests/test_layer_deps.py`` and generalised from two layers
(``serving``/``attention``) to the seven this repo declares in :mod:`ascend_platform._acl`.

The property under test is not stylistic.  ``runtime/`` holds the plan cache and the page geometry,
i.e. exactly the code a laptop can test; the day it imports ``vllm``, that testability disappears
*silently*, because the machine it runs on has vLLM installed.  A static rule is the only thing that
notices.  The second half of this file proves the same property dynamically, by importing each
engine-free layer in a clean interpreter and checking ``sys.modules``.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
import textwrap
from typing import Iterable, List, Tuple

import pytest

from ascend_platform import _acl

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ascend_platform"


def _layer_dir(layer: str) -> pathlib.Path:
    return SRC / layer


def _files(layer: str) -> List[pathlib.Path]:
    return sorted(_layer_dir(layer).rglob("*.py"))


ENGINE_FREE_FILES = [path for layer in sorted(_acl.ENGINE_FREE) for path in _files(layer)]
ALL_FILES = [path for layer in _acl.LAYERS for path in _files(layer)]


def _all_imports(tree: ast.Module) -> List[Tuple[str, int]]:
    found: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if base:
                found.append((base, node.lineno))
            for alias in node.names:
                if alias.name:
                    found.append((f"{base}.{alias.name}" if base else alias.name, node.lineno))
    return found


def _module_level_imports(tree: ast.Module) -> List[Tuple[str, int]]:
    """Imports executed at import time (module scope only, not inside a function)."""
    found: List[Tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.module or "", node.lineno))
    return found


def _matches(module: str, forbidden: str) -> bool:
    return module == forbidden or module.startswith(forbidden + ".")


def test_the_acl_covers_every_layer_on_disk():
    on_disk = {path.name for path in SRC.iterdir() if path.is_dir() and path.name in _acl.LAYERS}
    assert on_disk == set(_acl.LAYERS), (
        f"a directory exists that the ACL does not know about: {sorted(on_disk ^ set(_acl.LAYERS))}"
    )
    for layer in _acl.LAYERS:
        assert _files(layer), f"{layer}/ has no python modules"


def test_every_layer_may_import_itself_and_only_declared_siblings():
    for layer in _acl.LAYERS:
        assert layer not in _acl.allowed(layer), (
            f"{layer} does not need to list itself; same-layer imports are always allowed"
        )


def test_the_engine_free_layers_are_the_five_leaves_plus_ops():
    assert _acl.ENGINE_FREE == frozenset({"contracts", "device", "profiles", "runtime", "ops"})
    assert "engine" not in _acl.ENGINE_FREE and "doctor" not in _acl.ENGINE_FREE


@pytest.mark.parametrize("path", ENGINE_FREE_FILES, ids=lambda p: p.relative_to(SRC).as_posix())
def test_an_engine_free_layer_never_imports_the_engine(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = [
        f"{module} (line {line})"
        for module, line in _all_imports(tree)
        if any(_matches(module, root) for root in _acl.FORBIDDEN_IN_ENGINE_FREE)
    ]
    assert not offenders, (
        f"{path.relative_to(ROOT)} imports something an engine-free layer may not: {offenders}. "
        f"Move it into engine/ or behind a Protocol (see ascend_platform/_acl.py)."
    )


@pytest.mark.parametrize("path", ENGINE_FREE_FILES, ids=lambda p: p.relative_to(SRC).as_posix())
def test_an_engine_free_layer_keeps_torch_npu_optional(path: pathlib.Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = [
        f"{module} (line {line})"
        for module, line in _module_level_imports(tree)
        if any(_matches(module, root) for root in _acl.FUNCTION_ONLY_IMPORTS)
    ]
    assert not offenders, (
        f"{path.relative_to(ROOT)} imports torch_npu at module level: {offenders}. "
        "Import it inside the function that probes for a device."
    )


@pytest.mark.parametrize("path", ALL_FILES, ids=lambda p: p.relative_to(SRC).as_posix())
def test_the_dependency_direction_holds(path: pathlib.Path) -> None:
    layer = path.relative_to(SRC).parts[0]
    if layer not in _acl.LAYERS:
        return  # module-level files: _version.py, utils.py, _acl.py
    allowed = _acl.allowed(layer)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = []
    for module in {name for name, _line in _all_imports(tree)}:
        if not module.startswith("ascend_platform."):
            continue
        target = ".".join(module.split(".")[1:3])
        sibling = module.split(".")[1]
        if sibling == layer:
            continue
        if sibling not in _acl.LAYERS:
            continue
        if sibling in allowed or target in allowed:
            continue
        offenders.append(f"{module} (line ?) — {layer} may only import {sorted(allowed)}")
    assert not offenders, f"{path.relative_to(ROOT)}: {offenders}"


def _operator_roots():
    """The port's list, not a copy: the ACL rule is about *which* packages, and the port owns them."""
    from ascend_platform.contracts.ports import OPERATOR_ROOTS

    return OPERATOR_ROOTS


def test_only_the_port_imports_an_operator_package():
    offenders = []
    for path in ALL_FILES:
        relative = path.relative_to(ROOT).as_posix()
        if relative.endswith("contracts/ports.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module, line in _all_imports(tree):
            if any(_matches(module, root) for root in _operator_roots()):
                offenders.append(f"{relative}:{line} -> {module}")
    assert not offenders, (
        f"operator packages may only be reached through {_acl.PORT_MODULE}: {offenders}"
    )


# ---------------------------------------------------------------------------------------------
# The dynamic half: prove the engine-free layers really import cleanly
# ---------------------------------------------------------------------------------------------

_CLEAN_PROGRAM = textwrap.dedent(
    """
    import importlib
    import sys

    layer = sys.argv[1]
    importlib.import_module("ascend_platform." + layer)
    forbidden = ("vllm", "torch_npu", "tilelang", "tileinfer", "ascend_attention")
    leaked = [name for name in forbidden if name in sys.modules]
    assert not leaked, f"{layer} imported {leaked} as a side effect"
    print("clean")
    """
)


@pytest.mark.parametrize("layer", sorted(_acl.ENGINE_FREE))
def test_an_engine_free_layer_imports_in_a_clean_interpreter(layer: str) -> None:
    """Not a mock: a fresh process with only ``src`` on the path, asserting what it pulled in."""
    result = subprocess.run(
        [sys.executable, "-c", _CLEAN_PROGRAM, layer],
        cwd=str(ROOT),
        env={
            "PYTHONPATH": str(ROOT / "src"),
            "PATH": "/usr/bin:/bin",
            "ASCEND_PLATFORM_OPS": "definitely_not_installed",
            # torch auto-loads any installed device backend at import; switching that off is what
            # makes "torch_npu is in sys.modules" mean *our* code imported it.
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_the_acl_module_itself_is_data_only():
    text = (SRC / "_acl.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = {
        name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for name in (alias.name for alias in node.names)
    } | {
        (node.module or "") for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert imported <= {"__future__", "typing"}, (
        f"_acl.py must stay importable with nothing but the standard library; it imports {imported}"
    )


def test_describe_is_the_single_source_the_docs_can_quote():
    rows = _acl.describe()
    assert tuple(row["layer"] for row in rows) == _acl.LAYERS
    assert sum(1 for row in rows if row["engine_free"]) == len(_acl.ENGINE_FREE)
    for row in rows:
        assert isinstance(row["may_import"], list)
