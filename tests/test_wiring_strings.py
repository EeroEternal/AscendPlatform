"""The dotted paths the engine is told to load must point at something that exists.

vLLM loads our worker and attention backend **by string**: ``parallel_config.worker_cls`` and the
``attention_backend_cls`` slot are dotted paths, and the entry points in ``pyproject.toml`` are
``module:attribute``.  A typo in one of those strings is invisible to the test suite as written,
because the assertions compare the config against *the same constant* (``worker_cls ==
WORKER_CLASS_PATH``) — both sides move together and both are wrong.

Two ways this goes wrong in practice, both of which happened during this migration:

* the package was renamed, so every path needs rewriting (done, and audited by
  ``test_migration_parity.py``'s module-level layer);
* a path is *edited* later (``worker.py`` -> ``workers.py``) and only the constant is updated in one
  of the two places it is defined.

So the strings are resolved here, statically (no vLLM, no torch, no device): the module half must be
a file in this package, and the attribute half must be defined in it.
"""

from __future__ import annotations

import ast
import pathlib
import re
from typing import Dict, List, Tuple

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "ascend_platform"

#: Names whose value the engine (or pip) treats as a load path.
WIRING_NAME = re.compile(r"(_CLASS_PATH|_ENTRY_POINT|_MODULE|worker_cls|attention_backend_cls)$")


def _module_file(dotted: str) -> pathlib.Path | None:
    """``ascend_platform.engine.worker`` -> the file that defines it (or None)."""
    parts = dotted.split(".")
    if parts[0] != "ascend_platform":
        return None
    relative = PKG.joinpath(*parts[1:])
    module = relative.with_suffix(".py")
    if module.is_file():
        return module
    package = relative / "__init__.py"
    return package if package.is_file() else None


def _defines(path: pathlib.Path, name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
    return False


def _wiring_constants() -> Dict[str, Tuple[str, pathlib.Path]]:
    found: Dict[str, Tuple[str, pathlib.Path]] = {}
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                continue
            if not isinstance(node.value.value, str):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and WIRING_NAME.search(target.id):
                    found[f"{path.relative_to(ROOT)}::{target.id}"] = (node.value.value, path)
    return found


WIRING = _wiring_constants()


def test_the_scan_found_the_wiring_constants() -> None:
    assert len(WIRING) >= 6, f"only {len(WIRING)} wiring constants found; the scan is broken"
    values = " ".join(value for value, _path in WIRING.values())
    assert "NPUWorker" in values and "TileInferPlatform" in values


@pytest.mark.parametrize("key", sorted(WIRING), ids=sorted(WIRING))
def test_a_wiring_string_points_at_a_real_module(key: str) -> None:
    value, origin = WIRING[key]
    module_part = value.split(":")[0] if ":" in value else value.rsplit(".", 1)[0]
    assert _module_file(module_part) is not None, (
        f"{key} = {value!r} (in {origin.relative_to(ROOT)}) names the module {module_part!r}, "
        "which is not a file in this package"
    )


@pytest.mark.parametrize("key", sorted(WIRING), ids=sorted(WIRING))
def test_a_wiring_string_points_at_a_defined_attribute(key: str) -> None:
    value, origin = WIRING[key]
    if _module_file(value) is not None:
        return  # a bare module path (``WORKER_MODULE``): the module check above is the whole story
    if ":" in value:
        module_part, attribute = value.split(":", 1)
    else:
        module_part, attribute = value.rsplit(".", 1)
    module = _module_file(module_part)
    assert module is not None, f"{key} names an unknown module {module_part!r}"
    assert _defines(module, attribute), (
        f"{key} = {value!r} (in {origin.relative_to(ROOT)}) names {attribute!r}, which "
        f"{module.relative_to(ROOT)} does not define"
    )


def test_the_entry_points_in_pyproject_resolve_too() -> None:
    """The same rule for ``pyproject.toml``, which is what vLLM actually reads."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    values = re.findall(r'"_?([\w.]+:[\w.]+)"', text)
    values = [v for v in values if v.startswith("ascend_platform")]
    assert values, "pyproject.toml declares no ascend_platform entry point"
    for value in values:
        module_part, attribute = value.split(":", 1)
        module = _module_file(module_part)
        assert module is not None, f"pyproject entry point {value!r} names an unknown module"
        assert _defines(module, attribute), f"pyproject entry point {value!r}: {attribute!r} undefined"
