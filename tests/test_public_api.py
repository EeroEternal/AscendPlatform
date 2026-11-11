"""The package is a library as well as a plugin: the surface somebody reads has to be documented.

``architecture.md`` §7 claims every layer directory carries a docstring.  That covers the *layout*;
it says nothing about the 195 public classes and functions in it.  This file checks the surface a
maintainer actually meets — ``help()``, an error message, an IDE popup — because "migrated" and
"explained" are different deliverables, and the two new interfaces (``DeviceProfile``, the operator
manifest) are exactly the ones a second author has to work from.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "ascend_platform"

#: Public names that are documented *at their source*, not here.  Each entry is a function copied
#: verbatim from the source repo, and the parity audit compares token streams — adding a docstring
#: would be an undocumented difference from the code this repository mirrors.  The audit's own
#: allowlist (``EXPECTED_DIFFERENCES``) is the precedent: record, then check.
VERBATIM_UNDOCUMENTED = {
    "engine/compat/patches.py::register": "tileinfer/vllm/compat/patches.py:133",
    "engine/models/req_ids.py::reset": "tileinfer/vllm/models/dsv41_req_ids.py:39",
    "profiles/qwen3_6.py::mamba_page_size_padded": "tileinfer/vllm/profiles/qwen3_6.py:63",
    "utils.py::npu_device_count": "tileinfer/utils.py:53",
    "profiles/base.py::known_profiles": "tileinfer/vllm/profiles/base.py (no docstring there)",
    "profiles/base.py::profile_by_name": "tileinfer/vllm/profiles/base.py (no docstring there)",
}

MIN_DOCSTRING = 12


def _modules():
    return sorted(PKG.rglob("*.py"))


def _relative(path: pathlib.Path) -> str:
    return path.relative_to(PKG).as_posix()


def _public_objects(path: pathlib.Path):
    """Top-level classes and functions whose name does not start with an underscore."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                yield node


def test_the_surface_check_is_not_vacuous() -> None:
    objects = [node for path in _modules() for node in _public_objects(path)]
    assert len(objects) >= 150, f"only {len(objects)} public objects found; the scan is broken"


@pytest.mark.parametrize("key", sorted(VERBATIM_UNDOCUMENTED), ids=lambda k: k.replace("/", "."))
def test_an_undocumented_entry_still_exists_and_still_lacks_a_docstring(key: str) -> None:
    """The allowlist may not rot: a stale entry would silently excuse a *new* undocumented name."""
    module, _, name = key.partition("::")
    path = PKG / module
    assert path.is_file(), f"{key}: {module} does not exist"
    found = {node.name for node in _public_objects(path)}
    assert name in found, f"{key}: {name} is no longer a public top-level name in {module}"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if getattr(n, "name", None) == name)
    assert ast.get_docstring(node) is None, (
        f"{key} now has a docstring — remove it from VERBATIM_UNDOCUMENTED and move the reason to "
        "the parity audit if the body really did change"
    )


def test_every_public_top_level_object_has_a_docstring() -> None:
    missing = []
    for path in _modules():
        for node in _public_objects(path):
            if ast.get_docstring(node):
                continue
            key = f"{_relative(path)}::{node.name}"
            if key in VERBATIM_UNDOCUMENTED:
                continue
            missing.append(f"{_relative(path)}:{node.lineno} {node.name}")
    assert not missing, (
        "public objects without a docstring (add one, or record why it is verbatim in "
        "VERBATIM_UNDOCUMENTED):\n  " + "\n  ".join(missing)
    )


def test_docstrings_say_something_beyond_the_name() -> None:
    """A one-line ``Return X.'' is a restatement; the reader already has the name."""
    too_short = []
    for path in _modules():
        for node in _public_objects(path):
            text = (ast.get_docstring(node) or "").strip()
            if not text:
                continue
            first = text.splitlines()[0].strip().rstrip(".")
            if len(first) < MIN_DOCSTRING:
                too_short.append(f"{_relative(path)}:{node.lineno} {node.name}: {first!r}")
    assert not too_short, "docstrings that restate the name instead of explaining it:\n  " + "\n  ".join(
        too_short
    )


def _module_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, (ast.Try, ast.If)):
            names |= _module_level_names(ast.Module(body=node.body, type_ignores=[]))
    return names


def test_every_all_entry_is_defined_once() -> None:
    """``__all__`` is the advertised surface; a name in it that is not defined breaks ``import *``."""
    broken = []
    checked = 0
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        exported = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ):
                exported = ast.literal_eval(node.value)
        if exported is None:
            continue
        checked += 1
        defined = _module_level_names(tree)
        # A package may advertise its submodules (``__all__ = ["dsv41", ...]``); those are not
        # names defined in the module body but they are legitimately importable via ``from x import``.
        children = path.parent if path.name == "__init__.py" else path.parent
        defined |= {child.stem for child in children.glob("*.py")}
        defined |= {child.name for child in children.iterdir() if child.is_dir() and (child / "__init__.py").is_file()}
        duplicated = {name for name in exported if exported.count(name) > 1}
        unknown = [name for name in exported if name not in defined]
        if duplicated:
            broken.append(f"{_relative(path)}: duplicate __all__ entries {sorted(duplicated)}")
        if unknown:
            broken.append(f"{_relative(path)}: __all__ names not defined here {unknown}")
    assert checked >= 20, f"only {checked} modules declare __all__; the scan is broken"
    assert not broken, "\n".join(broken)
