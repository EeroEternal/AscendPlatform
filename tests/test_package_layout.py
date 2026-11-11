"""The packaging ratchet: entry points, the import surface, and the reviewed vLLM allowlist.

The *layering* rules live in :mod:`ascend_platform._acl` and are enforced in
``tests/test_layer_acl.py``; this file is about the repository as an installable distribution: the
entry points vLLM discovers, the extras that pin the engine line, and the set of modules allowed to
import vLLM at all.
"""

from __future__ import annotations

import ast
import pathlib
import re
import shutil
import subprocess
from typing import List

import pytest

from ascend_platform import _acl

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - the 3.9 CI venv
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ascend_platform"

def _modules():
    return sorted(SRC.rglob("*.py"))


def _imported_roots(path: pathlib.Path):
    """Every absolute module name imported at module level or inside a function."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def _layer_of(path: pathlib.Path) -> str:
    relative = path.relative_to(SRC)
    return relative.parts[0] if len(relative.parts) > 1 else ""


def test_every_package_directory_has_an_init():
    for directory in sorted({path.parent for path in _modules()}):
        if directory == SRC:
            continue
        assert (directory / "__init__.py").is_file(), f"{directory} has no __init__.py"


def test_every_module_compiles():
    for path in _modules():
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_the_package_root_imports_nothing_heavy():
    """``import ascend_platform`` must stay a version string and a docstring."""
    tree = ast.parse((SRC / "__init__.py").read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or ".")
    imported = [name for name in imported if name != "__future__"]
    assert imported == ["._version"] or imported == ["_version"], (
        f"the package root imports {imported}; submodules must be imported by the caller"
    )


def test_no_module_manipulates_sys_path():
    offenders = [
        str(path.relative_to(ROOT))
        for path in _modules()
        if re.search(r"sys\.path\.(insert|append|extend)", path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"sys.path manipulation in {offenders} makes imports order-dependent"


def _pyproject():
    with open(ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def test_the_two_vllm_entry_points_are_declared_and_resolvable():
    """vLLM discovers the platform by entry point; a typo here means the plugin silently vanishes."""
    import importlib

    pytest.importorskip("vllm", reason="the entry points target vLLM-backed modules")
    entry_points = _pyproject()["project"]["entry-points"]
    for group in ("vllm.platform_plugins", "vllm.general_plugins"):
        assert group in entry_points, f"{group} is not declared"
        for name, target in entry_points[group].items():
            module_name, _, attribute = target.partition(":")
            module = importlib.import_module(module_name)
            resolved = getattr(module, attribute)
            assert callable(resolved), f"{group}.{name} -> {target} is not callable"


def test_the_console_script_points_at_the_doctor():
    scripts = _pyproject()["project"]["scripts"]
    assert scripts["ascend-platform"] == "ascend_platform.doctor.cli:main"


def test_the_package_is_found_under_src():
    find = _pyproject()["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["src"]
    assert (ROOT / "src" / "ascend_platform" / "__init__.py").is_file()


def test_the_declared_dependencies_are_deliberately_small():
    project = _pyproject()["project"]
    assert project["dependencies"] == ["torch>=2.0"]
    optional = project["optional-dependencies"]
    assert "vllm>=0.23,<0.24" in optional["vllm"], "the engine extra must pin the 0.23 line"


@pytest.mark.parametrize("layer", _acl.LAYERS)
def test_each_layer_has_a_docstring_that_says_what_it_is_for(layer: str) -> None:
    module = (SRC / layer / "__init__.py").read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(module))
    assert docstring and len(docstring) > 80, f"{layer}/__init__.py needs a real docstring"


def test_the_operator_layer_is_never_a_hard_dependency_of_the_engine_import_path():
    """The engine may import ``vllm`` (it is the vLLM plugin), but only behind the documented guard."""
    guards = {
        "engine/attention/impl.py",
        "engine/attention/backend.py",
        "engine/attention/metadata.py",
        "engine/attention/prefill.py",
        "engine/attention/kv_cache.py",
        "engine/worker.py",
        "engine/platform.py",
        "engine/platform_dsv41.py",
        "engine/plugin.py",
        "engine/graphs.py",
        "engine/compat/patches.py",
        "engine/models/batching.py",
        "engine/models/state_slots.py",
        "engine/models/req_ids.py",
        "ops/install.py",
        "ops/candidates.py",
        "ops/switch.py",
        "ops/families/gdn_moe.py",
        "doctor/cli.py",
    }
    # Every engine/ops module that imports vllm must be one we have reviewed; a new one shows up
    # here first, which is the point.
    found = set()
    for path in _modules():
        if any(name == "vllm" or name.startswith("vllm.") for name in _imported_roots(path)):
            found.add(str(path.relative_to(SRC)))
    assert found <= guards, f"new modules import vLLM outside the reviewed set: {sorted(found - guards)}"


# ---------------------------------------------------------------------------------------------
# Deliverable 3 is "per-file green on CPU" - so the documented command has to work as written
# ---------------------------------------------------------------------------------------------


def test_pytest_puts_the_source_tree_on_the_path_for_single_file_runs() -> None:
    """``python -m pytest tests/test_x.py`` must work without an install and without ``PYTHONPATH``.

    ``[tool.pytest.ini_options] pythonpath`` only affects the pytest process; a bare checkout has no
    ``ascend_platform`` on ``sys.path`` otherwise, so every file would fail collection with
    ``ModuleNotFoundError`` unless the caller remembered an environment variable.  That is the kind
    of instruction that rots silently, so the ini option is pinned here rather than in prose.
    """
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    options = config["tool"]["pytest"]["ini_options"]
    entries = options.get("pythonpath") or []
    assert any(pathlib.Path(entry) == pathlib.Path("src") for entry in entries), (
        "pyproject.toml must keep 'src' in [tool.pytest.ini_options] pythonpath, or a single-file "
        f"run stops collecting; got {entries!r}"
    )


def test_one_file_runs_in_a_child_process_with_no_pythonpath() -> None:
    """End to end: a fresh child with ``PYTHONPATH`` removed still collects and imports this package.

    A test that runs the documented command the way a new checkout would.  It deliberately does not
    re-use the current interpreter's ``sys.path``: the point is the environment a reader has.
    """
    import os
    import subprocess
    import sys

    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.setdefault("PATH", "/usr/bin:/bin")
    env.setdefault("HOME", os.path.expanduser("~"))
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_wiring_strings.py", "--collect-only"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, (
        "a single-file run failed without PYTHONPATH; pyproject's pythonpath is what makes the "
        f"documented per-file command work\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )
    assert "tests/test_wiring_strings.py" in proc.stdout


# =============================================================================================
# the ignore file is part of the package: a rule that matches too much drops a whole layer
# =============================================================================================


def _git_files(*args: str) -> List[str]:
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("no git checkout here, so ignore rules cannot be observed")
    result = subprocess.run(
        [git, "ls-files", *args, "--exclude-standard", "--", "src", "tests"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_no_source_file_is_hidden_by_a_gitignore_rule():
    """``runtime/`` (unanchored) also matches ``src/ascend_platform/runtime/`` — it did, once.

    That is a silent packaging bug: every test in this repo still passes on the machine where the
    file exists, and the failure only appears in a *clone* (``ModuleNotFoundError: No module named
    'ascend_platform.runtime'``).  So the check is mechanical and asks git, not a human: list every
    file under ``src``/``tests`` that git is ignoring, and allow only caches.
    """
    ignored = _git_files("--others", "--ignored")
    allowed = ("__pycache__", ".pytest_cache", ".egg-info", ".pyc")
    offenders = [path for path in ignored if not any(token in path for token in allowed)]
    assert not offenders, (
        "these source files are ignored by .gitignore, so a clone would not contain them: "
        f"{offenders[:10]}{' ...' if len(offenders) > 10 else ''}. Anchor the rule (RUNTIME/ -> /RUNTIME/)"
    )


def test_the_runtime_layer_is_visible_to_git():
    """The positive half: the layer the rules once swallowed must be addable.

    ``--cached --others`` on purpose: before the first commit these files are untracked-but-stageable,
    afterwards they are tracked.  Both mean "git sees this path", which is the invariant; ``--others``
    alone silently stopped testing anything the moment the tree was committed.
    """
    visible = _git_files("--cached", "--others")
    assert "src/ascend_platform/runtime/step.py" in visible
    assert "src/ascend_platform/runtime/__init__.py" in visible


# =============================================================================================
# the license claim in pyproject must be backed by a file
# =============================================================================================


def test_the_declared_license_is_backed_by_a_file():
    """``license = { text = "Apache-2.0" }`` with no ``LICENSE`` is a claim, not a grant.

    The file is a verbatim copy of the source repo's license (the code is derived from it), and
    setuptools puts it in the wheel as ``dist-info/licenses/LICENSE`` — so the three statements
    (declaration, file, contents) have to agree or the distribution misstates its terms.
    """
    declaration = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'license = { text = "Apache-2.0" }' in declaration
    license_file = ROOT / "LICENSE"
    assert license_file.is_file(), "pyproject declares Apache-2.0 but there is no LICENSE file"
    text = license_file.read_text(encoding="utf-8")
    assert "Apache License" in text and "Version 2.0" in text
    # The source repo's license is the one this code came with; a divergence means someone
    # relicensed a derived work.
    source = pathlib.Path("/home/userlist/zxs/TileInfer/LICENSE")
    if source.is_file():
        assert text == source.read_text(encoding="utf-8"), "LICENSE differs from the source repo's"
