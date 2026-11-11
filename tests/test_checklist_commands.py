"""The bring-up manual is a program people follow; check the parts that can be checked offline.

`docs/hardware-checklist.md` is the hardware gate.  It was rewritten mechanically from the source
repo (``tileinfer`` -> ``ascend-platform``), and the rewrite produced *commands that cannot run*:

* ``python -c "import ascend-platform.vllm.plugin"`` — a hyphen is not an identifier, so the very
  first command of step 2 was a ``SyntaxError``;
* ``entry_points(group=...)`` filtered by ``e.name == "ascend-platform"`` while ``pyproject.toml``
  declares the name ``ascend_platform`` — so the check printed ``MISSING`` on a correctly installed
  package and sent the reader to the "only one platform plugin" triage row.

Both are invisible to a doc build and to a reviewer who knows what the line *means*.  They are not
invisible to ``compile()`` and to a comparison against ``pyproject.toml``.
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKLIST = ROOT / "docs" / "hardware-checklist.md"
#: Every document a reader *executes*.  The gate note matters most: it runs once, on the card, and a
#: typo there costs the session (``p5-gate.md`` is the only place the comparison script lives).
EXECUTABLE_DOCS = (CHECKLIST, ROOT / "docs" / "notes" / "p5-gate.md")
PYPROJECT = ROOT / "pyproject.toml"
PKG = ROOT / "src"

#: ``python -c "…"`` on one line, and ``python - <<'PY' … PY`` blocks.
ONE_LINER = re.compile(r'python(?:3)? -c "(?P<code>[^"]*)"')
HEREDOC = re.compile(r"python(?:3)? - <<'(?P<tag>[A-Z]+)'\n(?P<code>.*?)\n(?P=tag)\n", re.S)


def _snippets(path: pathlib.Path) -> list[tuple[str, int, str, str]]:
    """``(document, lineno, kind, source)`` for every Python snippet in a document."""
    text = path.read_text(encoding="utf-8")
    found = []
    for match in ONE_LINER.finditer(text):
        found.append((path.name, text[: match.start()].count("\n") + 1, "-c", match.group("code")))
    for match in HEREDOC.finditer(text):
        found.append(
            (path.name, text[: match.start()].count("\n") + 1, "heredoc", match.group("code"))
        )
    return found


SNIPPETS = [snippet for path in EXECUTABLE_DOCS for snippet in _snippets(path)]


def test_the_manual_has_snippets_to_check() -> None:
    kinds = [kind for _d, _l, kind, _s in SNIPPETS]
    assert kinds.count("-c") >= 5, f"only {kinds.count('-c')} one-liners found; the scan is broken"
    assert kinds.count("heredoc") >= 3, f"only {kinds.count('heredoc')} heredocs found"
    assert {doc for doc, _l, _k, _s in SNIPPETS} == {p.name for p in EXECUTABLE_DOCS}


def _code_only(source: str) -> str:
    """The snippet with comments removed: an ellipsis in prose is not a placeholder."""
    return "\n".join(line.split("#", 1)[0] for line in source.splitlines())


#: Snippets that are illustrative rather than runnable (``$MODEL``, literal ``...`` in code).
COMPILABLE = [
    snippet
    for snippet in SNIPPETS
    if "$" not in _code_only(snippet[3]) and "..." not in _code_only(snippet[3])
]


@pytest.mark.parametrize(
    "document,lineno,kind,source",
    COMPILABLE,
    ids=lambda v: f"line{v}" if isinstance(v, int) else "",
)
def test_every_python_snippet_in_the_manual_compiles(
    document: str, lineno: int, kind: str, source: str
) -> None:
    try:
        compile(source, f"{document}:{lineno}", "exec")
    except SyntaxError as exc:  # noqa: PERF203 - one bad snippet must not hide the others
        pytest.fail(
            f"{document}:{lineno} ({kind}) is not valid Python: {exc.msg}\n"
            f"  {source.strip().splitlines()[0] if source.strip() else ''}"
        )


def _pyproject_entry_points() -> dict[str, dict[str, str]]:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # Python 3.9: the repo's default interpreter
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["entry-points"]


def test_the_entry_point_names_quoted_in_the_manual_match_pyproject() -> None:
    """The manual's step 2 filters by name; a mismatch prints MISSING on a good install."""
    text = CHECKLIST.read_text(encoding="utf-8")
    declared = _pyproject_entry_points()
    for group, entries in declared.items():
        assert group in text, f"the manual never mentions the entry-point group {group}"
        for name, value in entries.items():
            assert name in text, f"the manual does not use the entry-point name {name}"
            assert value in text, f"the manual does not quote {group} = {name} -> {value}"
    assert "ascend-platform =" not in text, (
        "the manual filters entry points by 'ascend-platform', but pyproject declares "
        "'ascend_platform' - the check would print MISSING on a correct install"
    )


def test_every_module_the_manual_tells_you_to_import_exists() -> None:
    """Catches a doc that says ``import ascend_platform.engine.attn`` after the package moved."""
    text = CHECKLIST.read_text(encoding="utf-8")
    names = set(re.findall(r"(?:from|import)\s+(ascend_platform[\w.]*)", text))
    names |= set(re.findall(r"\b(ascend_platform(?:\.[a-z_][\w]*)+)", text))
    # ``engine.platform.register`` is an attribute, not a module: drop trailing parts until a real
    # module is found, and only then judge.  ``ascend_platform.__file__`` is an attribute of the
    # package itself, which resolves at length 1.
    missing = []
    for dotted in sorted(names):
        parts = dotted.split(".")
        while parts:
            relative = pathlib.Path(*parts)
            if (PKG / relative).with_suffix(".py").is_file() or (
                PKG / relative / "__init__.py"
            ).is_file():
                break
            parts.pop()
        else:
            missing.append(dotted)
    assert not missing, "docs/hardware-checklist.md names modules that do not exist: " + ", ".join(missing)


#: Every document that names a module a reader may ``import`` (the notes describe internals).
PROSE_DOCS = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "MIGRATION_LOG.md", ROOT / "README.md"]


def _names_defined_in(path: pathlib.Path) -> set[str]:
    """Every name a module body defines: classes, functions (any depth), assignments, imports."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _resolves(dotted: str) -> bool:
    """A module path, or ``module + one member`` (``...platform.register``, ``...worker.init_device``)."""
    parts = dotted.split(".")
    module = PKG / pathlib.Path(*parts)
    if module.with_suffix(".py").is_file() or (module / "__init__.py").is_file():
        return True
    if len(parts) < 2:
        return False
    parent = PKG / pathlib.Path(*parts[:-1])
    parent_file = parent.with_suffix(".py")
    if not parent_file.is_file() and not (parent / "__init__.py").is_file():
        return False
    if parent_file.is_file():
        return parts[-1] in _names_defined_in(parent_file)
    # a package: the member may live in any submodule it re-exports
    return any(
        parts[-1] in _names_defined_in(path)
        for path in [parent / "__init__.py", *sorted(parent.glob("*.py"))]
        if path.is_file()
    )


def _provider_entry_point_group() -> str:
    """The provider group string - the one dotted ``ascend_platform.*`` name that is not a module."""
    from ascend_platform.contracts.provider import PROVIDER_GROUP

    return PROVIDER_GROUP


def test_no_document_names_a_module_or_member_that_does_not_exist() -> None:
    """Checks every ``ascend_platform.…`` name the documentation tells a reader to use.

    A note that says ``ascend_platform.engine.attn.impl`` (a package that moved) or
    ``ascend_platform.engine.worker.init_device`` (a method that was renamed) is not caught by a
    link checker — it is plain text — and not by the reader, because the name looks plausible.  The
    check distinguishes modules from members: the parent must be a module file or package, and the
    last segment must be defined *in* it.  Everything else is a broken name.
    """
    missing: list[str] = []
    checked = 0
    for path in PROSE_DOCS:
        text = path.read_text(encoding="utf-8")
        for dotted in sorted(set(re.findall(r"\b(ascend_platform(?:\.[a-z_][\w]*)+)", text))):
            if dotted.endswith(("__file__", "__version__")):
                continue  # attributes of the package object itself, not paths a reader imports
            if dotted == _provider_entry_point_group():
                continue  # an entry-point *group* name, not a module: it has no file to resolve
            checked += 1
            if not _resolves(dotted):
                missing.append(f"{path.relative_to(ROOT)}: {dotted}")
    assert checked >= 8, f"only {checked} module references found; the scan is broken"
    assert not missing, "documents name modules or members that do not exist: " + ", ".join(missing[:10])


# =============================================================================================
# the distribution name and the import name are different strings
# =============================================================================================

#: ``ascend-platform.something`` — the hyphen belongs to the distribution, never to a module path.
#: Not preceded by a path/hyphen character: ``/tmp/p5-ascend-platform.json`` is a file name, not a
#: module path, and a dotted name that follows ``import``/``from`` or stands alone is the bug.
DISTRIBUTION_AS_MODULE = re.compile(r"(?<![/\w-])ascend-platform\.[A-Za-z_]")


def test_no_document_uses_the_distribution_name_as_a_module() -> None:
    """``pip install ascend-platform`` is right; ``import ascend-platform.vllm`` is not a thing.

    The mechanical rename that produced this repository turned ``tileinfer.vllm`` into
    ``ascend-platform.vllm`` in a few prose files.  It is invisible to a reader who knows what the
    sentence *means* (the file even looked like a module path) and it poisons the next copy-paste.
    ``<!-- not-a-citation -->`` marks the lines that quote the bad form on purpose.
    """
    offenders = []
    for path in sorted((ROOT / "docs").rglob("*.md")) + [
        ROOT / "README.md",
        ROOT / "MIGRATION_LOG.md",
    ]:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "not-a-citation" in line:
                continue
            if DISTRIBUTION_AS_MODULE.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:100]}")
    assert not offenders, (
        "documents use the distribution name as a module path (use `ascend_platform.…` for modules, "
        "`ascend-platform` only for pip/metadata):\n  " + "\n  ".join(offenders)
    )
