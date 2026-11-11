"""Every path a docstring or error message names must resolve — the move broke this twice.

The migration changed the depth of the tree (``tileinfer/vllm/doctor.py`` became
``src/ascend_platform/doctor/cli.py``) and the package name (``tileinfer.vllm`` became
``ascend_platform.engine``).  Both are invisible to a reader and to the type checker, and they broke
two things that nothing else noticed:

* ``doctor``'s step 4 looked for ``src/reference/`` because it counted parent directories, so it
  skipped on every machine while claiming "this is not a source checkout" — a check that could never
  pass, reported as "not applicable";
* the "no model profile matches" error told the reader to register a profile in
  ``tileinfer/vllm/profiles/``, a directory that does not exist in this repo.

A wrong *citation* is worse than no citation: it is read as an instruction.  So the paths in the
package's prose are checked mechanically, with three allowed resolutions (repo root, ``src/``, and the
package itself, which is how module-relative paths like ``engine/attention/backend.py`` are written)
and one explicit escape hatch: a citation that names the source repo or the operator repo
(``tileinfer/...``, "the operator repo's ...") is provenance, not a promise about this tree.
"""

from __future__ import annotations

import ast
import pathlib
import re
from typing import Iterator, List, Set, Tuple

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "ascend_platform"

#: A path-like citation: something with a slash that ends in a filename.
PATH_RE = re.compile(r"[A-Za-z_][\w./-]*\.[a-z]{2,4}(?::\d+(?:-\d+)?)?")

#: Where a citation may resolve.  ``""`` is the package itself (module-relative paths).
RESOLUTION_ROOTS = (ROOT, ROOT / "src", PKG)

#: Markers that turn a citation into provenance ("this is where the code came from").
PROVENANCE = ("tileinfer", "source repo", "operator repo", "TileInfer")

#: Only citations that claim to live *in this tree* are checked.  A bare ``v1/attention/backend.py``
#: is a vLLM path the source repo cites the same way (``reference/`` holds the copies); policing
#: those would mean rewriting several hundred upstream citations and would make this test meaningless.
OWN_TREE = (
    "src/", "tests/", "docs/", "reference/",
    "ascend_platform/", "engine/", "contracts/", "device/", "profiles/",
    "runtime/", "ops/", "doctor/",
)

#: Citations that name the *other* two repos while looking like ours.  Spelled out rather than
#: pattern-matched: the whole point of this file is that a path in a message is a promise, and a
#: blanket bypass ("anything under docs/ is fine") would recreate the bug it exists to catch.
FOREIGN = frozenset(
    {
        # the source repo's decision records and notes (not shipped here)
        "docs/plan-vllm-native-backend.md",
        "docs/performance.md",
        "docs/status.md",
        "docs/known-issues.md",
        "docs/notes/dsv41-facts.md",
        "docs/notes/dsv41-perf-plan.md",
        "docs/notes/vllm-api-0.23.md",
        "docs/notes/perf-comparison-rules.md",
        # the operator layer's modules (they live in TileInfer today)
        "ops/gdn.py",
        "ops/moe.py",
    }
)


def _strings() -> Iterator[Tuple[pathlib.Path, int, str]]:
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield path, node.lineno, node.value


def _citations(text: str) -> Set[str]:
    found = set()
    for match in PATH_RE.finditer(text):
        raw = match.group(0).split(":")[0]
        if "/" not in raw or raw.startswith(("/", "http")):
            continue
        if "..." in raw:
            continue  # an abbreviation ("reference/vllm-0.23/...gpu_model_runner.py"), not a path
        found.add(raw)
    return found


def _resolves(relative: str) -> bool:
    return any((base / relative).exists() for base in RESOLUTION_ROOTS)


VIOLATIONS: List[str] = []
for _path, _line, _text in _strings():
    if any(marker in _text for marker in PROVENANCE):
        continue
    for citation in sorted(_citations(_text)):
        if not citation.startswith(OWN_TREE) or citation in FOREIGN:
            continue
        if _resolves(citation):
            continue
        VIOLATIONS.append(f"{_path.relative_to(ROOT)}:{_line} cites {citation!r}")


@pytest.mark.parametrize("violation", VIOLATIONS, ids=lambda v: v.split(" cites ")[0])
def test_a_cited_path_resolves_or_says_it_is_provenance(violation: str) -> None:
    pytest.fail(
        f"{violation}\n"
        "Fix the citation (the file moved), or mark it as provenance by naming the repo it belongs to "
        "(`tileinfer/...`, \"the source repo's ...\", \"the operator repo's ...\")."
    )


def test_the_citation_check_is_not_vacuous() -> None:
    """Guards the checker itself: there must be citations, and they must mostly resolve."""
    citations = {c for _p, _l, t in _strings() for c in _citations(t)}
    assert len(citations) > 40, f"only {len(citations)} path-like strings found; the scan is broken"
    resolved = {c for c in citations if c.startswith(OWN_TREE) and _resolves(c)}
    assert len(resolved) > 15, (
        f"only {len(resolved)} own-tree citations resolve; the allowlist or the resolution roots "
        "are doing the work instead of the code"
    )


# =============================================================================================
# The same rule, for ``docs/``: a manual is read as a sequence of commands, so a stale path in one
# is worse than a stale comment in code.  ``docs/hardware-checklist.md`` is the bring-up manual and
# was rewritten mechanically from the source repo, which left it citing nine test files this repo
# does not contain (operator-side tests) and a ``scripts/`` directory it does not have.
# =============================================================================================

DOC_MARKERS = PROVENANCE + ("源仓", "算子仓", "AscendAttention", "今日", "旧")


def _docs() -> Iterator[pathlib.Path]:
    yield from sorted(ROOT.glob("*.md"))
    yield from sorted((ROOT / "docs").rglob("*.md"))


def _doc_violations(pattern: str, *, allow_placeholder: bool = True) -> List[str]:
    found = []
    compiled = re.compile(pattern)
    for doc in _docs():
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for match in compiled.finditer(line):
                target = match.group(0).rstrip(".,;:)`")
                if allow_placeholder and ("<" in target or "..." in target):
                    continue
                if any(marker in line for marker in DOC_MARKERS):
                    continue
                if any((base / target).exists() for base in RESOLUTION_ROOTS):
                    continue
                found.append(f"{doc.relative_to(ROOT)}:{lineno} cites {target!r}")
    return found


DOC_VIOLATIONS = (
    _doc_violations(r"src/ascend_platform/[A-Za-z0-9_./<>-]+")
    + _doc_violations(r"(?<![\w/])ascend_platform/[A-Za-z0-9_./<>-]+")
    + _doc_violations(r"tests/test_[a-z0-9_]+\.py", allow_placeholder=False)
)


@pytest.mark.parametrize("violation", DOC_VIOLATIONS, ids=lambda v: v.split(" cites ")[0])
def test_a_path_in_the_docs_resolves_or_says_whose_it_is(violation: str) -> None:
    pytest.fail(
        f"{violation}\n"
        "Fix the path, or say whose it is: this repo vs. the operator repo (`tileinfer/...`, "
        "『算子仓』) vs. the source repo (`docs/notes/dsv41-facts.md` style provenance)."
    )


def test_every_test_file_citation_names_a_file_that_exists() -> None:
    """The suite's own file names are cited in docstrings; a rename must not leave them dangling."""
    dangling = []
    for path, line, text in _strings():
        if any(marker in text for marker in PROVENANCE):
            continue
        for name in re.findall(r"tests/test_[a-z0-9_]+\.py", text):
            if not (ROOT / name).is_file():
                dangling.append(f"{path.relative_to(ROOT)}:{line} cites {name}")
    assert not dangling, "\n".join(dangling)


# =============================================================================================
# "§N" is a pointer, and a pointer to a section that changed meaning is worse than none
# =============================================================================================

#: ``docs/foo.md`` §3 / [`MIGRATION_LOG.md`](MIGRATION_LOG.md) §9 — a document plus a section number.
SECTION_REF = re.compile(
    r"(?:\[[^\]]*\]\()?`?([\w./-]+\.md)`?(?:\))?[^\S\n]*[）)]?[^\S\n]*§(\d+(?:\.\d+)?[a-z]?)"
)

PROSE = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md", ROOT / "MIGRATION_LOG.md"]


def _headings(path: pathlib.Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {m.group(1) for m in re.finditer(r"^#{2,4}\s+(\d+(?:\.\d+)?[a-z]?)\.?\s", text, re.M)}


def test_every_section_reference_points_at_a_section_that_exists() -> None:
    """A ``§`` pointer is read as "the part that explains this" — it has to exist.

    Two of these were wrong when this test was written: the checklist sent the reader to
    ``architecture.md`` §8 for the "not doing this" list, which lives in ``MIGRATION_LOG.md`` §9
    (and §8 is about *not renaming* things).  The heading numbers still existed, so only a check of
    the *target* catches it — and this one checks existence, which is the mechanical half.
    """
    broken: list[str] = []
    checked = 0
    for doc in PROSE:
        text = doc.read_text(encoding="utf-8")
        for match in SECTION_REF.finditer(text):
            name, section = match.group(1), match.group(2)
            target = next(
                (
                    candidate
                    for candidate in (doc.parent / name, ROOT / name, ROOT / "docs" / name)
                    if candidate.is_file()
                ),
                None,
            )
            if target is None:
                broken.append(f"{doc.relative_to(ROOT)}:{text[: match.start()].count(chr(10)) + 1} "
                              f"points at a document that does not exist: {name}")
                continue
            checked += 1
            if section in _headings(target):
                continue
            if any(heading.startswith(section + ".") for heading in _headings(target)):
                continue  # §4 pointing at the 4.x family is fine
            broken.append(
                f"{doc.relative_to(ROOT)}:{text[: match.start()].count(chr(10)) + 1} "
                f"cites {name} §{section}, which does not exist (headings: "
                f"{sorted(_headings(target))})"
            )
    assert checked >= 12, f"only {checked} section references found; the scan is broken"
    assert not broken, "\n".join(broken)
