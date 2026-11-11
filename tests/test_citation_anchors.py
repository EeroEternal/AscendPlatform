"""Every ``file:line`` citation must point at a line a reader can actually open.

The docstrings in this repository cite vLLM by ``path:line`` — the migration rule is that the
reader can check every claim against ``reference/vllm-0.23/`` (verbatim excerpts) or, for claims
about the source repo, against ``tileinfer/`` itself.  A citation that no longer resolves is worse
than no citation: it reads exactly like evidence.

Three ways a citation rots, and this file catches all three:

* **the file moved or was never copied** — the excerpt directory is the only place these paths
  exist, so a rename there invalidates every citation into it;
* **the line number drifted** — the file is there, the line is not (or is now blank, or points
  into a different function);
* **the path is ambiguous** — ``registry.py:99`` matches two vLLM modules; the reader has a 50%
  chance of checking the wrong one.

Categories are explicit.  Anything that is *not* one of the four known kinds is a failure, which is
what keeps a new citation style from slipping in unchecked.

Scope: ``xxx.py:line`` citations.  Citations into ``.md`` files (``known-issues.md:59-114``) are
out of scope here — ``tests/test_doc_references.py`` owns path-level checks for those, and their
line numbers belong to a document that is edited far more often than code.
"""

from __future__ import annotations

import collections
import pathlib
import re
from typing import Dict, List, Optional, Tuple

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REFERENCE = ROOT / "reference" / "vllm-0.23"
SOURCE_REPO = pathlib.Path("/home/userlist/zxs/TileInfer")

#: The manual and the migration log also cite by ``file:line``.  They are checked with one
#: deliberate difference: a *bare* basename (``platform.py:1121``) is skipped there, because prose
#: says "the platform method" and the surrounding sentence names the module.  Code docstrings get
#: no such licence - they are read out of context, so they must be unambiguous.
#: Escape hatch for prose that has to *quote* a bad citation (a drift report, an example of the
#: notation).  It is greppable, so the exemption cannot be used quietly.
NOT_A_CITATION = "<!-- not-a-citation -->"

DOCS = sorted(pathlib.Path("docs").rglob("*.md")) + [
    pathlib.Path("MIGRATION_LOG.md"),
    pathlib.Path("README.md"),
]

#: ``path.py:12``, ``path.py:12-34`` (en dash accepted: they appear in copied docstrings), or the
#: durable form ``path.py::symbol`` used for dependencies that are not vendored.
CITATION = re.compile(r"([A-Za-z0-9_./-]+\.py)(?:::(\w+)|:(\d+)(?:[-\u2013](\d+))?)")

#: A citation with no directory: allowed only when it is unambiguous (see ``_resolve``).
FLOOR = {"vllm": 500, "source": 35, "torch": 8}


def _flattened_to_path(name: str) -> str:
    """``vllm__v1__worker__gpu_worker.py`` -> ``vllm/v1/worker/gpu_worker.py``.

    ``__init__`` needs the placeholder dance: stripping ``__`` -> ``/`` blindly turns the dunder
    into ``/init/``, and then a whole family of citations silently stops resolving.
    """
    if "____init__" in name:
        return name.replace("____init__", "/<INIT>").replace("__", "/").replace("<INIT>", "__init__")
    return name.replace("__", "/")


def _excerpt_index() -> Tuple[Dict[str, pathlib.Path], Dict[str, pathlib.Path]]:
    """(flattened name -> path, logical vLLM path -> path)."""
    by_name: Dict[str, pathlib.Path] = {}
    by_path: Dict[str, pathlib.Path] = {}
    for path in REFERENCE.glob("*.py"):
        by_name[path.name] = path
        by_path[_flattened_to_path(path.name)] = path
    return by_name, by_path


EXCERPTS_BY_NAME, EXCERPTS_BY_PATH = _excerpt_index()


def _resolve(cited: str) -> Tuple[Optional[pathlib.Path], str]:
    """(path, kind) for a citation path, or (None, reason).

    Falls back to a unique basename, which is how most docstrings write these (``gpu_worker.py:219``):
    the shorthand is fine *because* it is unambiguous, and this is where that is checked.
    """
    normal = cited.lstrip("./")
    if normal.startswith("reference/vllm-0.23/"):
        normal = normal.split("reference/vllm-0.23/", 1)[1]
    if normal in EXCERPTS_BY_NAME:
        return EXCERPTS_BY_NAME[normal], "vllm"
    if normal in EXCERPTS_BY_PATH:
        return EXCERPTS_BY_PATH[normal], "vllm"
    matches = [path for name, path in EXCERPTS_BY_PATH.items() if name == normal or name.endswith("/" + normal)]
    if len(matches) == 1:
        return matches[0], "vllm"
    if len(matches) > 1:
        return None, f"ambiguous among {sorted(p.name for p in matches)}"
    if normal.startswith("tileinfer/"):
        return SOURCE_REPO / normal, "source"
    if normal.startswith("torch/"):
        return None, "torch"  # resolved against the installed package, see below
    if normal.startswith(("src/", "src/ascend_platform/")):
        return ROOT / normal, "own"
    own = SRC / "ascend_platform" / normal
    if own.is_file():
        return own, "own"
    return None, "unresolved"


def _torch_file(relative: str) -> Optional[pathlib.Path]:
    torch = pytest.importorskip("torch")
    root = pathlib.Path(torch.__file__).resolve().parent
    candidate = root / relative[len("torch/"):]
    return candidate if candidate.is_file() else None


def _doc_citations() -> List[Tuple[pathlib.Path, int, str, int, int, str]]:
    found = []
    for path in DOCS:
        if not (ROOT / path).is_file():
            continue
        for lineno, line in enumerate((ROOT / path).read_text(encoding="utf-8").splitlines(), 1):
            if NOT_A_CITATION in line:
                continue  # a line that quotes a citation in order to say it is wrong
            for match in CITATION.finditer(line):
                cited = match.group(1)
                if "/" not in cited:
                    continue  # prose shorthand; see the DOCS comment
                symbol = match.group(2) or ""
                start = int(match.group(3) or 0)
                end = int(match.group(4) or start)
                found.append((path, lineno, cited, start, end, symbol))
    return found


def _doc_target(cited: str) -> Tuple[Optional[pathlib.Path], str]:
    """Resolve a documentation citation: excerpt (with the flattened spelling) then the usual ways."""
    normal = cited.lstrip("./")
    if normal in EXCERPTS_BY_NAME:
        return EXCERPTS_BY_NAME[normal], "vllm"
    if normal in EXCERPTS_BY_PATH:
        return EXCERPTS_BY_PATH[normal], "vllm"
    # Tables in docs/notes write excerpt names as ``fused_moe__fused_moe.py``; that is the
    # flattened form with the ``vllm__`` prefix dropped.
    unflattened = normal.replace("__", "/")
    matches = [path for name, path in EXCERPTS_BY_PATH.items() if name.endswith("/" + unflattened)]
    if len(matches) == 1:
        return matches[0], "vllm"
    if (ROOT / normal).is_file():
        return ROOT / normal, "own"  # the manual cites its own tests and sibling notes by path
    return _resolve(cited)


def _citations() -> List[Tuple[pathlib.Path, int, str, int, int, str]]:
    """(file, lineno, cited path, start, end, symbol).  ``start=end=0`` for a symbol citation."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in CITATION.finditer(line):
                symbol = match.group(2) or ""
                start = int(match.group(3) or 0)
                end = int(match.group(4) or start)
                found.append((path, lineno, match.group(1), start, end, symbol))
    return found


CITATIONS = _citations()
DOC_CITATIONS = _doc_citations()

#: Citations that cannot be verified here, with the reason.  Empty is the goal; every entry is a
#: claim a reader cannot check, so they are listed rather than silently skipped.
EXTERNAL: Dict[str, str] = {}


def test_the_doc_scan_found_the_citations() -> None:
    assert len(DOC_CITATIONS) >= 40, f"only {len(DOC_CITATIONS)} documentation citations found"


@pytest.mark.parametrize(
    "path,lineno,cited,start,end,symbol",
    DOC_CITATIONS,
    ids=lambda v: "" if not isinstance(v, str) else v,
)
def test_documentation_citations_point_at_real_lines(
    path: pathlib.Path, lineno: int, cited: str, start: int, end: int, symbol: str
) -> None:
    target, kind = _doc_target(cited)
    if kind == "torch":
        target = _torch_file(cited)
        if target is None:
            pytest.skip("torch citation, file not in this torch version")
    if kind == "source" and not SOURCE_REPO.is_dir():
        pytest.skip("the source repo is not present")
    assert target is not None, (
        f"{path}:{lineno} cites {cited}, which resolves to nothing (kind={kind}); the manual's "
        f"citations are checked so a line that moved cannot stay quoted as evidence"
    )
    if kind == "torch" and not target.is_file():
        pytest.skip("torch citation, file not in this torch version")
    assert target.is_file(), f"{path}:{lineno} cites {cited}, but {target} does not exist"
    lines = target.read_text(encoding="utf-8").splitlines()
    if symbol:
        assert any(symbol in line for line in lines), (
            f"{path}:{lineno} cites {cited}::{symbol}, which does not appear in {target}"
        )
        return
    where = f"{path}:{lineno} cites {cited}:{start}" + (f"-{end}" if end != start else "")
    assert end <= len(lines), f"{where}: the file has {len(lines)} lines"
    assert any(line.strip() for line in lines[start - 1:end]), f"{where}: the range is blank"


def test_the_scan_found_the_citations() -> None:
    assert len(CITATIONS) >= 500, f"only {len(CITATIONS)} citations found; the scan is broken"
    kinds = collections.Counter(_resolve(c[2])[1] for c in CITATIONS)
    assert kinds["vllm"] >= FLOOR["vllm"], kinds
    assert kinds["source"] >= FLOOR["source"], kinds


@pytest.mark.parametrize("cited", sorted({c[2] for c in CITATIONS}))
def test_every_cited_path_resolves(cited: str) -> None:
    """One failure per *path*, not per occurrence: the message names the sites."""
    path, kind = _resolve(cited)
    if kind == "torch":
        if _torch_file(cited) is not None:
            return
        pytest.skip(f"{cited} is not in this torch version; the citation targets upstream torch")
    if kind == "own":
        assert path is not None
        return
    if kind == "source" and not SOURCE_REPO.is_dir():
        pytest.skip("the source repo is not present, so its citations cannot be checked")
    assert path is not None, (
        f"{cited!r} is cited in "
        f"{sorted({str(c[0].relative_to(ROOT)) for c in CITATIONS if c[2] == cited})} but does not "
        f"resolve: {kind}. A citation the reader cannot open reads like evidence and is not."
    )


@pytest.mark.parametrize(
    "path,lineno,cited,start,end,symbol",
    CITATIONS,
    ids=lambda v: "" if not isinstance(v, str) else v,
)
def test_every_cited_line_exists_and_is_not_empty(
    path: pathlib.Path, lineno: int, cited: str, start: int, end: int, symbol: str
) -> None:
    target, kind = _resolve(cited)
    if kind == "torch":
        target = _torch_file(cited)
        if target is None:
            pytest.skip("torch citation, file not in this torch version")
    if kind == "source" and not SOURCE_REPO.is_dir():
        pytest.skip("the source repo is not present")
    if target is None:
        pytest.skip("covered by test_every_cited_path_resolves")
    lines = target.read_text(encoding="utf-8").splitlines()
    if symbol:
        # A dependency is not vendored, so a line number inside it goes stale on every upgrade.
        # ``::symbol`` survives upgrades and is still checkable: the name must exist in the file.
        assert any(symbol in line for line in lines), (
            f"{path.relative_to(ROOT)}:{lineno} cites {cited}::{symbol}, which does not appear in "
            f"the file this interpreter has ({target})"
        )
        return
    where = f"{path.relative_to(ROOT)}:{lineno} cites {cited}:{start}" + (f"-{end}" if end != start else "")
    assert end >= start, f"{where}: reversed range"
    assert end <= len(lines), f"{where}: the file has {len(lines)} lines"
    window = lines[start - 1:end]
    assert any(line.strip() for line in window), (
        f"{where}: the whole range is blank, so the citation points at nothing"
    )
