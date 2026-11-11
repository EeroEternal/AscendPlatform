"""Parity with the source repo: no definition was dropped, no body was rewritten in transit.

The migration's claim is "moved, not rewritten".  That claim is only worth something if it can be
checked, so this file re-derives it from the source tree when that tree is available:

1. **Nothing missing.**  Every top-level function and every method of the TileInfer file behind a
   migrated module exists in the migrated module.
2. **Nothing rewritten.**  Each body is compared as a canonical token stream — node types,
   identifiers, argument names and numeric/monkey-patch constants — with string literals (docstrings,
   log messages) and import module names normalised away, because those are exactly what the move is
   allowed to change.  Four bodies legitimately differ; they are listed with their reason, and any
   *new* difference fails this test.
3. **Nothing trimmed.**  The source lines occupied by the shared definitions are still occupied.

Where the source repo is absent (a clean checkout of this repo alone, CI), the file skips: it is a
migration audit, not a runtime property.
"""

from __future__ import annotations

import ast
import os
import pathlib
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PKG = ROOT / "src" / "ascend_platform"
SOURCE_REPO = pathlib.Path(os.environ.get("TILEINFER_SRC", "/home/userlist/zxs/TileInfer"))
SOURCE_PKG = SOURCE_REPO / "tileinfer"

#: (source path relative to ``tileinfer/``, target path relative to ``src/ascend_platform/``)
PAIRS: Tuple[Tuple[str, str], ...] = (
    ("vllm/platform.py", "engine/platform.py"),
    ("vllm/platform_dsv41.py", "engine/platform_dsv41.py"),
    ("vllm/worker.py", "engine/worker.py"),
    ("vllm/graphs.py", "engine/graphs.py"),
    ("vllm/plugin.py", "engine/plugin.py"),
    ("vllm/compat/patches.py", "engine/compat/patches.py"),
    ("vllm/attn/backend.py", "engine/attention/backend.py"),
    ("vllm/attn/metadata.py", "engine/attention/metadata.py"),
    ("vllm/attn/impl.py", "engine/attention/impl.py"),
    ("vllm/attn/kv_cache.py", "engine/attention/kv_cache.py"),
    ("vllm/attn/prefill.py", "engine/attention/prefill.py"),
    ("vllm/doctor.py", "doctor/cli.py"),
    ("vllm/ops/install.py", "ops/install.py"),
    ("vllm/ops/candidates.py", "ops/candidates.py"),
    ("vllm/ops/switch.py", "ops/switch.py"),
    ("vllm/ops/families/__init__.py", "ops/families/__init__.py"),
    ("vllm/ops/families/gdn_moe.py", "ops/families/gdn_moe.py"),
    ("vllm/models/dsv41_batching.py", "engine/models/batching.py"),
    ("vllm/models/dsv41_state_slots.py", "engine/models/state_slots.py"),
    ("vllm/models/dsv41_req_ids.py", "engine/models/req_ids.py"),
    ("utils.py", "utils.py"),
    ("vllm/profiles/__init__.py", "profiles/__init__.py"),
    ("vllm/profiles/base.py", "profiles/base.py"),
    ("vllm/profiles/qwen3_6.py", "profiles/qwen3_6.py"),
    ("vllm/profiles/dsv41.py", "profiles/dsv41.py"),
    ("serving/buckets.py", "runtime/buckets.py"),
    ("serving/planner.py", "runtime/planner.py"),
    ("serving/step.py", "runtime/step.py"),
    ("serving/phases.py", "runtime/phases.py"),
    ("serving/counters.py", "runtime/counters.py"),
    ("serving/flags.py", "runtime/flags.py"),
)

#: Bodies that differ, and why.  A new entry here is a decision; a new *difference* is a regression.
EXPECTED_DIFFERENCES: Mapping[str, str] = {
    "vllm/plugin.py::install": (
        "one call added at the top: _ensure_visible_logs() gives this package's logger a destination "
        "under `vllm serve`, where vLLM's dictConfig puts the only handler on the 'vllm' logger with "
        "propagate=False (vllm/logger.py, DEFAULT_LOGGING_CONFIG).  Without it the INFO banner the "
        "runbook greps ('ascend-platform vLLM plugin') is dropped by the logging module and WARNING+ "
        "degrades to logging.lastResort's bare stderr - measured on the card: a healthy serve logged "
        "zero 'ascend-platform' lines.  Everything after the call (probe, patches, backend, graph, "
        "ops, models, banner) is unchanged"
    ),
    "vllm/plugin.py::_install_models": (
        "the hardcoded model-registration tuple became the provider loop "
        "(engine/model_providers.py), so a second model family is one table entry"
    ),
    "vllm/attn/prefill.py::PrefillPath.run": (
        "the kernel calls (forward_prefill / _pack_indices / RaggedMetadata) now go through the "
        "operator port instead of importing tileinfer.kernels directly"
    ),
    "vllm/ops/families/__init__.py::family_names": (
        "docstring added: it was one of the few migrated functions that arrived undocumented, and it "
        "is the one an operator reads to find out which families exist; the executable body (a list "
        "comprehension over FAMILIES) is unchanged"
    ),
    "vllm/ops/families/gdn_moe.py::_reference_experts": (
        "the oracle and the unquantised-config refusal are resolved through the operator port"
    ),
    "vllm/ops/install.py::_plan": (
        "the impl-missing branch prefers OpBinding.impl_note: a family now resolves each "
        "implementation separately, so 'which name is missing from the operator layer, and why' is "
        "recorded per op instead of the whole table failing to build.  The abi_ok / target-resolution "
        "/ alias branches are unchanged"
    ),
    "vllm/plugin.py::_install_ops": (
        "one keyword argument: install_ops(engine_factories=_engine_factories()).  The MoE method "
        "class is a vLLM shape (CustomOp.register_oot), so it is built in engine/abi and injected - "
        "the alternative was ops/families importing engine/abi, which is the wrong direction "
        "(engine -> ops).  Everything else in the wrapper is unchanged"
    ),
    "vllm/doctor.py::_check_ops_binding": (
        "one segment prepended to the detail line: 'provider <name> <version> (<n> op(s))'.  With "
        "more than one operator package installable, '10 bound' is not actionable until the reader "
        "knows who served them - and the provider is the new contract (contracts/provider.py).  The "
        "bind/decline/unexpected decision itself is unchanged"
    ),
    "vllm/doctor.py::_run_check": (
        "a third branch: OperatorLayerUnavailable is an environment state ('install the operator "
        "layer or set ASCEND_PLATFORM_OPS'), not a bug in this package - the generic branch told the "
        "reader to report a traceback, sending them away from the one fixable thing; the "
        "wrong-shape / unknown-status checks and the generic branch are unchanged"
    ),
    "vllm/doctor.py::_repo_root": (
        "the file moved two levels deeper (tileinfer/vllm/doctor.py -> "
        "src/ascend_platform/doctor/cli.py), so parents[2] pointed at src/ instead of the checkout; "
        "it now walks up to pyproject.toml, which survives both layouts"
    ),
    "vllm/doctor.py::_check_runner_source": (
        "consumes the new _repo_root() (which may be None for an installed copy) and reports that "
        "case accurately instead of claiming 'not a source checkout' from inside one"
    ),
    "vllm/ops/install.py::install_ops": (
        "two additions, both about saying *why* nothing was bound: a missing operator layer is "
        "recorded in the report (``layer_error``) instead of only logged, because a bare '0 bound' "
        "reads as 'nothing to bind' — the one thing it does not mean — and the operator layer's "
        "version is now verified against SUPPORTED_OPERATOR_RANGE *before* the family table is read "
        "(a mismatch is the cause, 'no module named ...' is the symptom); the family resolution "
        "moved into _resolve_families unchanged, and the binding logic itself is untouched"
    ),
    "vllm/ops/install.py::OpsReport.summary": (
        "carries the layer error appended to the one-line summary the plugin and doctor print"
    ),
    "vllm/ops/families/gdn_moe.py::bind": (
        "the eight operator imports became ops_module(...) lookups: the table names implementations, "
        "it does not import them"
    ),
}

#: Calls that may legitimately stop appearing in a file, with the reason.  Both entries below are
#: *moves*, not drops: the work still happens, one level away (a rename, or the operator port).
EXPECTED_REMOVED_CALLS: Dict[Tuple[str, str], str] = {
    ("vllm/plugin.py", "register_config"): (
        "renamed register_configs() (engine/model_providers.py), which runs every provider package's "
        "own register_config(); the source's vllm/config.py is a model-provider file that stayed out "
        "of this repo (MIGRATION_LOG.md §8)"
    ),
    ("vllm/attn/prefill.py", "_pack_indices"): (
        "the kernel direct call became a port call (contracts/ports.py::pack_indices delegates to the "
        "operator layer's decode_kernel) - deliverable 5: no kernel symbols in src/"
    ),
}

pytestmark = pytest.mark.skipif(
    not SOURCE_PKG.is_dir(),
    reason=f"the source repo is not present at {SOURCE_REPO}; set TILEINFER_SRC to enable the audit",
)


def _definitions(path: pathlib.Path) -> Dict[str, ast.AST]:
    """Top-level functions plus every method, keyed ``qualname``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[f"{node.name}.{item.name}"] = item
    return found


def _canonical_module(module: str) -> str:
    """Import targets are allowed to change: the whole package moved."""
    if module.startswith(("tileinfer", "ascend_platform")):
        return "<LAYER>"
    return module


def _tokens(node: ast.AST) -> List[object]:
    """The body as a canonical stream: shape + identifiers + numbers; strings dropped."""
    out: List[object] = []
    stack: List[object] = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(reversed(current))
            continue
        if not isinstance(current, ast.AST):
            continue
        if isinstance(current, ast.Constant) and isinstance(current.value, str):
            out.append("STR")  # docstrings, log messages, error texts: the move may edit them
            continue
        if isinstance(current, ast.Name):
            out.append(("Name", current.id))
        elif isinstance(current, ast.Attribute):
            out.append(("Attr", current.attr))
        elif isinstance(current, ast.arg):
            out.append(("arg", current.arg))
        elif isinstance(current, ast.Constant):
            out.append(("const", repr(current.value)))
        elif isinstance(current, ast.ImportFrom):
            out.append(("importfrom", _canonical_module(current.module or ""),
                        tuple(sorted(alias.name for alias in current.names))))
            continue  # do not descend into the alias names again
        elif isinstance(current, ast.Import):
            out.append(("import", tuple(sorted(_canonical_module(alias.name) for alias in current.names))))
            continue
        else:
            out.append(type(current).__name__)
        stack.extend(ast.iter_child_nodes(current))
    return out


def _span(node: ast.AST) -> int:
    return (getattr(node, "end_lineno", None) or node.lineno) - node.lineno + 1


def _pairs() -> Iterator[Tuple[str, pathlib.Path, pathlib.Path]]:
    for source_rel, target_rel in PAIRS:
        yield source_rel, SOURCE_PKG / source_rel, SRC_PKG / target_rel


@pytest.mark.parametrize("source_rel,source,target", list(_pairs()), ids=[rel for rel, _s, _t in _pairs()])
def test_no_definition_was_left_behind(source_rel, source, target) -> None:
    missing: List[str] = []
    if not target.is_file():
        pytest.fail(f"{source_rel} was migrated to {target.relative_to(ROOT)}, which does not exist")
    source_defs, target_defs = _definitions(source), _definitions(target)
    missing = sorted(set(source_defs) - set(target_defs))
    assert not missing, (
        f"{source_rel} -> {target.relative_to(ROOT)}: {missing} never arrived. "
        "If the omission is deliberate, record it in MIGRATION_LOG.md and split the pair list."
    )


@pytest.mark.parametrize("source_rel,source,target", list(_pairs()), ids=[rel for rel, _s, _t in _pairs()])
def test_no_body_was_rewritten_in_transit(source_rel, source, target) -> None:
    source_defs, target_defs = _definitions(source), _definitions(target)
    unexpected = []
    for name in sorted(set(source_defs) & set(target_defs)):
        before, after = _tokens(source_defs[name]), _tokens(target_defs[name])
        if before == after:
            continue
        key = f"{source_rel}::{name}"
        if key in EXPECTED_DIFFERENCES:
            continue
        index = next(
            (i for i, (x, y) in enumerate(zip(before, after)) if x != y), min(len(before), len(after))
        )
        unexpected.append(f"{key} (first difference at token {index}: {before[index:index+3]} vs {after[index:index+3]})")
    assert not unexpected, (
        "these bodies changed during the move:\n  " + "\n  ".join(unexpected) + "\n"
        "A deliberate change belongs in EXPECTED_DIFFERENCES with its reason; anything else means "
        "the move rewrote behaviour."
    )


@pytest.mark.parametrize("source_rel,source,target", list(_pairs()), ids=[rel for rel, _s, _t in _pairs()])
def test_the_shared_definition_volume_is_unchanged(source_rel, source, target) -> None:
    source_defs, target_defs = _definitions(source), _definitions(target)
    # A documented rewrite is allowed to be shorter (that is what "the import became a port call"
    # means); every other shared definition must still occupy at least its source extent.
    rewritten = {
        key.split("::", 1)[1] for key in EXPECTED_DIFFERENCES if key.startswith(f"{source_rel}::")
    }
    shared = (set(source_defs) & set(target_defs)) - rewritten
    source_lines = sum(_span(source_defs[name]) for name in shared)
    target_lines = sum(_span(target_defs[name]) for name in shared)
    assert target_lines >= source_lines, (
        f"{source_rel}: the shared definitions occupy {target_lines} lines here against "
        f"{source_lines} in the source; a body was trimmed"
    )


def test_the_audit_covers_the_files_the_objective_names() -> None:
    """Cheap self-check: the pair list is not allowed to silently shrink."""
    assert len(PAIRS) == 31
    targets = {target for _source, target in PAIRS}
    assert "engine/platform.py" in targets and "engine/worker.py" in targets
    assert "runtime/buckets.py" in targets and "ops/install.py" in targets
    assert "doctor/cli.py" in targets

# =============================================================================================
# Beyond function bodies: the parts of a module that are not inside a ``def``
# =============================================================================================
#
# The body audit above compares what is *inside* functions.  It cannot see a dropped class field, a
# module constant whose value silently changed, or a decorator that did not come along.  Those are
# exactly the things a rename breaks, and one of them (``WORKER_CLASS_PATH``) is the string vLLM uses
# to import our worker: a missed one is a boot failure, not a style issue.

#: Module-level items whose value legitimately changed, keyed like ``EXPECTED_DIFFERENCES``.
EXPECTED_MODULE_LEVEL: Mapping[str, str] = {
    "vllm/platform.py::ATTN_BACKEND_CLASS_PATH": (
        "the class lives at ascend_platform.engine.attention.backend now; this string is what vLLM "
        "imports, so it must name the new module"
    ),
    "vllm/platform.py::WORKER_CLASS_PATH": "same: the worker's dotted path moved with the package",
    "vllm/platform.py::WORKER_MODULE": "same: the module half of the worker path",
    "vllm/worker.py::WORKER_CLASS_PATH": "same constant, defined next to the class it names",
    "vllm/plugin.py::PLUGIN_ENTRY_POINT": (
        "the entry-point value quoted back to the user: it must match pyproject.toml or the advice "
        "is wrong"
    ),
    "vllm/plugin.py::PLATFORM_ENTRY_POINT": "same, for the platform plugin",
    "vllm/plugin.py::PLATFORM_CLASS_PATH": "same, for the platform class",
    "vllm/doctor.py::_HEADER": (
        "the banner is now 'ascend-platform doctor'; the source repo's name is not printed by a "
        "package called ascend_platform"
    ),
    "vllm/ops/install.py::class OpBinding": (
        "one field added: impl_note carries 'the operator layer has no <module>.<attr>' for the one "
        "record that needs it, so a half-installed layer reports per op instead of 0 bound with a "
        "warning that blames the model profile"
    ),
    "vllm/graphs.py::UNMAPPED_TORCH_CUDA_NAMES": (
        "two entries appended: `_is_compiled` / `is_initialized` are used by "
        "vllm/utils/platform_utils.py's `cuda_is_initialized()`, whose excerpt was vendored to "
        "make the citation audit possible; both are CUDA-only predicates and aliasing them would "
        "make vLLM believe CUDA is up on an NPU box, so they are recorded as deliberately unmapped. "
        "The six original entries are unchanged"
    ),
    "vllm/ops/install.py::class OpsReport": (
        "the report gained layer_error so a missing operator layer is named instead of summarised "
        "as '0 bound'"
    ),
}


def _module_items(path: pathlib.Path) -> Dict[str, object]:
    """Everything a module declares outside its functions: classes (+fields), constants, blocks."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: Dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = "func"
        elif isinstance(node, ast.ClassDef):
            fields: Dict[str, str] = {}
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    fields[item.target.id] = ast.unparse(item.annotation)
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name):
                            fields[target.id] = ast.unparse(item.value)[:40]
            found[f"class {node.name}"] = ("class", tuple(sorted(fields)))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = _literal(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found[node.target.id] = _literal(node.value)
        elif isinstance(node, ast.If):
            head = ast.unparse(node.test)[:40]
            body = tuple(
                ast.unparse(stmt).split("(")[0][:40]
                for stmt in node.body
                if not isinstance(stmt, (ast.Import, ast.ImportFrom))
            )
            found[f"if {head}"] = body
        else:
            found[type(node).__name__] = "stmt"
    return found


def _literal(node: Optional[ast.AST]) -> str:
    if node is None:
        return "<none>"
    try:
        return repr(ast.literal_eval(node))[:60]
    except Exception:  # noqa: BLE001 - a computed value is compared as "<expr>"
        return "<expr>"


MODULE_ITEMS = {
    source_rel: (_module_items(source), _module_items(target))
    for source_rel, source, target in _pairs()
    if source.is_file() and target.is_file()
}


def test_the_module_level_audit_is_not_vacuous() -> None:
    total = sum(len(before) for before, _after in MODULE_ITEMS.values())
    assert total > 400, f"only {total} module-level items compared; the scan is broken"


@pytest.mark.parametrize("source_rel", sorted(MODULE_ITEMS), ids=sorted(MODULE_ITEMS))
def test_no_module_level_item_was_left_behind(source_rel: str) -> None:
    before, after = MODULE_ITEMS[source_rel]
    missing = [key for key in before if key not in after]
    assert not missing, (
        f"{source_rel}: these module-level items are in the source and not here: "
        + ", ".join(sorted(missing))
    )


@pytest.mark.parametrize("source_rel", sorted(MODULE_ITEMS), ids=sorted(MODULE_ITEMS))
def test_no_module_level_value_changed_unexplained(source_rel: str) -> None:
    before, after = MODULE_ITEMS[source_rel]
    unexplained = []
    for key, value in before.items():
        if key not in after or after[key] == value:
            continue
        if f"{source_rel}::{key}" in EXPECTED_MODULE_LEVEL:
            continue
        unexplained.append(f"{key}: {value!r} -> {after[key]!r}")
    assert not unexplained, (
        f"{source_rel}: these module-level values changed during the move:\n  "
        + "\n  ".join(unexplained)
        + "\nA deliberate change belongs in EXPECTED_MODULE_LEVEL with its reason."
    )


@pytest.mark.parametrize("source_rel,source,target", list(_pairs()), ids=[rel for rel, _s, _t in _pairs()])
def test_no_decorator_was_dropped_in_transit(source_rel, source, target) -> None:
    """``@staticmethod``, ``@dataclass(frozen=True)``, ``@cache`` all change behaviour when lost."""
    before, after = _decorators(source), _decorators(target)
    dropped = [
        key for key, decorators in before.items() if key in after and after[key] != decorators
    ]
    assert not dropped, (
        f"{source_rel}: the decorators changed on " + ", ".join(sorted(dropped))
    )


def _decorators(path: pathlib.Path) -> Dict[str, List[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = [ast.unparse(d) for d in node.decorator_list]
        elif isinstance(node, ast.ClassDef):
            found[f"class {node.name}"] = [ast.unparse(d) for d in node.decorator_list]
    return found


# ---------------------------------------------------------------------------------------------
# The ratchet's own blind spot: an allowlisted body is no longer compared *at all*
# ---------------------------------------------------------------------------------------------


def _calls(node: ast.AST) -> Set[str]:
    """Every callee name in ``node``, flattened to its last component (``f`` for ``a.b.f``)."""
    found: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                found.add(func.id)
            elif isinstance(func, ast.Attribute):
                found.add(func.attr)
    return found


@pytest.mark.parametrize("source_rel,source,target", list(_pairs()), ids=[rel for rel, _s, _t in _pairs()])
def test_the_allowlist_does_not_silence_a_file(source_rel, source, target) -> None:
    """``EXPECTED_DIFFERENCES`` exempts one *definition*; it must not exempt the rest of the file.

    Allowlisting ``install_ops`` means its body is not compared any more - so the next edit to that
    body would be invisible, and the entry would keep claiming only the reason recorded the first
    time.  The invariant that survives an allowlist is coarser and durable: across the file, every
    call the source made is still made somewhere in the target (code may move between functions,
    as ``_resolve_families`` did; it may not disappear).
    """
    source_calls: Set[str] = set()
    for definition in _definitions(source).values():
        source_calls |= _calls(definition)
    target_calls: Set[str] = set()
    for definition in _definitions(target).values():
        target_calls |= _calls(definition)

    exempt = {name for (rel, name), _reason in EXPECTED_REMOVED_CALLS.items() if rel == source_rel}
    lost = sorted(source_calls - target_calls - exempt)
    assert not lost, (
        f"{source_rel} -> {target.relative_to(ROOT)}: these calls no longer happen anywhere in the "
        f"file: {lost}.  A moved call is fine; a dropped one changes behaviour silently, which is "
        "exactly what an allowlisted body would hide."
    )


# =============================================================================================
# Completeness, checked from the *source* side: every file is migrated or excluded on purpose
# =============================================================================================
#
# ``PAIRS`` is a hand-written list, so it can only prove that the files it names still match.  It
# cannot notice a source file that nobody ever listed — and "全量迁移" (deliverable 2) is a claim
# about the source tree, not about the list.  The two tables below close that hole: the audit
# enumerates ``tileinfer/**/*.py`` and requires every path to be one of the three cases.

#: Whole source subtrees that stay out of this repository, with the reason.  The key is a prefix.
EXCLUDED_TREES: Mapping[str, str] = {
    "kernels/": (
        "operator implementations (tilelang/Triton kernels).  They belong to the operator repo; "
        "this package keeps zero kernel code and _acl forbids importing it"
    ),
    "models/": (
        "the DeepSeek-V4.1-Flash model forward (block/layer/offload/spec_decode/weights).  The "
        "boundary excludes model forward; the platform keeps only its batching semantics "
        "(engine/models/{batching,req_ids,state_slots}.py)"
    ),
    "testing/": (
        "reference implementations of the kernels and their comparison helpers - the operator "
        "layer's own acceptance harness, not platform semantics"
    ),
    "attention/": (
        "the engine-agnostic attention API and its backends (reference / tilelang-ascend950).  "
        "Operators again: this package keeps the vLLM-side adaptation (engine/attention/*), and the "
        "port's docstrings name which backend a degraded path falls back to"
    ),
    "integrations/": (
        "pre-warm integration (shape/cache warm-up for the model): model-side, not session semantics"
    ),
    "vllm/ops/fla/": (
        "FLA/GDN operator implementations (chunk/wu/kkt/solve_tril/recurrent/...): operator layer"
    ),
}

#: Individual files, same idea.  Package ``__init__``s appear here because the ones here are written
#: for this layout, not copied: the audit compares tables and bodies, and a rewritten ``__init__``
#: has no shared body to compare.
EXCLUDED_FILES: Mapping[str, str] = {
    "__init__.py": (
        "package metadata and the export table: this repo's __init__ is written for the new layout, "
        "and the public surface is pinned by tests/test_public_api.py instead"
    ),
    "_version.py": (
        "the version literal: provided by this repo's pyproject.toml and re-exported from __init__"
    ),
    "metadata.py": (
        "operator-layer data types (RaggedMetadata, TileSchedule).  Not copied: they are reached "
        "through contracts/ports.py, and this repo keeps only the pure-data contract "
        "(contracts/manifest.py)"
    ),
    "plan.py": (
        "operator-layer planning functions (plan_query_tiles_from_host and friends): the port is a "
        "thin forwarder and contracts/version.py gates the range at start-up"
    ),
    "serving/__init__.py": "package init: runtime/__init__.py is written for the new layout",
    "vllm/__init__.py": "package init: engine/__init__.py is written for the new layout",
    "vllm/attn/__init__.py": "package init: engine/attention/__init__.py is written for the new layout",
    "vllm/compat/__init__.py": "package init: engine/compat/__init__.py is written for the new layout",
    "vllm/models/__init__.py": (
        "package init listing the model providers: engine/models/__init__.py exports the three "
        "batching modules and nothing else (the model itself is out of scope)"
    ),
    "vllm/ops/__init__.py": (
        "the operator layer's facade (re-exports the gdn/moe implementations and applies the extern "
        "binding).  The platform does not copy it: names come from contracts/ports.py"
    ),
    "vllm/ops/extern.py": (
        "superseded by contracts/ports.py + contracts/manifest.py (MIGRATION_LOG.md §4.8): the old "
        "bridge rebound module attributes at import and fell back silently; the new one resolves "
        "lazily and checks the version range at start-up"
    ),
    "vllm/config.py": (
        "the model-provider registration file: its registration role moved to "
        "engine/model_providers.py (MIGRATION_LOG.md §8)"
    ),
    "vllm/models/dsv41.py": (
        "the model itself (layers, attention wiring, spec-decode hooks): model forward, which the "
        "boundary excludes - the batching helpers next to it are migrated"
    ),
    "vllm/models/mmap_weights.py": (
        "weight loading via mmap: part of the model's load path, not session semantics"
    ),
    "vllm/ops/gdn.py": "operator implementation (GDN/FLA entry points): operator layer",
    "vllm/ops/moe.py": "operator implementation (fused_experts / topk_softmax / silu_and_mul): operator layer",
    "vllm/ops/fused_moe_oot.py": (
        "operator implementation (the OOT MoE method subclass): operator layer, and the boundary "
        "excludes kernel code"
    ),
}


def _source_files() -> List[str]:
    return sorted(
        path.relative_to(SOURCE_PKG).as_posix()
        for path in SOURCE_PKG.rglob("*.py")
        if "__pycache__" not in str(path)
    )


def _migrated_sources() -> set:
    return {source for source, _target in PAIRS}


def test_every_source_file_is_migrated_or_excluded_on_purpose() -> None:
    files = _source_files()
    assert len(files) >= 100, f"the audit found only {len(files)} source files; the scan is broken"
    migrated = _migrated_sources()
    unaccounted = [
        path
        for path in files
        if path not in migrated
        and path not in EXCLUDED_FILES
        and not any(path.startswith(prefix) for prefix in EXCLUDED_TREES)
    ]
    assert not unaccounted, (
        "these source files are neither in PAIRS nor in EXCLUDED_TREES/EXCLUDED_FILES, so nobody ever "
        "decided about them:\n  " + "\n  ".join(unaccounted)
    )


def test_the_three_tables_account_for_every_file_exactly_once() -> None:
    files = set(_source_files())
    migrated = _migrated_sources()
    excluded = set(EXCLUDED_FILES)
    trees = {
        path
        for path in files
        if any(path.startswith(prefix) for prefix in EXCLUDED_TREES)
    }
    assert migrated <= files, f"PAIRS names sources that do not exist: {sorted(migrated - files)}"
    assert excluded <= files, f"EXCLUDED_FILES names sources that do not exist: {sorted(excluded - files)}"
    overlaps = (migrated & excluded) | (migrated & trees) | (excluded & trees)
    assert not overlaps, f"a file is classified twice: {sorted(overlaps)}"
    assert migrated | excluded | trees == files, (
        f"unaccounted: {sorted(files - (migrated | excluded | trees))}"
    )
    # The numbers this file's header quotes.
    assert len(files) == 107
    assert len(migrated) == 31


def test_every_whole_tree_exclusion_actually_matches_something() -> None:
    files = _source_files()
    silent = [prefix for prefix in EXCLUDED_TREES if not any(p.startswith(prefix) for p in files)]
    assert not silent, f"EXCLUDED_TREES entries that match no source file (stale): {silent}"


@pytest.mark.parametrize("table", ["EXCLUDED_TREES", "EXCLUDED_FILES"])
def test_every_exclusion_gives_a_reason_a_reader_can_act_on(table: str) -> None:
    entries = EXCLUDED_TREES if table == "EXCLUDED_TREES" else EXCLUDED_FILES
    thin = [key for key, reason in entries.items() if len(reason) < 30]
    assert not thin, f"{table} entries without a real reason: {thin}"


#: What ``MIGRATION_LOG.md`` §3's results table quotes.  Recomputed below, not typed from memory.
AUDIT_COUNTS = {
    "pairs": 31,
    "definitions": 415,
    "definitions_identical": 401,
    "definitions_explained": 14,
    "module_items": 521,
    "module_items_identical": 510,
    "module_items_explained": 11,
    "decorated": 456,
    "calls": 1045,
}


def _audit_counts() -> Dict[str, int]:
    counts = dict.fromkeys(AUDIT_COUNTS, 0)
    counts["pairs"] = len(PAIRS)
    for source_rel, target_rel in PAIRS:
        before = _definitions(SOURCE_PKG / source_rel)
        after = _definitions(SRC_PKG / target_rel)
        for name, node in before.items():
            counts["definitions"] += 1
            other = after.get(name)
            if other is not None and _tokens(node) == _tokens(other):
                counts["definitions_identical"] += 1
            else:
                counts["definitions_explained"] += 1
        counts["decorated"] += len(_decorators(SOURCE_PKG / source_rel))
        counts["calls"] += len(_calls(ast.parse((SOURCE_PKG / source_rel).read_text(encoding="utf-8"))))
    for source_rel, (before, after) in MODULE_ITEMS.items():
        for key, value in before.items():
            counts["module_items"] += 1
            if key in after and after[key] == value:
                counts["module_items_identical"] += 1
            else:
                counts["module_items_explained"] += 1
    return counts


def test_the_numbers_the_migration_log_quotes_are_these_numbers() -> None:
    """The log's results table is evidence; evidence that is typed by hand rots silently."""
    assert _audit_counts() == AUDIT_COUNTS


def test_the_migration_log_heading_quotes_these_counts() -> None:
    """§2.1's heading is the first number a reader sees; it has to be the computed one."""
    log = (ROOT / "MIGRATION_LOG.md").read_text(encoding="utf-8")
    heading = next(
        line for line in log.splitlines() if line.startswith("## 2.1")
    )
    counts = _audit_counts()
    expected = (
        f"定义级 {counts['definitions_identical']}/{counts['definitions']}"
        f" + 模块级 {counts['module_items_identical']}/{counts['module_items']}"
        f" + 装饰器 {counts['decorated']}/{counts['decorated']}"
        f" + 调用级 {counts['calls']}"
    )
    assert expected in heading, f"§2.1 heading is stale: {heading!r} should contain {expected!r}"


def test_the_allowlists_are_exactly_consumed() -> None:
    """A stale entry is worse than no entry: it silently covers the *next* change to that body.

    ``vllm/ops/install.py::_resolve_families`` sat in EXPECTED_DIFFERENCES after its extraction had
    been re-aligned byte for byte, so the table claimed a difference that no longer existed - and
    would have absorbed any future rewrite of that function.  Both allowlists are now required to be
    used exactly once each, no more and no less.
    """
    real = set()
    for source_rel, target_rel in PAIRS:
        before = _definitions(SOURCE_PKG / source_rel)
        after = _definitions(SRC_PKG / target_rel)
        for name, node in before.items():
            other = after.get(name)
            if other is None or _tokens(node) != _tokens(other):
                real.add(f"{source_rel}::{name}")
    assert real == set(EXPECTED_DIFFERENCES), (
        "EXPECTED_DIFFERENCES and the real differences disagree.\n"
        f"  allowlisted but unchanged (delete them): {sorted(set(EXPECTED_DIFFERENCES) - real)}\n"
        f"  changed but not allowed (explain them): {sorted(real - set(EXPECTED_DIFFERENCES))}"
    )
    real_module = set()
    for source_rel, (before, after) in MODULE_ITEMS.items():
        for key, value in before.items():
            if key not in after or after[key] != value:
                real_module.add(f"{source_rel}::{key}")
    assert real_module == set(EXPECTED_MODULE_LEVEL), (
        "EXPECTED_MODULE_LEVEL and the real differences disagree.\n"
        f"  allowlisted but unchanged: {sorted(set(EXPECTED_MODULE_LEVEL) - real_module)}\n"
        f"  changed but not allowed: {sorted(real_module - set(EXPECTED_MODULE_LEVEL))}"
    )


def test_no_migrated_file_is_also_declared_as_a_rewritten_init() -> None:
    """A copied ``__init__`` would be compared body-by-body; a rewritten one must not pretend."""
    migrated = _migrated_sources()
    init_like = sorted(path for path in migrated if path.endswith("__init__.py"))
    assert init_like == ["vllm/ops/families/__init__.py", "vllm/profiles/__init__.py"], (
        "which package __init__s are verbatim copies is now this list: " + repr(init_like)
    )
