"""The boundary, proved instead of promised: this package contains no operator math.

The objective's line is "device and execution semantics, not operator math".  A README sentence is
not a proof, and neither is a code review: the tempting edit is always the small inline fallback
("just this one conv1d"), and it looks like a fix.  So the boundary is checked mechanically, four
ways:

1. **No kernel source files** — only ``.py`` under ``src/``; no ``.cu``/``.cpp``/``.cce``/``.tile``.
2. **No kernel-authoring DSL** — an AST scan for ``tilelang.jit``, ``triton.jit``, ``T.prim_func``,
   ``T.Kernel``, ``torch.utils.cpp_extension.load_inline`` and friends.  Docstrings may *mention*
   them (provenance), code may not *use* them.
3. **No operator implementation is defined** — a curated list of the operator entry points the
   ledger tracks (``fused_experts``, ``causal_conv1d_fn``, ``chunk_gated_delta_rule``, ...) must not
   be *defined* here.  ``fused_halves`` and ``fused_kv_shape`` are page-geometry helpers whose names
   merely start with "fused"; the list is exact so they are not confused with kernels.
4. **The ports only forward** — every function in :mod:`ascend_platform.contracts.ports` that reaches
   the operator layer has a body of exactly one ``return <call>``.  That is what makes "the platform
   calls kernels through an interface" true rather than aspirational: a forwarder cannot accumulate
   math without failing this test.

The fourth check is the one with teeth.  Checks 1–3 catch the obvious move (paste the kernel);
check 4 catches the subtle one (grow a fallback inside the forwarding function, where nobody looks).
"""

from __future__ import annotations

import ast
import pathlib
from typing import Iterator, List, Set, Tuple

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "ascend_platform"


def _files() -> Iterator[pathlib.Path]:
    yield from sorted(PKG.rglob("*.py"))


def _trees() -> Iterator[Tuple[pathlib.Path, ast.Module]]:
    for path in _files():
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# ---------------------------------------------------------------------------------------------
# 1. no kernel source files
# ---------------------------------------------------------------------------------------------


def test_src_contains_only_python() -> None:
    others = [
        p.relative_to(ROOT)
        for p in PKG.rglob("*")
        if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in p.parts
    ]
    assert not others, "kernel sources / binaries are not part of this package: " + ", ".join(
        map(str, others)
    )


# ---------------------------------------------------------------------------------------------
# 2. no kernel-authoring DSL
# ---------------------------------------------------------------------------------------------

#: ``module.attribute`` chains that only appear when someone is *authoring* a kernel.
FORBIDDEN_ATTR_CHAINS = {
    ("tilelang", "jit"),
    ("tl", "jit"),
    ("triton", "jit"),
    ("T", "prim_func"),
    ("T", "Kernel"),
    ("tvm", "te"),
    ("torch", "utils", "cpp_extension"),
}

FORBIDDEN_CALL_NAMES = {"load_inline", "load", "prim_func", "jit", "alloc_shared", "alloc_fragment"}


def _dotted(node: ast.AST) -> List[str]:
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return list(reversed(parts))


def _dsl_uses() -> List[str]:
    found = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                decorators = list(node.decorator_list)
            elif isinstance(node, ast.Call):
                decorators = [node.func]
            else:
                continue
            for target in decorators:
                chain = _dotted(target)
                if not chain:
                    continue
                if tuple(chain[:3]) in FORBIDDEN_ATTR_CHAINS or tuple(chain[:2]) in FORBIDDEN_ATTR_CHAINS:
                    found.append(f"{path.relative_to(ROOT)}:{node.lineno} uses {'.'.join(chain)}")
    return found


def test_no_kernel_authoring_dsl_is_used() -> None:
    assert not _dsl_uses(), "this package does not author kernels:\n" + "\n".join(_dsl_uses())


# ---------------------------------------------------------------------------------------------
# 3. no operator implementation is defined
# ---------------------------------------------------------------------------------------------

#: Entry points the op-coverage ledger tracks.  Exact names — see the module docstring for why.
OPERATOR_ENTRY_POINTS = {
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "chunk_local_cumsum",
    "fused_experts",
    "fused_moe_impl",
    "fused_recurrent_gated_delta_rule",
    "fused_sigmoid_gating_delta_rule_update",
    "gdn_prefill_post_conv",
    "l2norm_fwd",
    "moe_align_block_size",
    "paged_decode",
    "paged_prefill",
    "rms_norm_forward",
    "silu_and_mul",
    "topk_softmax",
}


def _defined_names() -> Set[str]:
    names: Set[str] = set()
    for _path, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
    return names


def test_no_operator_implementation_is_defined_here() -> None:
    defined = _defined_names() & OPERATOR_ENTRY_POINTS
    assert not defined, (
        "these are operator implementations and belong in the operator layer: "
        + ", ".join(sorted(defined))
        + " (the platform reaches them through ascend_platform.contracts.ports)"
    )


# ---------------------------------------------------------------------------------------------
# 4. the ports only forward
# ---------------------------------------------------------------------------------------------

PORTS = PKG / "contracts" / "ports.py"


def _wrappers() -> List[ast.FunctionDef]:
    """The pass-through kernel wrappers: ``def f(*args, **kwargs)``."""
    tree = ast.parse(PORTS.read_text(encoding="utf-8"), filename=str(PORTS))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        if args.vararg is not None and args.kwarg is not None and not args.args:
            found.append(node)
    return found


def _body_statements(func: ast.FunctionDef) -> List[ast.stmt]:
    """The executable body, minus docstrings and bare-constant statements."""
    return [
        stmt
        for stmt in func.body
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
    ]


def test_the_ports_only_forward() -> None:
    """A forwarder is one ``return``; math growing inside it is the boundary leaking.

    Only the ``(*args, **kwargs)`` wrappers are held to this — they are the kernel call sites.
    The rest of ``ports`` is plumbing (resolve the root, describe the layer, hand back a module) and
    is judged by checks 1-3 like any other code.
    """
    wrappers = _wrappers()
    assert len(wrappers) >= 8, f"only {len(wrappers)} forwarding wrappers found; the scan is broken"
    offenders = []
    for func in wrappers:
        body = _body_statements(func)
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            offenders.append(f"{func.name} (line {func.lineno})")
            continue
        reached = ", ".join(ast.unparse(node) for node in ast.walk(body[0]) if isinstance(node, ast.Call))
        if "operator_layer" not in reached and "ops_module" not in reached:
            offenders.append(f"{func.name} (line {func.lineno}, does not reach the layer)")
    assert not offenders, (
        "these ports do more than forward; the math belongs in the operator layer: "
        + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------------------------
# 5. against the operator repo itself: what lives there is not here
# ---------------------------------------------------------------------------------------------

SOURCE_PKG = pathlib.Path(
    __import__("os").environ.get("TILEINFER_SRC", "/home/userlist/zxs/TileInfer/tileinfer")
)

#: Names generic enough to appear in both trees without meaning anything.
GENERIC = {"__init__.py", "base.py", "utils.py", "types.py", "config.py", "registry.py"}


@pytest.mark.skipif(not SOURCE_PKG.is_dir(), reason=f"operator repo not present at {SOURCE_PKG}")
def test_the_operator_and_model_subtrees_were_not_copied() -> None:
    copied = []
    for sub in ("kernels", "models", "attention", "integrations"):
        tree = SOURCE_PKG / sub
        if not tree.is_dir():
            continue
        for source in tree.rglob("*.py"):
            if source.name in GENERIC:
                continue
            if list(PKG.rglob(source.name)):
                copied.append(f"{source.relative_to(SOURCE_PKG)} -> {source.name}")
    assert not copied, (
        "these files belong to the operator layer and must stay there: " + ", ".join(copied)
    )
