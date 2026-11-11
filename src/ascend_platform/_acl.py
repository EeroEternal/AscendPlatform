"""The directory ACL: which layer may import which, and what must stay importable on a CPU box.

This module is *data*, not a checker: it says what the rules are, so the rules can be read in one
place and enforced from several (`tests/test_layer_acl.py`, `tests/test_package_layout.py`, and
``doctor`` when it explains an import failure).  Everything here is importable without torch, vLLM,
torch_npu or an operator package — that is what makes it usable as the reference.

Two kinds of rule
-----------------
**Direction** (:data:`ALLOWED_IMPORTS`)
    For a module in layer *L*, every ``ascend_platform.<T>`` import must have ``T`` in
    :func:`allowed` ``(L)``.  A layer may always import itself.  Dotted prefixes are allowed so a
    layer can depend on one subpackage instead of a whole one (``engine.compat``).
**Free-standing** (:data:`ENGINE_FREE`)
    Five layers must import with no engine, no NPU and no operator package installed.  The property
    is what makes the CPU test suite worth having: the day ``runtime/`` imports ``vllm``, the
    scheduling logic stops being testable on a laptop, and the failure is silent because the box it
    runs on has vLLM.

Why ``ops`` is engine-free even though it imports ``engine.compat``
-------------------------------------------------------------------
``engine/compat/patches.py`` is written to import and *skip* without vLLM (it resolves its targets
lazily), so the dependency does not leak the engine into a CPU-only process.  The direction rule
still records the one permitted edge, so a second one has to be argued for explicitly.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

__all__ = [
    "LAYERS",
    "ALLOWED_IMPORTS",
    "ENGINE_FREE",
    "FORBIDDEN_IN_ENGINE_FREE",
    "FUNCTION_ONLY_IMPORTS",
    "OPERATOR_ROOTS",
    "PORT_MODULE",
    "allowed",
    "describe",
]

#: The seven layers, in dependency order (leaves first).  ``_version``/``utils`` are module-level
#: and belong to no layer.
LAYERS: Tuple[str, ...] = (
    "contracts",
    "device",
    "profiles",
    "runtime",
    "ops",
    "engine",
    "doctor",
)

#: Layer -> the sibling layers it may import, as dotted prefixes.
#:
#: * ``contracts`` and ``device`` are leaves: data, parsing and a registry.
#: * ``profiles`` may read the device record and the kernel page size; it holds model facts.
#: * ``runtime`` is the engine-agnostic skeleton: no engine, no vLLM.
#: * ``ops`` wires names to implementations; its one engine edge is the patch status it reports.
#: * ``engine`` is the only layer allowed to import vLLM.
#: * ``doctor`` inspects everything, so it may import everything.
ALLOWED_IMPORTS: Dict[str, FrozenSet[str]] = {
    "contracts": frozenset(),
    "device": frozenset(),
    "profiles": frozenset({"device", "runtime.buckets"}),
    "runtime": frozenset(),
    "ops": frozenset({"contracts", "device", "profiles", "runtime", "engine.compat"}),
    "engine": frozenset({"contracts", "device", "profiles", "runtime", "ops"}),
    "doctor": frozenset({"contracts", "device", "engine", "ops", "profiles", "runtime"}),
}

#: Layers that must import with nothing but the standard library and torch.
ENGINE_FREE: FrozenSet[str] = frozenset({"contracts", "device", "profiles", "runtime", "ops"})

#: Module roots that may **never** be imported from an engine-free layer, at any nesting level
#: (``if TYPE_CHECKING:`` blocks are not an exception: use a ``Protocol`` or a string annotation).
FORBIDDEN_IN_ENGINE_FREE: Tuple[str, ...] = (
    "vllm",
    "vllm_ascend",
    "tilelang",
    "tileinfer",
    "ascend_attention",
)

#: Module roots allowed only **inside a function**, where they stay an optional capability probe.
FUNCTION_ONLY_IMPORTS: Tuple[str, ...] = ("torch_npu",)

#: Roots reached only through :data:`PORT_MODULE`.
OPERATOR_ROOTS: Tuple[str, ...] = ("tileinfer", "ascend_attention")

#: The single module allowed to import an operator package.
PORT_MODULE: str = "ascend_platform.contracts.ports"


def allowed(layer: str) -> FrozenSet[str]:
    """The sibling layers ``layer`` may import (dotted prefixes included)."""
    return ALLOWED_IMPORTS.get(layer, frozenset())


def describe() -> Tuple[Dict[str, object], ...]:
    """A JSON-able snapshot for ``doctor`` and for docs that must not drift."""
    return tuple(
        {
            "layer": layer,
            "may_import": sorted(ALLOWED_IMPORTS[layer]),
            "engine_free": layer in ENGINE_FREE,
        }
        for layer in LAYERS
    )
