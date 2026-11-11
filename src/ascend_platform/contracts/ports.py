"""The operator layer, as this platform sees it — the only legal dependency direction.

Model
-----
``platform → operators`` (never the reverse).  The platform may ask *which* implementation serves a
name and *whether* it can enter a graph; it may not contain the math.  The operator layer is
optional: with nothing installed, names keep their vendor implementation and the gap is reported,
which is what keeps "no operator layer, still serves" true.

Why a lazy facade instead of a direct import
--------------------------------------------
The source repo imported ``tileinfer.attention.api`` at module scope in the attention impl, and
``tileinfer.plan`` at module scope in the prefill path.  That makes the platform unimportable
without the operator layer — unacceptable for a repo whose whole point is the seam.  Every symbol
below is therefore a thin function that resolves on **first call**, not on import.

Two implementations exist during the migration: the current operator layer (``tileinfer``) and the
target one (``ascend_attention``).  ``ASCEND_PLATFORM_OPS`` pins one explicitly; otherwise the first
importable root wins, so a machine with both installed prefers the target.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = [
    "OperatorLayerUnavailable",
    "ENV_OPS_LAYER",
    "OPERATOR_ROOTS",
    "operator_root",
    "operator_available",
    "operator_version",
    "check_operator_version",
    "operator_info",
    "operator_layer",
    "register_operator_layout",
    "reset_operator_layer",
    "BatchAttention",
    "RaggedMetadata",
    "PageTable",
    "TileSchedule",
    "plan_query_tiles_from_host",
    "pack_indices",
    "forward_prefill",
    "reference_moe",
    "ops_module",
]

#: Pin the operator layer explicitly (``tileinfer`` or ``ascend_attention``).
ENV_OPS_LAYER = "ASCEND_PLATFORM_OPS"

#: Candidate roots, in preference order: the target layer first, the migration source second.
OPERATOR_ROOTS: Tuple[str, ...] = ("ascend_attention", "tileinfer")

#: The logical modules the platform may ask for; a layout must name every one of them.
_REQUIRED_LOGICAL: Tuple[str, ...] = (
    "attention",
    "metadata",
    "plan",
    "decode_kernel",
    "testing",
    "ops",
)

#: Logical module name -> dotted path under the operator layer's root.  The two layers differ only
#: in where the attention API sits (``attention.api`` vs ``api``); keeping the difference here means
#: the rest of the platform never learns either layout.
_LAYOUTS: Dict[str, Mapping[str, str]] = {
    "ascend_attention": {
        "attention": "api",
        "metadata": "metadata",
        "plan": "plan",
        "decode_kernel": "kernels.attention.paged_decode_ascend950",
        "testing": "testing.moe_reference",
        "ops": "ops",
    },
    "tileinfer": {
        "attention": "attention.api",
        "metadata": "metadata",
        "plan": "plan",
        "decode_kernel": "kernels.attention.paged_decode_ascend950",
        "testing": "testing.moe_reference",
        "ops": "vllm.ops",
    },
}


def register_operator_layout(root: str, layout: Mapping[str, str]) -> None:
    """Teach the port how a third-party operator package is laid out.

    ``layout`` maps the logical names the platform asks for (``attention``, ``metadata``,
    ``plan``, ``decode_kernel``, ``testing``, ``ops``) to dotted paths under ``root``.  A package
    with a different internal structure becomes usable without editing the platform — which is
    what makes "the operator layer is swappable" true rather than aspirational.
    """
    missing = set(_REQUIRED_LOGICAL) - set(layout)
    if missing:
        raise ValueError(f"layout for {root!r} is missing {sorted(missing)}")
    unknown = set(layout) - set(_REQUIRED_LOGICAL)
    if unknown:
        raise ValueError(f"layout for {root!r} has unknown entries {sorted(unknown)}")
    _LAYOUTS[root] = dict(layout)


class OperatorLayerUnavailable(RuntimeError):
    """No operator layer is importable, and the caller asked for something that needs one."""


class _Layer:
    """A resolved operator root: imports its submodules on demand, once each."""

    def __init__(self, root: str) -> None:
        self.root = root
        self._modules: Dict[str, Any] = {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<operator layer {self.root!r}>"

    def module(self, logical: str) -> Any:
        if logical in self._modules:
            return self._modules[logical]
        layout = _LAYOUTS.get(self.root)
        if layout is None or logical not in layout:
            raise OperatorLayerUnavailable(
                f"operator layer {self.root!r} declares no {logical!r} module; "
                f"add it to contracts/ports.py::_LAYOUTS"
            )
        module = importlib.import_module(f"{self.root}.{layout[logical]}")
        self._modules[logical] = module
        return module

    def get(self, logical: str, name: str) -> Any:
        module = self.module(logical)
        try:
            return getattr(module, name)
        except AttributeError:
            raise OperatorLayerUnavailable(
                f"operator layer {self.root!r} has no {logical}.{name}; the platform and the "
                f"operator layer are out of step (see docs/versioning.md)"
            ) from None

    def version(self) -> str:
        try:
            module = importlib.import_module(self.root)
        except Exception:  # noqa: BLE001 - version is informational
            return "unknown"
        return str(getattr(module, "__version__", "unknown"))


_LAYER: Optional[_Layer] = None
_Failure: Optional[str] = None


def _candidates() -> Tuple[str, ...]:
    pinned = os.environ.get(ENV_OPS_LAYER, "").strip()
    if pinned:
        return (pinned,)
    return OPERATOR_ROOTS


def operator_root() -> str:
    """The dotted root of the resolved operator layer, or ``""`` when none is importable."""
    layer = operator_layer(required=False)
    return "" if layer is None else layer.root


def operator_available() -> bool:
    """Whether an operator layer is importable — the tolerant form of :func:`operator_layer`.

    Platform start-up uses this to decide "bind nothing and say so" instead of raising.
    """
    return operator_layer(required=False) is not None


def operator_layer(required: bool = True) -> Optional[_Layer]:
    """Resolve (once) and return the operator layer, or ``None`` when none is importable."""
    global _LAYER, _Failure
    if _LAYER is not None:
        return _LAYER
    for root in _candidates():
        try:
            importlib.import_module(root)
        except Exception as exc:  # noqa: BLE001 - "not installed" is the normal case
            _Failure = f"{root}: {type(exc).__name__}: {exc}"
            continue
        _LAYER = _Layer(root)
        return _LAYER
    if not required:
        return None
    raise OperatorLayerUnavailable(
        "no operator layer is importable. Install one (``ascend_attention`` or ``tileinfer``) or "
        f"set {ENV_OPS_LAYER} to its import root. Tried: {', '.join(_candidates())}."
        + (f" Last error: {_Failure}" if _Failure else "")
    )


def reset_operator_layer() -> None:
    """Forget the resolved layer (tests, and after installing a layer in-process)."""
    global _LAYER, _Failure
    _LAYER = None
    _Failure = None


def operator_version() -> str:
    """The operator layer's declared version, or ``"unknown"`` when it declares none."""
    layer = operator_layer(required=False)
    return layer.version() if layer is not None else "unknown"


def check_operator_version() -> str:
    """Validate the operator layer against :data:`SUPPORTED_OPERATOR_RANGE`; return the version.

    Raises :class:`~ascend_platform.contracts.version.IncompatibleVersion` when the installed
    operator layer is outside the range this platform was written against (including when it
    declares no version at all).  The point is the same one the manifest check makes: a mismatch
    must be a start-up refusal with two numbers in it, not an ``AttributeError`` three hours later.
    """
    from .version import SUPPORTED_OPERATOR_RANGE, check_range

    layer = operator_layer(required=False)
    if layer is None:
        raise OperatorLayerUnavailable(
            "no operator layer is importable, so its version cannot be checked; install one or set "
            f"{ENV_OPS_LAYER} to its import root"
        )
    found = layer.version()
    check_range(None if found == "unknown" else found, SUPPORTED_OPERATOR_RANGE, what=f"operator layer {layer.root!r}")
    return found


def operator_info() -> Dict[str, Any]:
    """A JSON-able description for ``doctor`` and the boot banner."""
    layer = operator_layer(required=False)
    if layer is None:
        return {"available": False, "root": None, "version": None, "tried": list(_candidates())}
    return {"available": True, "root": layer.root, "version": layer.version(),
            "tried": list(_candidates())}


# ---------------------------------------------------------------------------------------------
# The public surface, one thin wrapper per symbol the platform actually uses.
#
# These are functions (not aliases) on purpose: an alias would resolve at import of the *calling*
# module, which is exactly the eager-import trap described in the module docstring.  Annotations in
# the callers are strings (``from __future__ import annotations``), so a function is a drop-in for a
# class in every position the platform uses one.
# ---------------------------------------------------------------------------------------------


def BatchAttention(*args: Any, **kwargs: Any) -> Any:
    """``BatchAttention(...)`` from the operator layer: the one object with ``plan`` / ``run``."""
    return operator_layer().get("attention", "BatchAttention")(*args, **kwargs)


def RaggedMetadata(*args: Any, **kwargs: Any) -> Any:
    """Serving-native batch description (pure data) — the kernels' input vocabulary."""
    return operator_layer().get("metadata", "RaggedMetadata")(*args, **kwargs)


def PageTable(*args: Any, **kwargs: Any) -> Any:
    """Dense ``[batch, max_pages]`` block-table form, convertible to ragged."""
    return operator_layer().get("metadata", "PageTable")(*args, **kwargs)


def TileSchedule(*args: Any, **kwargs: Any) -> Any:
    """The prefill work schedule (pure int32 data) returned by the planner."""
    return operator_layer().get("plan", "TileSchedule")(*args, **kwargs)


def plan_query_tiles_from_host(*args: Any, **kwargs: Any) -> Any:
    """The host-side prefill planner: query tiles for a batch, as plain integers."""
    return operator_layer().get("plan", "plan_query_tiles_from_host")(*args, **kwargs)


def pack_indices(*args: Any, **kwargs: Any) -> Any:
    """Pad a page-index vector into the kernel's fixed-size ABI slot."""
    return operator_layer().get("decode_kernel", "_pack_indices")(*args, **kwargs)


def forward_prefill(*args: Any, **kwargs: Any) -> Any:
    """The prefill/append kernel entry.  The one place a kernel is called by name."""
    return operator_layer().get("decode_kernel", "forward_prefill")(*args, **kwargs)


def reference_moe(*args: Any, **kwargs: Any) -> Any:
    """The pure-torch MoE oracle used to certify a candidate implementation."""
    return operator_layer().get("testing", "reference_moe")(*args, **kwargs)


def ops_module(name: str) -> Any:
    """A submodule of the operator layer's operator package (``"moe"``, ``"fla.chunk"``, …)."""
    layer = operator_layer()
    return layer.module("ops").__dict__ and importlib.import_module(
        f"{layer.module('ops').__name__}.{name}"
    )
