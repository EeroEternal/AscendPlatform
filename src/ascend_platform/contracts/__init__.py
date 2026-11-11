"""The two contracts this repo owes the outside world, both as data.

``ports``
    How the platform reaches the operator layer: one object with ``plan`` / ``run``, the pure-data
    types it accepts, and the three kernel entry points that are still called directly.  Resolved
    lazily, so the platform can be installed **with no operator layer at all**.
``manifest``
    What the operator layer publishes: a JSON manifest of names, shape/dtype/layout constraints,
    graph-capturability and measured ratios.  The platform reads it at start-up to bind names and to
    refuse a version mismatch instead of silently running.
``version``
    Range parsing and the start-up check that makes a mismatch loud.

Nothing in this package imports ``vllm``, ``torch_npu`` or an operator implementation at module
load.
"""

from __future__ import annotations

from .ports import (
    OperatorLayerUnavailable,
    operator_available,
    operator_info,
    operator_layer,
)
from .version import VersionRange, SUPPORTED_VLLM_RANGE, check_range

__all__ = [
    "OperatorLayerUnavailable",
    "operator_available",
    "operator_info",
    "operator_layer",
    "VersionRange",
    "SUPPORTED_VLLM_RANGE",
    "check_range",
]
