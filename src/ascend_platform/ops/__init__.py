"""Operator-name rebinding: the platform half of "which implementation serves this name".

`install`    the mechanism (``OpBinding``, the patch records, the per-call switch wrapper).
`switch`     which families are allowed to wire (``ASCEND_PLATFORM_OPS`` / legacy ``TILEINFER_OPS``).
`candidates` several implementations per name, chosen per call by a cheap predicate.
`families`   the per-model tables: which vLLM name maps to which operator, and where it comes from.

The tables live here (rebinding vLLM's names is *engine* knowledge, see the repo README); the
implementations they point at live in the operator layer and are imported lazily, through
:mod:`ascend_platform.contracts.ports`.  That split is what lets the platform be installed with no
operator layer at all — a name simply keeps its vendor implementation and the gap is reported.
"""

from __future__ import annotations

__all__: list[str] = []
