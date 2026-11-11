"""``ascend-platform doctor`` — the bring-up checklist as one command.

The implementation is :mod:`ascend_platform.doctor.cli`; this package only re-exports the entry
points so ``python -m ascend_platform.doctor`` and the console script agree.
"""

from __future__ import annotations

from .cli import Check, main, format_report, tail_line  # noqa: F401

__all__ = ["Check", "main", "format_report", "tail_line"]
