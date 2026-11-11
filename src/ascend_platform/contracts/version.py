"""Version ranges, and the start-up check that makes a mismatch loud.

The rule the repo promises: two repositories declare compatible ranges against each other and the
platform verifies them **at start-up**; a mismatch refuses to start rather than running quietly.
This module is the whole mechanism — a tiny range grammar (``>=0.23,<0.24``), not a dependency on
``packaging``, because the platform should stay importable with nothing but the standard library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SUPPORTED_VLLM_RANGE",
    "SUPPORTED_OPERATOR_RANGE",
    "VersionRange",
    "InvalidVersion",
    "IncompatibleVersion",
    "check_range",
]

#: The engine line these adapters are written against (vLLM 0.23).
SUPPORTED_VLLM_RANGE = ">=0.23,<0.24"

#: The operator layer version this platform was written against.  Enforced at start-up by
#: ``ports.check_operator_version()`` (called from ``ops.install_ops`` before the first binding) and
#: surfaced by ``doctor``'s ``ops.version`` check - see docs/versioning.md.
SUPPORTED_OPERATOR_RANGE = ">=0.1,<0.2"

#: ``v`` prefix, ``.``/``-``/``_`` separators and a PEP 440 local segment are cosmetic: a wheel built
#: from a git checkout reports ``0.1.0.dev0+git1234`` and a locally built vLLM reports
#: ``0.23.0+cu123``.  Neither is a different version line, and refusing to start over a suffix would
#: be a false negative at exactly the moment the operator is trying to bring the box up.
_VERSION_RE = re.compile(r"^[vV]?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[._-]?(?:dev|rc|a|b|post)\d*)?$")
_CLAUSE_RE = re.compile(r"^(>=|<=|==|!=|>|<)\s*(\S+)$")


class InvalidVersion(ValueError):
    """The string is not a version this module understands."""


class IncompatibleVersion(RuntimeError):
    """A declared range does not admit the version actually installed."""


def _parse(version: str) -> Tuple[int, ...]:
    text = str(version).strip()
    text = text.split("+", 1)[0]  # local version segment: cosmetic, not a different release line
    match = _VERSION_RE.match(text)
    if not match:
        raise InvalidVersion(f"cannot parse version {version!r} (expected e.g. 0.23.0)")
    parts = [int(group) if group else 0 for group in match.groups()]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


@dataclass(frozen=True)
class VersionRange:
    """A comma-separated conjunction of ``>=``/``<=``/``==``/``!=``/``>``/``<`` clauses."""

    spec: str
    clauses: Tuple[Tuple[str, Tuple[int, ...]], ...]

    @classmethod
    def parse(cls, spec: str) -> "VersionRange":
        text = str(spec).strip()
        if not text:
            raise InvalidVersion("empty version range")
        clauses: List[Tuple[str, Tuple[int, ...]]] = []
        for raw in text.split(","):
            clause = raw.strip()
            if not clause:
                continue
            match = _CLAUSE_RE.match(clause)
            if not match:
                raise InvalidVersion(f"cannot parse clause {clause!r} of range {spec!r}")
            op, version = match.groups()
            clauses.append((op, _parse(version)))
        if not clauses:
            raise InvalidVersion(f"range {spec!r} has no clauses")
        return cls(spec=text, clauses=tuple(clauses))

    def contains(self, version: str) -> bool:
        candidate = _parse(version)
        for op, bound in self.clauses:
            if op == ">=" and not candidate >= bound:
                return False
            if op == ">" and not candidate > bound:
                return False
            if op == "<=" and not candidate <= bound:
                return False
            if op == "<" and not candidate < bound:
                return False
            if op == "==" and not candidate == bound:
                return False
            if op == "!=" and candidate == bound:
                return False
        return True

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.spec


def check_range(found: Optional[str], required: str, *, what: str) -> VersionRange:
    """Return the parsed range when ``found`` is admitted; raise :class:`IncompatibleVersion`.

    ``found=None`` means "the other side did not declare a version"; that is a *warning* case in
    the callers, so this function accepts it only when the range is unbounded — which the platform
    never declares, hence the explicit error.
    """
    rng = VersionRange.parse(required)  # a malformed *requirement* is our bug: let it raise
    if found is None:
        raise IncompatibleVersion(
            f"{what}: no version declared, but this build requires {required}; "
            "refusing to start rather than running an unknown combination"
        )
    try:
        admitted = rng.contains(found)
    except InvalidVersion as exc:
        # An unreadable *installed* version is a mismatch, not a crash: the caller's contract is
        # "raise IncompatibleVersion for any combination this build will not run", and the message
        # has to say which string it could not read.
        raise IncompatibleVersion(
            f"{what}: cannot read version {found!r} ({exc}); refusing to start rather than guessing "
            "(docs/versioning.md)"
        ) from None
    if not admitted:
        raise IncompatibleVersion(
            f"{what}: version {found} is outside the supported range {required}; "
            "refusing to start (docs/versioning.md)"
        )
    return rng
