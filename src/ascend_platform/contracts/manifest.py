"""The operator layer's **pure-data manifest**: what it publishes, and how the platform reads it.

Why data and not imports
------------------------
The platform must decide "who serves this name" at start-up, on a CPU box, without pulling the
operator implementation (and therefore ``torch_npu``/``tilelang``) into the process.  A JSON file
shipped inside the operator distribution does that: it is inspectable, diffable, and it fails
validation in a unit test rather than at the first decode step.

Validation is deliberately strict about the two rules the source repo already enforces:

* an op without a ``measured_ratio`` **and** a bench script may not appear (an unmeasured
  implementation may not be enabled by default);
* ``graph_capturable`` must be stated, because a non-capturable op inside a captured region
  produces silently wrong numerics on this platform.

Layout (schema 1)::

    {"schema": 1,
     "library": {"name": "ascend-attention", "version": "0.3.0",
                 "requires_platform": ">=0.1,<0.2"},
     "ops": [{"name": "paged_decode_attention",
              "shapes": {"head_dim": [128, 256], "group": ">=2", "page": [128]},
              "dtypes": ["bfloat16", "float16"],
              "kv_layouts": ["ND"],
              "graph_capturable": true,
              "measured_ratio": 1.42,
              "bench": "benchmarks/bench_attention.py"}]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .version import InvalidVersion, VersionRange, check_range

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_SCHEMA",
    "ManifestError",
    "OpEntry",
    "OpManifest",
    "parse_manifest",
    "load_manifest",
    "load_manifest_text",
    "check_compatibility",
]

MANIFEST_SCHEMA_VERSION = 1

#: A compact JSON-Schema document, shipped so the operator layer can validate what it writes and so
#: a third party can check a manifest without importing this package.
MANIFEST_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AscendPlatform operator manifest",
    "type": "object",
    "required": ["schema", "library", "ops"],
    "properties": {
        "schema": {"const": MANIFEST_SCHEMA_VERSION},
        "library": {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "requires_platform": {"type": "string"},
            },
        },
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "dtypes", "kv_layouts", "graph_capturable"],
                "properties": {
                    "name": {"type": "string"},
                    "shapes": {"type": "object"},
                    "dtypes": {"type": "array", "items": {"type": "string"}},
                    "kv_layouts": {"type": "array", "items": {"type": "string"}},
                    "graph_capturable": {"type": "boolean"},
                    "measured_ratio": {"type": "number", "exclusiveMinimum": 0},
                    "bench": {"type": "string"},
                },
            },
        },
    },
}


class ManifestError(ValueError):
    """The manifest is malformed, or violates one of the two rules above."""


@dataclass(frozen=True)
class OpEntry:
    """One operation as the operator layer declares it — data, never code.

    ``shapes``/``dtypes``/``kv_layouts`` are the *guarantees* the platform plans against (what the
    op accepts), ``graph_capturable`` is the capture verdict, and ``measured_ratio`` is somebody
    else's measurement.  There is deliberately no callable in here: a manifest has to be readable by
    a machine that does not have the operator layer installed.
    """

    name: str
    shapes: Mapping[str, Any] = field(default_factory=dict)
    dtypes: Tuple[str, ...] = ()
    kv_layouts: Tuple[str, ...] = ()
    graph_capturable: bool = False
    measured_ratio: Optional[float] = None
    bench: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shapes": dict(self.shapes),
            "dtypes": list(self.dtypes),
            "kv_layouts": list(self.kv_layouts),
            "graph_capturable": self.graph_capturable,
            "measured_ratio": self.measured_ratio,
            "bench": self.bench,
        }


@dataclass(frozen=True)
class OpManifest:
    """A whole operator library: ``library``/``version`` plus one :class:`OpEntry` per op.

    ``requires_platform`` is the range this library was built for; :func:`check_compatibility`
    compares it with our own version at start-up, so a half-upgraded machine fails here instead of
    taking a wrong code path three hours later.
    """

    library: str
    version: str
    ops: Tuple[OpEntry, ...]
    requires_platform: Optional[str] = None

    def names(self) -> Tuple[str, ...]:
        return tuple(op.name for op in self.ops)

    def op(self, name: str) -> OpEntry:
        for entry in self.ops:
            if entry.name == name:
                return entry
        raise KeyError(f"manifest has no op {name!r}; known: {', '.join(self.names())}")

    def capturable(self) -> Tuple[OpEntry, ...]:
        return tuple(op for op in self.ops if op.graph_capturable)


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ManifestError(f"{where}: missing required field {key!r}")
    return mapping[key]


def parse_manifest(data: Mapping[str, Any]) -> OpManifest:
    """Validate ``data`` and return a typed manifest; every violation names its op."""
    if not isinstance(data, Mapping):
        raise ManifestError(f"manifest must be an object, got {type(data).__name__}")
    schema = _require(data, "schema", "manifest")
    if schema != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"manifest schema {schema!r} is not supported (this build reads "
            f"{MANIFEST_SCHEMA_VERSION}); see docs/op-manifest.md"
        )
    library = _require(data, "library", "manifest")
    if not isinstance(library, Mapping):
        raise ManifestError("manifest.library must be an object")
    name = str(_require(library, "name", "manifest.library"))
    version = str(_require(library, "version", "manifest.library"))
    requires_platform = library.get("requires_platform")
    if requires_platform is not None:
        # The range is the *other* repository's data, so a bad one is a manifest error, not a
        # crash in ``check_compatibility`` (whose callers catch ``ManifestError``/``IncompatibleVersion``).
        try:
            VersionRange.parse(str(requires_platform))
        except InvalidVersion as exc:
            raise ManifestError(
                f"manifest.library.requires_platform {requires_platform!r} is not a range this build "
                f"can read ({exc}); see docs/op-manifest.md"
            ) from None

    raw_ops = _require(data, "ops", "manifest")
    if not isinstance(raw_ops, Sequence) or isinstance(raw_ops, (str, bytes)):
        raise ManifestError("manifest.ops must be an array")

    ops: List[OpEntry] = []
    seen: set = set()
    for index, raw in enumerate(raw_ops):
        where = f"manifest.ops[{index}]"
        if not isinstance(raw, Mapping):
            raise ManifestError(f"{where}: must be an object")
        op_name = str(_require(raw, "name", where))
        where = f"manifest.ops[{op_name}]"
        if op_name in seen:
            raise ManifestError(f"{where}: duplicate op name")
        seen.add(op_name)

        dtypes = tuple(str(x) for x in _require(raw, "dtypes", where))
        kv_layouts = tuple(str(x) for x in _require(raw, "kv_layouts", where))
        if not dtypes:
            raise ManifestError(f"{where}: dtypes must not be empty")
        if not kv_layouts:
            raise ManifestError(f"{where}: kv_layouts must not be empty")

        capturable = _require(raw, "graph_capturable", where)
        if not isinstance(capturable, bool):
            raise ManifestError(
                f"{where}: graph_capturable must be a bool; a missing answer is what produces "
                "silently wrong numerics inside a captured region"
            )

        ratio = raw.get("measured_ratio")
        bench = raw.get("bench")
        if ratio is None or bench is None:
            raise ManifestError(
                f"{where}: needs both measured_ratio and bench. An implementation without a "
                "recorded measurement may not be advertised (repo rule: no ratio, no mainline)"
            )
        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            raise ManifestError(f"{where}: measured_ratio must be a number, got {ratio!r}") from None
        if ratio <= 0:
            raise ManifestError(f"{where}: measured_ratio must be > 0, got {ratio}")

        shapes = raw.get("shapes", {})
        if not isinstance(shapes, Mapping):
            raise ManifestError(f"{where}: shapes must be an object")

        ops.append(
            OpEntry(
                name=op_name,
                shapes=dict(shapes),
                dtypes=dtypes,
                kv_layouts=kv_layouts,
                graph_capturable=capturable,
                measured_ratio=ratio,
                bench=str(bench),
            )
        )

    return OpManifest(
        library=name,
        version=version,
        ops=tuple(ops),
        requires_platform=str(requires_platform) if requires_platform is not None else None,
    )


def load_manifest_text(text: str) -> OpManifest:
    """Parse a manifest that is already in memory (a fixture, an env var, a test).

    Invalid JSON raises :class:`ManifestError`; invalid structure raises from
    :func:`parse_manifest`, so callers have one exception type to catch either way.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from None
    return parse_manifest(data)


def load_manifest(path: str) -> OpManifest:
    """Parse a manifest file (UTF-8 JSON), with the same error contract as the text form."""
    with open(path, encoding="utf-8") as handle:
        return load_manifest_text(handle.read())


def check_compatibility(
    manifest: OpManifest,
    *,
    platform_version: str,
    device: Any = None,
) -> None:
    """Refuse a manifest this platform cannot serve, or one whose op surface the device lacks.

    Called once at start-up, before any name is bound: the whole point is that a mismatch is a
    start-up failure with a readable reason, not a wrong number three hours later.
    """
    if manifest.requires_platform:
        check_range(
            platform_version,
            manifest.requires_platform,
            what=f"operator layer {manifest.library}=={manifest.version}",
        )
    if device is not None:
        known = set(getattr(device, "dtypes", ()))
        layouts = set(getattr(device, "layouts", ()))
        for op in manifest.ops:
            if known and not (set(op.dtypes) & known):
                raise ManifestError(
                    f"{op.name}: none of its dtypes {sorted(set(op.dtypes))} are supported by "
                    f"{getattr(device, 'name', device)} {sorted(known)}"
                )
            if layouts and not (set(op.kv_layouts) & layouts):
                raise ManifestError(
                    f"{op.name}: none of its KV layouts {sorted(set(op.kv_layouts))} are supported "
                    f"by {getattr(device, 'name', device)} {sorted(layouts)}"
                )
