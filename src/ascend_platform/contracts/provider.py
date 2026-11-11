"""The **provider contract**: how an operator package tells this platform what it serves.

The platform does not guess where an implementation lives.  It asks a *provider* - one package that
implements :func:`describe` (a pure-data descriptor) and :func:`load` (a callable per op name) - and
the provider answers.  ``ascend_attention`` is one such package; a user's own operators are another;
nothing in this module knows either name.

Why the indirection exists
--------------------------
Before this contract the platform resolved implementations by **hardcoded internal paths**
(``ascend_attention.ops.fla.gating``, decided in ``contracts/ports.py::_LAYOUTS``).  That makes one
package's directory structure a public interface, and it produced two failures worth naming:

* a module the operator repo had deliberately *not* written (``fused_moe_oot``) raised during the
  table build, and because the build imported every module unconditionally, **all thirteen ops went
  unbound** - the engine then ran its own CUDA-shaped kernels on an NPU with one warning that
  blamed the model profile;
* a third-party operator package could only be used by editing the platform (or by calling
  ``register_operator_layout`` with a blob of dotted paths, which is the same coupling with extra
  steps).

Discovery
---------
Two sources, in this order:

1. ``ASCEND_PLATFORM_OPS`` - the same variable that pins the import root.  When the pinned package
   exposes ``describe()``, it is the provider; this is how a user points the platform at their own
   package without installing any metadata.
2. the ``ascend_platform.providers`` entry-point group - the PyPA-standard way for an installed
   distribution to announce itself.

Failure policy
--------------
Everything here is **tolerant at boot and loud where it matters**: an unparsable descriptor, a
version outside this platform's range, or an op whose target cannot be imported is recorded against
that provider/op and the boot continues on vLLM's own implementations.  What is *not* tolerated is
silence: the report says which provider was found, which ops it served, and why the rest did not.

The descriptor is data (``docs/op-manifest.md`` documents the sibling manifest schema).  A provider
never imports ``vllm``: it names callables, and this platform is the only side that knows which
vLLM symbol to put them behind (``engine/abi/``).
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from .manifest import ManifestError
from .version import SUPPORTED_OPERATOR_RANGE, check_range
from .vocabulary import OpKind

logger = logging.getLogger("ascend_platform.contracts.provider")

__all__ = [
    "PROVIDER_GROUP",
    "ProviderOp",
    "ProviderDescriptor",
    "ProviderError",
    "parse_descriptor",
    "discover",
    "provider",
    "reset_provider",
]


#: Entry-point group an installed operator distribution declares.  The value is a callable
#: returning a descriptor mapping (or the mapping itself, for a static file).
PROVIDER_GROUP = "ascend_platform.providers"

#: The version range a provider must declare it supports.  Same constant as the operator layer's
#: own gate: one range, so "installed but out of range" cannot mean two different things.
SUPPORTED_PROVIDER_RANGE = SUPPORTED_OPERATOR_RANGE


class ProviderError(ManifestError):
    """A provider descriptor is malformed, or asks for something this platform cannot do."""


@dataclass(frozen=True)
class ProviderOp:
    """One op a provider serves.

    ``target`` is ``"package.module:attribute"`` - deliberately the same notation the operator repo
    already uses in its own requirement table, so the two sides can be compared mechanically.
    """

    name: str
    target: str
    dtypes: Tuple[str, ...] = ()
    graph: str = "unknown"
    note: str = ""
    #: ``compute`` (a tensor-in/tensor-out callable) or ``class_factory`` (``base_cls -> subclass``).
    #: A descriptor that omits it is assumed to be compute; :mod:`contracts.vocabulary` says which
    #: names *must* be factories, and the binding refuses the mismatch instead of trusting this.
    kind: OpKind = OpKind.COMPUTE

    def __post_init__(self) -> None:
        # Both descriptor shapes (mapping and list) land here, so the one notation check lives here
        # rather than in the parser: a bare module path would otherwise be "resolved" as a package.
        if ":" not in self.target or not all(self.target.split(":", 1)):
            raise ProviderError(
                f"provider op {self.name!r}: target {self.target!r} is not 'module:attribute'"
            )

    def resolve(self) -> Any:
        """Import and return the callable, or raise :class:`ProviderError` naming the target."""
        module_name, _, attribute = self.target.partition(":")
        if not module_name or not attribute:
            raise ProviderError(
                f"provider op {self.name!r}: target {self.target!r} is not 'module:attribute'"
            )
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - a missing module is a fact about this op
            raise ProviderError(
                f"provider op {self.name!r}: cannot import {module_name} ({type(exc).__name__}: {exc})"
            ) from None
        try:
            return getattr(module, attribute)
        except AttributeError:
            raise ProviderError(
                f"provider op {self.name!r}: {module_name} has no attribute {attribute!r}"
            ) from None


@dataclass(frozen=True)
class ProviderDescriptor:
    """What a provider package says about itself.  Pure data; no callables, no engine names."""

    name: str
    version: str
    requires_platform: str = ""
    ops: Mapping[str, ProviderOp] = field(default_factory=dict)
    #: Where the descriptor came from ("ASCEND_PLATFORM_OPS=my_ops", "entry point ..."), for reports.
    source: str = ""

    def op_names(self) -> Tuple[str, ...]:
        return tuple(self.ops)

    def summary(self) -> str:
        served = ", ".join(self.op_names()) or "no ops"
        where = f" from {self.source}" if self.source else ""
        return f"provider {self.name} {self.version}{where}: {len(self.ops)} op(s) [{served}]"


def _as_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value)
    raise ProviderError(f"expected a string or a list of strings, got {type(value).__name__}")


def _op_from(name: Any, spec: Any) -> ProviderOp:
    if isinstance(spec, str):
        return ProviderOp(name=str(name), target=spec)
    if not isinstance(spec, Mapping):
        raise ProviderError(
            f"op {name!r}: expected 'module:attribute' or a mapping, got {type(spec).__name__}"
        )
    target = spec.get("target") or spec.get("impl")
    if not isinstance(target, str) or ":" not in target:
        raise ProviderError(f"op {name!r}: target must be 'module:attribute', got {target!r}")
    raw_kind = spec.get("kind", OpKind.COMPUTE.value)
    try:
        kind = OpKind(str(raw_kind))
    except ValueError:
        raise ProviderError(
            f"op {name!r}: kind must be one of {[k.value for k in OpKind]}, got {raw_kind!r}"
        ) from None
    return ProviderOp(
        name=str(name),
        target=target,
        dtypes=_as_tuple(spec.get("dtypes")),
        graph=str(spec.get("graph", "unknown")),
        note=str(spec.get("note", "")),
        kind=kind,
    )


def parse_descriptor(data: Any, *, source: str = "") -> ProviderDescriptor:
    """Validate a descriptor mapping.  Raises :class:`ProviderError` - never a bare ``KeyError``.

    The checks are the ones whose absence produces a wrong number instead of a message: a name, a
    version, a range this platform can read, and targets that are callables-shaped strings.
    """
    if not isinstance(data, Mapping):
        raise ProviderError(f"descriptor must be a mapping, got {type(data).__name__}")
    name = data.get("name")
    version = data.get("version")
    if not isinstance(name, str) or not name:
        raise ProviderError("descriptor needs a non-empty 'name'")
    if not isinstance(version, str) or not version:
        raise ProviderError(f"provider {name!r}: descriptor needs a non-empty 'version'")
    requires = data.get("requires_platform", "")
    if requires and not isinstance(requires, str):
        raise ProviderError(f"provider {name!r}: 'requires_platform' must be a string")
    if requires:
        # A malformed *requirement* is the provider's bug; a platform outside it is a mismatch.
        check_range(version, requires, what=f"provider {name!r}")

    raw_ops = data.get("ops", {})
    if isinstance(raw_ops, Mapping):
        ops = {str(op_name): _op_from(op_name, spec) for op_name, spec in raw_ops.items()}
    elif isinstance(raw_ops, (list, tuple)):
        ops = {}
        for entry in raw_ops:
            if not isinstance(entry, Mapping) or not entry.get("name"):
                raise ProviderError(f"provider {name!r}: every op entry needs a 'name'")
            op = _op_from(entry["name"], entry)
            if op.name in ops:
                raise ProviderError(f"provider {name!r}: duplicate op {op.name!r}")
            ops[op.name] = op
    else:
        raise ProviderError(f"provider {name!r}: 'ops' must be a mapping or a list")
    return ProviderDescriptor(
        name=name, version=version, requires_platform=requires, ops=ops, source=source
    )


def _call_describe(value: Any, source: str) -> ProviderDescriptor:
    """Accept a descriptor mapping, a ``describe()`` callable, or a module carrying ``describe``."""
    if callable(value):
        data = value()
    else:
        describe = getattr(value, "describe", None)
        data = describe() if callable(describe) else value
    return parse_descriptor(data, source=source)


def _from_entry_points() -> Iterator[Tuple[str, ProviderDescriptor]]:
    try:
        from importlib.metadata import entry_points
    except Exception:  # pragma: no cover - Python < 3.8 is not supported anyway
        return
    try:
        found = entry_points(group=PROVIDER_GROUP)
    except TypeError:  # pragma: no cover - the pre-3.10 API
        found = entry_points().get(PROVIDER_GROUP, [])  # type: ignore[attr-defined]
    for entry in found:
        source = f"entry point {entry.name}"
        try:
            loaded = entry.load()
            yield entry.name, _call_describe(loaded(), source) if callable(loaded) else _call_describe(loaded, source)
        except ProviderError as exc:
            logger.warning("ascend-platform provider %s: bad descriptor (%s); ignored", entry.name, exc)
        except Exception as exc:  # noqa: BLE001 - a broken plugin must not stop the boot
            logger.warning(
                "ascend-platform provider %s: could not load (%r); ignored", entry.name, exc
            )


def _from_operator_layer() -> Optional[ProviderDescriptor]:
    """Ask the *resolved* operator layer first - whichever root won, pinned or by preference.

    Using the port's own resolution (rather than reading the environment here) is what makes a
    descriptor work on the normal path too: ``ascend_attention`` wins ``OPERATOR_ROOTS`` without
    anyone setting ``ASCEND_PLATFORM_OPS``, and its ``describe()`` must still be found.  A layer
    without ``describe()`` is not a provider, and the layout path keeps serving it.
    """
    import os

    from .ports import ENV_OPS_LAYER, operator_layer

    layer = operator_layer(required=False)
    if layer is None:
        return None
    pinned = os.environ.get(ENV_OPS_LAYER, "").strip()
    source = f"{ENV_OPS_LAYER}={layer.root}" if pinned else f"operator layer {layer.root}"
    try:
        module = importlib.import_module(layer.root)
    except Exception as exc:  # noqa: BLE001 - the port already proved it importable
        logger.debug("ascend-platform provider: %s does not import (%r)", layer.root, exc)
        return None
    describe = getattr(module, "describe", None)
    if describe is None:
        return None
    try:
        return _call_describe(describe, source)
    except ProviderError as exc:
        logger.warning("ascend-platform provider %s: bad descriptor (%s); ignored", layer.root, exc)
        return None


_CACHE: Dict[str, Any] = {"resolved": False, "provider": None}


def discover() -> Optional[ProviderDescriptor]:
    """The first provider that answers, or ``None``.  Cached; use :func:`reset_provider` in tests.

    Order: the operator layer the port resolved (``ASCEND_PLATFORM_OPS`` when pinned, otherwise the
    first importable root - an explicit instruction beats installed metadata), then entry points in
    the order the metadata reports them.
    """
    if _CACHE["resolved"]:
        return _CACHE["provider"]
    found = _from_operator_layer()
    if found is None:
        found = next((descriptor for _name, descriptor in _from_entry_points()), None)
    if found is not None:
        logger.info("ascend-platform %s", found.summary())
    _CACHE.update(resolved=True, provider=found)
    return found


class _Provider:
    """The resolved provider, with per-op caching so a miss is remembered and reported once."""

    def __init__(self, descriptor: ProviderDescriptor) -> None:
        self.descriptor = descriptor
        self._impls: Dict[str, Tuple[Any, str]] = {}

    def op(self, name: str) -> Optional[ProviderOp]:
        """The descriptor entry, so the caller can check ``kind`` before trusting the target."""
        return self.descriptor.ops.get(name)

    def load(self, name: str) -> Tuple[Any, str]:
        """``(callable, reason)`` - ``(None, why)`` when this provider does not serve ``name``."""
        if name in self._impls:
            return self._impls[name]
        op = self.descriptor.ops.get(name)
        if op is None:
            result: Tuple[Any, str] = (
                None,
                f"provider {self.descriptor.name!r} does not serve {name!r} "
                f"(it serves {len(self.descriptor.ops)} op(s))",
            )
        else:
            try:
                result = (op.resolve(), "")
            except ProviderError as exc:
                result = (None, str(exc))
        self._impls[name] = result
        return result


def provider() -> Optional[_Provider]:
    """The provider as a loader, or ``None`` when nothing announced itself."""
    descriptor = discover()
    return None if descriptor is None else _Provider(descriptor)


def reset_provider() -> None:
    """Forget the discovery result (tests, and after installing a provider in-process)."""
    _CACHE.update(resolved=False, provider=None)


def describe_provider() -> Dict[str, Any]:
    """A JSON-able description for the boot banner and ``doctor``."""
    descriptor = discover()
    if descriptor is None:
        return {"available": False, "name": None, "version": None, "ops": [], "source": None}
    return {
        "available": True,
        "name": descriptor.name,
        "version": descriptor.version,
        "ops": list(descriptor.op_names()),
        "source": descriptor.source,
    }
