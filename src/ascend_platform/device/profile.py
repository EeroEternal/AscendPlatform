"""``DeviceProfile`` — every device constant in one frozen record, plus a registry.

Why this module exists
----------------------
In the source repo the device facts were spread over ``platform.py`` (core counts, memory probes),
``worker.py`` (page sizes, snapshot spellings), ``graphs.py`` (which ``torch.cuda`` names have NPU
counterparts) and three docs.  That is fine for one card and wrong the moment there is a second one:
nothing then says which statement was a *hardware* fact and which was a *policy* choice.

This module draws that line.  A :class:`DeviceProfile` answers "what is this box", and platform code
reads the profile instead of a literal.  Model choices stay in
:mod:`ascend_platform.profiles.base`; engine mechanics stay in :mod:`ascend_platform.engine`.

What is deliberately **not** here
---------------------------------
* Model facts (head widths, GQA group of *this* checkpoint, op family) — that is ``ModelProfile``.
* Engine mechanics (worker class names, the vLLM ABC reject policy) — that is ``engine``.
* Anything that needs ``import vllm`` or ``torch_npu`` at module load: the probes are functions and
  are only called from ``engine``.

The values below are the ones the source repo measured on the reference box; each field's docstring
says how it was obtained, because a wrong guess here mis-sizes the KV pool or silently changes
numerics rather than raising.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DeviceProfile",
    "EnvRequirement",
    "DEFAULT_DEVICE_PROFILE",
    "register_device_profile",
    "unregister_device_profile",
    "known_device_profiles",
    "device_profile",
    "detect_device_profile",
]


@dataclass(frozen=True)
class EnvRequirement:
    """One environment variable that must hold a given value for the stack to behave.

    ``failure_mode`` is prose because the interesting part is *how it fails*: ``ACL_OP_INIT_MODE``
    unset produces **wrong numbers**, not an error, and that is the kind of fact a test can only
    assert if the text is written down next to the variable.
    """

    name: str
    expected: Optional[str] = None
    required: bool = True
    failure_mode: str = ""
    source: str = ""


@dataclass(frozen=True)
class DeviceProfile:
    """The hardware as data: identity, memory, page geometry, dtype/layout surface, capabilities.

    ``kernel_page_size``
        Tokens per compiled kernel page (the kernels' ``BC``).  An engine page must be a whole
        number of these, because the KV pool is re-viewed as ``[pages, kernel_page, heads, dim]``
        without a copy.
    ``engine_page_multiple``
        The engine page (``block_size``) a hybrid attention+state model must use, i.e. a whole
        number of kernel pages **and** of the state page.  2048 on the reference box, which is why
        the hybrid KV test asserts that number rather than "whatever the model profile says".
    ``state_page_bytes``
        Padded per-page size of the linear-attention (mamba/GDN) state.  vLLM compares pages across
        the attention and state groups, so this has to be padded to exactly one attention page.
    ``capabilities``
        Free-form capability bits the platform gates on (``graph_capture``, ``hybrid_kv``,
        ``mamba_state``, ``pin_memory``, ``hccl``).  A capability that is absent must be *refused at
        start-up*, never silently degraded — that is the repo's contract.
    """

    name: str
    vendor: str = "ascend"
    dispatch_key: str = "NPU"

    #: Bytes of HBM usable by this process's device (not the driver's figure).
    hbm_bytes: int = 0
    #: CUDA-style "SM count": what vLLM asks for as ``num_compute_units``.
    compute_units: int = 1
    cube_units: int = 0
    vector_units: int = 0

    #: Tokens per kernel page, and the engine page a hybrid pool must use.
    kernel_page_size: int = 128
    engine_page_multiple: int = 2048
    #: Padded bytes per linear-attention state page.
    state_page_bytes: int = 0

    #: Compute dtypes and KV-cache dtypes this device's kernels are built for.
    dtypes: Tuple[str, ...] = ("bfloat16", "float16")
    kv_cache_dtypes: Tuple[str, ...] = ("auto", "bfloat16", "float16")
    #: KV layouts the kernels can read ("ND" order; "NZ" is the device-internal copy layout).
    layouts: Tuple[str, ...] = ("ND", "NZ")

    #: Attention head sizes with a compiled kernel, and the GQA group floor (1 reads out of range).
    head_sizes: Tuple[int, ...] = (128, 256)
    min_gqa_group: int = 2

    capabilities: Tuple[str, ...] = ()
    env: Tuple[EnvRequirement, ...] = ()

    #: Substrings of ``torch.npu.get_device_name()`` that select this profile.
    name_patterns: Tuple[str, ...] = ()
    #: Where the numbers came from (a measurement, a doc, a ticket) — printed by ``doctor``.
    evidence: str = ""

    # -- validation -------------------------------------------------------------------------
    def validate(self) -> "DeviceProfile":
        """Fail loudly on an incoherent profile; returning ``self`` keeps it usable inline."""
        if self.kernel_page_size <= 0 or self.kernel_page_size & (self.kernel_page_size - 1):
            raise ValueError(
                f"{self.name}: kernel_page_size must be a positive power of two, "
                f"got {self.kernel_page_size}"
            )
        if self.engine_page_multiple % self.kernel_page_size:
            raise ValueError(
                f"{self.name}: engine_page_multiple ({self.engine_page_multiple}) must be a "
                f"multiple of kernel_page_size ({self.kernel_page_size})"
            )
        if not self.dtypes:
            raise ValueError(f"{self.name}: at least one compute dtype must be declared")
        if not self.head_sizes:
            raise ValueError(f"{self.name}: at least one head size must be compiled")
        if self.min_gqa_group < 1:
            raise ValueError(f"{self.name}: min_gqa_group must be >= 1")
        return self

    # -- queries ----------------------------------------------------------------------------
    def supports_dtype(self, dtype: Any) -> bool:
        name = getattr(dtype, "name", dtype)
        return str(name) in self.dtypes

    def supports_kv_cache_dtype(self, dtype: Any) -> bool:
        name = getattr(dtype, "name", dtype)
        return str(name) in self.kv_cache_dtypes

    def has(self, capability: str) -> bool:
        return capability in self.capabilities

    def check_engine_page(self, block_size: int) -> None:
        """Raise unless ``block_size`` is a whole number of kernel pages."""
        if block_size % self.kernel_page_size:
            raise ValueError(
                f"{self.name}: engine block_size {block_size} is not a multiple of the "
                f"{self.kernel_page_size}-token kernel page"
            )

    def as_dict(self) -> Dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        data["env"] = [e.__dict__ for e in self.env]
        return data


#: Fallback when detection finds nothing: a coherent single-card profile with no claims, so code
#: paths that only need the page geometry keep working on a CPU box.
DEFAULT_DEVICE_PROFILE = DeviceProfile(
    name="generic-ascend",
    hbm_bytes=0,
    compute_units=1,
    capabilities=(),
    evidence="fallback: no detected device; page geometry follows the 950 kernel page",
).validate()

_LOCK = threading.RLock()
_REGISTRY: Dict[str, DeviceProfile] = {}
_ACTIVE: Optional[str] = None


def register_device_profile(profile: DeviceProfile, *, replace_existing: bool = False) -> DeviceProfile:
    """Register ``profile`` by name; a duplicate is an error unless explicitly replaced."""
    profile.validate()
    with _LOCK:
        if profile.name in _REGISTRY and not replace_existing:
            raise ValueError(f"device profile {profile.name!r} is already registered")
        _REGISTRY[profile.name] = profile
    return profile


def unregister_device_profile(name: str) -> None:
    """Forget a registered profile (tests, and re-registration after an edit); a no-op if absent."""
    with _LOCK:
        _REGISTRY.pop(name, None)


def known_device_profiles() -> Tuple[str, ...]:
    """Every registered profile name, sorted — what ``doctor`` prints and what errors list."""
    with _LOCK:
        return tuple(sorted(_REGISTRY))


def device_profile(name: str) -> DeviceProfile:
    """Look a profile up by name; a miss is a ``KeyError`` that lists the known names.

    This is the *explicit* accessor: engine code that wants "the profile for this box" calls
    :func:`resolve_device_profile` instead, which probes and falls back.
    """
    with _LOCK:
        try:
            return _REGISTRY[name]
        except KeyError:
            raise KeyError(
                f"unknown device profile {name!r}; known: {', '.join(sorted(_REGISTRY)) or '(none)'}"
            ) from None


def _match(device_name: str) -> Optional[DeviceProfile]:
    lowered = (device_name or "").lower()
    with _LOCK:
        profiles = list(_REGISTRY.values())
    for profile in profiles:
        for pattern in profile.name_patterns:
            if pattern.lower() in lowered:
                return profile
    return None


def detect_device_profile(*, npu: Any = None, device_name: Optional[str] = None) -> DeviceProfile:
    """Best-effort detection: ``torch.npu.get_device_name`` → a registered profile.

    Never raises and never guesses numbers: an unknown card yields :data:`DEFAULT_DEVICE_PROFILE`
    with ``evidence`` saying so, which is exactly what a second-card bring-up should start from.
    """
    name = device_name
    if name is None:
        try:
            if npu is None:
                import torch_npu  # noqa: F401  (registers torch.npu)

                import torch

                npu = torch.npu
            name = str(npu.get_device_name(0))
        except Exception:  # noqa: BLE001 - no device is a normal state on a CPU box
            name = None
    if name:
        matched = _match(name)
        if matched is not None:
            return matched
        return replace(
            DEFAULT_DEVICE_PROFILE,
            name=name,
            name_patterns=(name,),
            evidence=f"unprofiled device {name!r}: page geometry assumed, numbers unverified",
        )
    return DEFAULT_DEVICE_PROFILE
