"""Which package supplies out-of-tree architectures and their ``transformers`` config shims.

The platform does **not** own model implementations (repo README: it is a platform plugin, not a
model zoo).  What it does own is the *hook*: vLLM's ``general_plugins`` entry point runs in every
process, so it is the natural place to let a model package register itself before a checkpoint is
resolved.

Two facts decide the shape of this module:

* registration is **by class path string**, so the model module (and its parameter tree) is only
  imported when a checkpoint with that architecture is actually loaded;
* a general plugin must never raise — a missing model package is a logged skip, not a boot failure.

During the migration the default provider is the in-tree model package of the source repo
(``tileinfer``).  Once the model repo declares its own provider, set ``ASCEND_PLATFORM_MODELS`` and
this default disappears.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Sequence, Tuple

__all__ = ["MODEL_PROVIDERS_ENV", "DEFAULT_MODEL_PROVIDERS", "model_providers", "register_configs"]

#: ``module:callable[,module:callable]`` — each callable registers its architectures with vLLM and
#: returns ``{"registered": [...]}``.
MODEL_PROVIDERS_ENV = "ASCEND_PLATFORM_MODELS"

#: Migration-era default: the model package that used to live inside the platform repo.
DEFAULT_MODEL_PROVIDERS: Tuple[Tuple[str, str], ...] = (
    ("tileinfer.vllm.models.dsv41", "register_models"),
)


def _parse(raw: str) -> Tuple[Tuple[str, str], ...]:
    providers: List[Tuple[str, str]] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        module, _, entry = item.partition(":")
        providers.append((module.strip(), (entry or "register_models").strip()))
    return tuple(providers)


def model_providers() -> Tuple[Tuple[str, str], ...]:
    """The declared providers, filtered to the ones whose module is importable."""
    raw = os.environ.get(MODEL_PROVIDERS_ENV, "").strip()
    declared = _parse(raw) if raw else DEFAULT_MODEL_PROVIDERS
    available: List[Tuple[str, str]] = []
    for module_name, entry in declared:
        try:
            importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - "not installed" is the normal case
            continue
        available.append((module_name, entry))
    return tuple(available)


def register_configs() -> str:
    """Run each provider package's ``register_config()``, if it has one.

    vLLM reads ``config.json`` through ``transformers.AutoConfig`` before it looks at
    ``architectures``, so a checkpoint whose ``model_type`` is unknown to released transformers needs
    the name taught first.  Best effort by contract: returns a short status, never raises.
    """
    statuses: List[str] = []
    for module_name, _entry in model_providers():
        parts = module_name.split(".")
        root = ".".join(parts[:-2]) if len(parts) > 2 else module_name
        shim = f"{root}.config"
        try:
            module = importlib.import_module(shim)
            register = getattr(module, "register_config", None)
            if register is None:
                continue
            result: Dict[str, Any] = register()
            statuses.append(f"{result.get('config', 'ok')}")
        except Exception as exc:  # noqa: BLE001 - a general plugin must not raise
            statuses.append(f"skipped ({type(exc).__name__})")
    return ", ".join(statuses) if statuses else "no provider"
