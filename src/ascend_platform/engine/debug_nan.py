"""Trace the first module that *introduces* NaN/Inf into the forward pass (debug only).

Why it exists: the Qwen3.6 run produced 14 good tokens and then ``!`` for ever.  A NaN probe in the
attention impl showed the NaN is already in ``q``/``k``/``v`` *entering* attention, so the attention
kernel and everything downstream of it are exonerated - but "upstream" is 40 decoder layers.  This
module hooks every decoder layer and its direct children and reports the **first module whose output
is bad while its input was fine**, which is the module that introduced it.  Text in a log file cannot
tell you that; a growing list of "everything is NaN from layer 7 on" cannot either.

Disabled unless ``TILEINFER_DEBUG_NAN=1``; every hook is a device sync, so it is a debugging tool and
never a serving path.  It is imported from the ``load_model`` patch, i.e. it runs exactly once, after
the weights exist and before the first forward.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger("ascend_platform.engine.debug_nan")

__all__ = ["enabled", "install", "install_on", "reported"]

#: How many introducing modules to report before going quiet (the first one is the answer; the rest
#: is usually "everything after it", which is noise).
_MAX_REPORTS = 3

#: ``model.layers.7`` / ``layers.7`` - the decoder-layer naming every vLLM model uses.
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)$")

_REPORTS: List[str] = []
_SEEN: Dict[str, int] = {}


def enabled() -> bool:
    """``TILEINFER_DEBUG_NAN=1``.  Same variable the attention impl reads."""
    import os

    raw = os.environ.get("TILEINFER_DEBUG_NAN", "").strip().lower()
    return raw not in ("", "0", "false", "no")


def reported() -> Tuple[str, ...]:
    """The qualified names reported so far (tests, and a post-mortem read of the process)."""
    return tuple(_REPORTS)


def _bad_tensors(value: Any, *, limit: int = 8) -> Tuple[int, int]:
    """``(nan, inf)`` counts over the tensors inside ``value`` (nested tuples/lists/dicts).

    Looks at a bounded number of tensors: a layer output can be a tuple of a dozen tensors, and a
    diagnostic that walks all of them (or that syncs once per tensor) costs more than it is worth.
    """
    nan = inf = 0
    seen = 0
    stack = [value]
    while stack and seen < limit:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
            continue
        if isinstance(item, dict):
            stack.extend(item.values())
            continue
        if not hasattr(item, "isnan"):
            continue
        seen += 1
        try:
            nan += int(item.isnan().sum().item())
            inf += int(item.isinf().sum().item())
        except Exception:  # noqa: BLE001 - integer/opaque tensors are not our business
            continue
    return nan, inf


def _describe(module: Any, nan: int, inf: int) -> str:
    return f"{type(module).__name__} (nan={nan}, inf={inf})"


def _out_param_names(kwargs: Dict[str, Any]) -> Tuple[str, ...]:
    """kwargs that are **out-parameters**, i.e. buffers the callee is expected to fill.

    vLLM's decoder layer allocates ``torch.empty_like(hidden_states)`` and passes it as
    ``output=`` to the attention/GDN module (``qwen3_next.py:417-424``); the return value is
    discarded and the buffer is read instead.  Those buffers contain *uninitialised* memory on entry,
    so treating them as an input would make every such call look "already dirty" and hide the very
    module that wrote into them.
    """
    names = []
    for name, value in kwargs.items():
        if name in ("output", "out", "outputs") and hasattr(value, "isnan"):
            names.append(name)
    return tuple(names)


def _hook(module: Any, qualname: str) -> None:
    """Report when ``module`` turns a clean input into a bad output (return value *or* out-param)."""
    state: Dict[str, Any] = {"dirty_input": False, "out": ()}

    def _pre(_module, args, kwargs):  # noqa: ANN001 - torch hook signature
        outs = _out_param_names(kwargs)
        state["out"] = outs
        # The out-param buffers are *not* evidence about the input: they arrive uninitialised.
        inputs = {k: v for k, v in kwargs.items() if k not in outs}
        nan, inf = _bad_tensors((args, inputs))
        state["dirty_input"] = bool(nan or inf)

    def _post(_module, args, kwargs, output):  # noqa: ANN001
        if len(_REPORTS) >= _MAX_REPORTS or state["dirty_input"] or qualname in _REPORTS:
            return
        nan, inf = _bad_tensors(output)
        where = "return value"
        if not (nan or inf):
            # The 0.23 decoder layer reads the out-param, not the return value: check that instead,
            # which is also the only way to catch a kernel that writes NaN into a caller's buffer.
            for name in state["out"]:
                nan, inf = _bad_tensors(kwargs[name])
                if nan or inf:
                    where = f"out-param {name}"
                    break
        if not (nan or inf):
            return
        _REPORTS.append(qualname)
        _SEEN[qualname] = _SEEN.get(qualname, 0) + 1
        logger.error(
            "ascend-platform nan-trace: %s wrote NaN/Inf into its %s: %s (input was finite)",
            qualname,
            where,
            _describe(module, nan, inf),
        )

    module.register_forward_pre_hook(_pre, with_kwargs=True)
    module.register_forward_hook(_post, with_kwargs=True)
    setattr(module, "_ascend_platform_nan_trace", True)


def _decoder_layers(model: Any) -> List[Tuple[str, Any]]:
    """The ``layers.N`` modules, in index order.  Works for any vLLM model layout."""
    found: List[Tuple[int, str, Any]] = []
    for name, module in model.named_modules():
        match = _LAYER_RE.search(name)
        if match is None:
            continue
        found.append((int(match.group(1)), name, module))
    found.sort(key=lambda row: row[0])
    return [(name, module) for _index, name, module in found]


def install_on(model: Any) -> int:
    """Hook every decoder layer and its direct children; returns how many layers were instrumented."""
    layers = _decoder_layers(model)
    for name, layer in layers:
        if getattr(layer, "_ascend_platform_nan_trace", False):
            continue
        _hook(layer, name)
        for child_name, child in layer.named_children():
            if getattr(child, "_ascend_platform_nan_trace", False):
                continue
            _hook(child, f"{name}.{child_name}")
            # Two levels, because vLLM's attention wrapper holds the backend call one level down
            # (``self_attn.attn`` is the module whose ``forward`` reaches this platform's impl); the
            # difference between "the wrapper's own projections" and "the attention backend" is the
            # difference between a model-math bug and ours.
            for grand_name, grand in child.named_children():
                if getattr(grand, "_ascend_platform_nan_trace", False):
                    continue
                _hook(grand, f"{name}.{child_name}.{grand_name}")
    return len(layers)


def install(model: Any) -> Optional[int]:
    """Install when the environment asks for it; silent no-op otherwise."""
    if model is None or not enabled():
        return None
    try:
        count = install_on(model)
    except Exception:  # noqa: BLE001 - a diagnostic must never break a run
        logger.warning("ascend-platform nan-trace: could not instrument the model", exc_info=True)
        return None
    logger.warning(
        "ascend-platform nan-trace: instrumented %d decoder layer(s); the first module that turns a "
        "finite input into NaN/Inf is reported at error level (TILEINFER_DEBUG_NAN=1)",
        count,
    )
    return count
