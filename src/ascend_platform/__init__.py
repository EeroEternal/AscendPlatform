"""``ascend_platform`` — the Ascend platform plugin for vLLM (device and execution semantics).

What this package is
--------------------
An **out-of-tree vLLM platform plugin**: it tells the engine what the device is, how much memory it
may use, how the KV pool is shaped, when a step may be captured into a graph, and *which*
implementation serves an operator name.  It does **not** implement operator math; that lives in a
separate operator layer (today ``tileinfer``, target ``ascend_attention``) and is reached only
through :mod:`ascend_platform.contracts.ports`.

Layers, and the import direction that keeps them apart
------------------------------------------------------
``device``      what the hardware *is* (``DeviceProfile``, probes).  No vLLM, no torch at import.
``contracts``   the two跨仓 interfaces: the operator manifest / ports, and version ranges.
``runtime``     engine-independent session semantics (buckets, plan cache, step buffers, phases).
``profiles``    model facts (``ModelProfile``): head widths, page policy, op family, reject list.
``engine``      the only place allowed to ``import vllm`` — platform, worker, graphs, patches,
                plugin, attention adapters.
``ops``         operator-name rebinding (mechanism) + the model family tables.
``doctor``      the bring-up self check, as a CLI.

The direction above is not a convention to remember: it is data in :mod:`ascend_platform._acl`,
enforced from ``tests/test_layer_acl.py`` (statically *and* by importing each engine-free layer in a
clean interpreter) and printed by ``ascend-platform doctor``.

Importing this package is cheap and side-effect free: no vLLM, no torch_npu, no device probe.
"""

from __future__ import annotations

from ._version import __version__

__all__ = ["__version__"]
