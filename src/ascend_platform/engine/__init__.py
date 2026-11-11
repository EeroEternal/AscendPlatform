"""The vLLM 0.23 adapter layer — the **only** package in this repo allowed to import ``vllm``.

Everything here is about *engine* semantics: which platform class vLLM instantiates, which worker
hooks it calls, how a batch is described to an attention backend, and how a decode step is captured.
Device facts come from :mod:`ascend_platform.device`, session policy from
:mod:`ascend_platform.runtime`, model facts from :mod:`ascend_platform.profiles`, operator wiring
from :mod:`ascend_platform.ops` — this package only translates.

Importing it must stay cheap and side-effect free: the modules import ``vllm`` lazily (inside
functions) so the CPU test suite and ``ascend-platform doctor`` can run without an engine.
"""

from __future__ import annotations

__all__: list[str] = []
