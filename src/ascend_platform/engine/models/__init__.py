"""Per-request bookkeeping for the model layer — batch semantics, not model math.

The three modules here are the pieces a vLLM model needs to serve more than one sequence at a time,
and nothing else:

* :mod:`~ascend_platform.engine.models.batching` — split vLLM's flat token batch into per-request
  segments (pure boundary arithmetic, testable with no device);
* :mod:`~ascend_platform.engine.models.state_slots` — a bounded ``req_id -> state`` table, so a
  request keeps one state object and a burst of traffic cannot exhaust host memory;
* :mod:`~ascend_platform.engine.models.req_ids` — read "which requests are in this step" off the
  runner, because the model's ``forward`` only receives ``attn_metadata``.

The model itself (``DeepseekV41ForCausalLM`` and friends) lives in the model repository; this
package owns no ``nn.Module`` and no kernel.  ``MIGRATION_LOG.md`` §2 records the line-level
provenance of each module (they came from ``tileinfer/vllm/models/dsv41_*.py``).
"""

__all__ = ["batching", "req_ids", "state_slots"]
