# vLLM 0.23 source excerpts (citation anchors)

This directory holds **verbatim copies** of the vLLM 0.23.0 modules that this repository cites by
`file:line` — nothing here is compiled, imported, or shipped in the wheel.

## Why it exists

The native backend is written against vLLM 0.23's *official extension points*, not against
`vllm_ascend` internals. Every such claim in this repo ends with a citation like
`reference/vllm-0.23/vllm__v1__attention__selector.py:21-34`, and a citation is only worth
anything if the reader can open the file and check the line.

A citation is checked, not trusted: `tests/test_citation_anchors.py` extracts every `path.py:line`
in `src/` and verifies that this directory (or the source repo / the installed torch) contains that
line and that the line is not blank. Ambiguous short forms are rejected — `registry.py:99` matches
two vLLM modules, so it has to be written `v1/attention/backends/registry.py:99`.

## Naming

The path is flattened with `__` because these are single files lifted out of a package tree:

| here | upstream |
|---|---|
| `vllm__v1__attention__selector.py` | `vllm/v1/attention/selector.py` |
| `vllm__model_executor__layers__mamba__gdn__qwen_gdn_linear_attn.py` | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` |
| `fla_fused_sigmoid_gating.py` | `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py` (short alias, kept for historical `upstream_ref=` values) |

## Contents

Only the modules that are actually cited — **112 files**:

- **platform / worker contract** — `platforms__interface`, `config__parallel`,
  `v1__worker__worker_base`, `v1__worker__gpu_worker`
- **attention** — `v1__attention__selector`, `v1__attention__backend`,
  `v1__attention__backends__registry`
- **model runner** — `v1__worker__gpu_model_runner`
- **graph capture** — `compilation__cuda_graph`
- **GDN / FLA** — `mamba__gdn__qwen_gdn_linear_attn`, `mamba__ops__causal_conv1d`,
  `fla__ops__fused_gdn_prefill_post_conv`, `fla__ops__chunk`, `fla_fused_recurrent`,
  `fla_fused_sigmoid_gating`, `fla_solve_tril`, `fla_cumsum`, `fla_l2norm`
- **utility paths that the platform makes claims about** — `utils__platform_utils`,
  `utils__system_utils`, `utils__mem_utils`, `v1__worker__utils`,
  `v1__attention__backends__utils`, `v1__core__kv_cache_utils`,
  `compilation__passes__pass_manager`, `v1__cudagraph_dispatcher`,
  `model_executor__models__registry`, `model_executor__models__config`

## Provenance and licence

Upstream: <https://github.com/vllm-project/vllm>, tag `v0.23.0`. vLLM is Apache-2.0; the same
licence covers this repository (see `../../LICENSE`), and the files are unmodified.

## Rules

1. **Do not edit these files.** They are line-number anchors; an edit silently invalidates every
   citation that points into them.
2. When upgrading vLLM, re-copy the cited modules from the new tag and re-check the line numbers
   the code cites. The upgrade checklist is in `../../docs/notes/vllm-api-0.23.md`. The citation
   test names every `file:line` that stopped resolving, so the re-check is mechanical.
3. Dependencies that are **not** vendored here (torch, torch_npu) are cited as `path.py::symbol`,
   never `path.py:line`: their line numbers change with every wheel, and a symbol anchor survives
   that. `tests/test_citation_anchors.py` enforces both forms.
3. If a citation cannot be re-verified, delete the claim rather than leaving a confident-sounding
   line that no longer matches the source.
