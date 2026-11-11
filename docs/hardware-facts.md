# Platform-layer facts: Ascend950PR + CANN 9.3.0 + torch_npu + vLLM 0.23

> **这份文件是什么**：迁移方案 [`migration-from-tileinfer.md`](migration-from-tileinfer.md) 的**素材清单**——
> 从 TileInfer 里抽出来的、平台层必须继承的硬件与契约事实，逐条带出处，用来对照"搬的时候有没有搬丢"。
> 它不是设计文档；设计在迁移方案里。
>
> 基线：TileInfer `339969a`（2026-11-10）。引用路径一律相对 TileInfer 根目录。
> 标注 `已作废` 的结论是历史上写进文档、后被实验推翻的，**不得进入新平台的文档或代码**。
> §J 是明确未核实的开放问题：它们不是事实，落地前必须上机确认。

_Source repo `/home/userlist/zxs/TileInfer` (read-only). Citations are `path:line` relative to that root. "已作废" = recorded as invalidated — keep it out of the new layer's design._

## A. 设备与显存事实 / device & memory
- **1 × Ascend950PR_9579** (npu-smi prints `Ascend950PR`); single card is the only validated shape. `architecture.md:138,246,375-377`, `upstream-issue-pto-cv-hang.md:68`, `README.md:96`.
- **HBM 131072 MB (~128 GiB)**. `status.md:81`, `notes/dsv41-facts.md:26` (same box: 628 GB RAM, 16 cores).
- Size source: `torch.npu.get_device_properties(0).total_memory`, **fallback `torch.npu.mem_get_info(0)[1]`**; if both fail the KV pool cannot be sized — never return 0/guess. `hardware-checklist.md:90,116`, `tileinfer/vllm/platform.py:1083-1114`.
- **AI-core count**: card reports `multi_processor_count=56`, `cube_core_num=28`, `vector_core_num=56` (measured torch_npu 2.10.0.post4); vLLM asks for it at runner construction and the ABC body raises. `platform.py:1121-1141`, `worker.py:167-171,221-222`.
- **Memory profiling is vLLM arithmetic over an NPU snapshot**: `NpuMemorySnapshot(total_memory, free_memory, torch_peak)` → `vllm.v1.worker.utils.request_memory`; peak = `torch.npu.memory_stats(dev)["allocated_bytes.all.peak"]` else `max_memory_allocated`. `worker.py:513-528,592-616`.
- Two `mem_get_info` spellings exist (`torch.npu.` / `torch.npu.memory.`); a wrong spelling is not an error, it silently mis-sizes the pool. `worker.py:554-590`.
- `get_current_memory_usage` **must** answer via `torch.npu.memory_allocated` (torch allocator counter, not driver HBM); leaving the ABC `raise` killed the engine inside `load_model` on device. `platform.py:1142-1161`.
- **Pinned host memory: answer `False`** — unproven under torch_npu; wrong `True` aborts, wrong `False` costs one copy. `platform.py:1166-1181`.
- The user needs membership in the group owning `/dev/davinci*` (`HwHiAiUser`). `architecture.md:174-178`.
- **KV-block zeroer is declined**: per-block bytes come from `stride(block_dim)` = half a page (`v1/worker/utils.py:149-150`); residual cost = garbage possible on a recycled attention page. `worker.py:239-248`.

## B. 布局、对齐与页几何 / layout & page geometry
- **Kernel page = 128 tokens** (`KERNEL_PAGE_SIZE=128`, the kernels' `BC`); engine block size must be an integer multiple (`MultipleOf(128)`). `tileinfer/serving/buckets.py:5,57,83`, `notes/vllm-attention-contract.md:401`.
- **Engine page = 2048** for the hybrid target model; platform aligns up (`1024→2048`, logged) or **refuses** a user-specified non-multiple. `hardware-checklist.md:293,306`, `extending.md:198-199`.
- **Mamba state page is padded to exactly one attention page: `mamba_page_size_padded=4194304` (4 MiB)**; attention page = 2(K+V)×2 kv_heads×256 head_dim×2 B = **2048 B/token**. `hardware-checklist.md:347`, `platform.py:1200-1240,1454-1560`.
- **ND→NZ copy template asserts `ROWS % 16 == 0`** and `dual_copy` splits M across 2 AIVs ⇒ **M tile must be `% 32 == 0`**, so GQA group 8 is padded to 32 (caller passes zero-padded Q/Out, slices back). `architecture.md:278-280`, `tileinfer/kernels/attention/paged_decode_ascend950.py:19-20,1124,1418`.
- **Descriptor granularity is 32 B, not 512 B**: the merge kernel faulted on a 16 B (group-4) GM→UB descriptor with `ACL_ERROR_RT_VECTOR_CORE_EXCEPTION` (507035); fix = stage at slot stride `BR` (128 B). `known-issues.md:161-170,196-199`.
- ⚠️ **No 512 B alignment requirement appears anywhere in these docs or in the kernel assertions.** Documented alignment facts = 16-row/32-row NZ rule, 32-byte descriptors, 128-token kernel pages, 2048 engine page, 4 MiB state page. Treat "512 B" as unverified for this repo.
- **KV layout is fused-KV NHD**: logical `(num_blocks, 2, block_size, kv_heads, dim)`, `get_kv_cache_stride_order() == (1,0,2,3,4)` = physical `[2][blocks][page][heads][dim]`, `block_dim == 0`. Declaring it makes vLLM **skip** its hybrid layout rewrite (`gpu_model_runner.py:7189-7191`); getting it wrong = silently reading K as V. `hardware-checklist.md:351-352,367`, `worker.py:239-247`, `extending.md:213-214`.
- GQA groups are contiguous (`q head = kv_head*group + i`); `group >= 2` is a hard floor (1 lets the kernel read out of range). `architecture.md:118-119`, `extending.md:211-212`.
- The page-index arg is a **fixed-size ABI slot**: pad `kv_indices` to the compiled pool capacity; capacity is rounded to a power of two (`declared_pool_cap`, pinnable with `TILEINFER_POOL_CAP`) and re-bound via `Tensor.set_` (`as_strided` rejects the oversized shape), so a snapshot-derived pool size does not invalidate a pre-warm. `architecture.md:199-206,281-282`.

## C. dtype 与拒绝清单 / dtype & unsupported surface
- Compute dtype: **bf16 / fp16 only** (bf16 first = `--dtype auto` target); `--kv-cache-dtype` stays `auto`/bf16/fp16 (no quantised KV). `platform.py:936-943,1183-1198`, `notes/vllm-attention-contract.md:403`.
- `head_size ∈ {128, 256}`; MLA, sparse, sliding-window, sink, mm-prefix, per-head quant scales, non-causal, batch invariance, KV connector, **speculative decode**, quantisation, TP/EP/PP/DP+HCCL, other models = rejected at start-up, not silently degraded. `platform.py:30-45,946-994,1359-1374`, `plan-vllm-native-backend.md:16-24`.
- **Mamba SSU is not extensible and not needed**: `MambaConfig.backend` is triton/flashinfer only and `ssu_dispatch._BACKEND_REGISTRY` is private, but GDN layers report `GDN_ATTN` so SSU is a no-op; a MAMBA1/2 model must be refused. `worker.py:228-260`, `hardware-checklist.md:368`.
- GDN/FLA ops have **no pure-torch fallback upstream** (Triton-only; the "fallback" gate is ROCm-only) ⇒ the platform/ops layer must supply them. `notes/vllm-api-0.23.md:61-86`.

## D. 执行模式：eager 与 ACLGraph / execution modes
- **Capture is reachable on this platform**; `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE` = capturable for pure single-token decode only, prefill/mixed falls back to the reference path. `notes/vllm-attention-contract.md:283-291,406`.
- Enabling needs a **`torch.cuda`→`torch.npu` bridge**: 4 **required** aliases (`CUDAGraph`, `graph`, `mem_get_info`, `Stream`) + 5 best-effort (`current_stream`, `stream`, `Event`, `OutOfMemoryError`, `set_stream`); all-or-nothing write with rollback, verified **by object identity**, plus `torch.npu.graph_pool_handle` (reached via `Platform.get_global_graph_pool`). `hardware-checklist.md:545-558,607`, `tileinfer/vllm/graphs.py:248-330`.
- The wrapper's 2 sites (`cuda_graph.py:283,:313`) are **not** the whole story: the runner itself calls `torch.cuda.mem_get_info` (`gpu_model_runner.py:6547,:6565`) and `torch.cuda.Stream` (`:696,:1121`, async-scheduling only). `worker.py:218-220,708-712`.
- **If the bridge is not ready the platform forces `enforce_eager=True`** and says so; `capture_calls == 0` is then a decision, not a fault. `platform.py:1743-1775`, `hardware-checklist.md:559`.
- `cudagraph_mode` resolves to `NONE` before the platform sees it on a non-CUDA stack; it must be re-applied post-resolution (`TILEINFER_ACLGRAPH=1` → `FULL_DECODE_ONLY`, capture sizes `[1,2,4,8,16,32,64,128,256]`; `TILEINFER_COMPILE=0` keeps `torch.compile` off while still capturing). `platform.py:1691-1740`.
- Capture-time contract: **no allocation, no compilation, no host sync** in the captured region; a shape whose plan was not pre-warmed surfaces as a `fallback`, never as an error. `architecture.md:23-33`, `hardware-checklist.md:610`.
- Cold compile = **70–100 s holding the GIL**; inside a server it looks exactly like an engine hang ⇒ pre-warm out of process, same `TILELANG_CACHE_DIR`. `architecture.md:185-217`, `hardware-checklist.md:115`.

## E. 同步与流语义陷阱 / sync & stream hazards
- TileLang kernels launch on **their own stream**; torch copies are ordered on torch's stream ⇒ a debug peek needs `torch.npu.synchronize()`, not just the current stream. `known-issues.md:100-104`.
- **Device wedge**: after `507035` every later op fails, and the exception surfaces asynchronously inside whatever torch op runs next — do not attribute it to that op. `known-issues.md:178-184`.
- **An out-of-range gather is not a fault on Ascend** — it returns unrelated device memory (deterministic "corruption"); bounds-check indices on the host. `known-issues.md:59-114`.
- Capture state is detected with `torch.npu.is_current_stream_capturing()`. `integration-vllm-native.md:98`.
- `EE1016` ("sync while the stream is captured") was **mis-diagnosed as a platform dead end — 已作废**. `plan-vllm-native-backend.md:74-77`.
- `torch.profiler(with_stack=True)` yields **no Python stacks** under torch_npu; V1 multiprocess hides EngineCore ops from the main process (`VLLM_ENABLE_V1_MULTIPROCESSING=0` for in-process probes, which itself changes overhead). `known-issues.md:415-434,281-318`.

## F. CANN / torch_npu 环境与工具链 / environment
- **CANN 9.3.0, side-by-side in a user prefix** (`.work/cann930/install/cann`); system 9.1.1 and the driver stay untouched; `source …/set_env.sh` per shell. `architecture.md:238-246,366-371`, `scripts/env-ascend102.sh:20-31`.
- **`ACL_OP_INIT_MODE=1` is required**: CANN silently upgrades it to `2` ("aclops disabled"); with it unset the failure mode is **wrong numbers, not an error**. `architecture.md:150-153,378-380`.
- CANN needs newer libstdc++ symbol versions on `LD_LIBRARY_PATH` (`GLIBCXX_3.4.30`, conda-forge `libstdcxx-ng 12.2.0`). `architecture.md:155-156,225`.
- `tilelang==0.1.15` **official wheel** (`tilelang.ascend`, `target="ascend"`, arch `dav-3510`); `target="pto"` is mandatory for `T.tile.fill`; the `tile-ai/tilelang-ascend` fork is a dead end on this SKU. `architecture.md:145,168-172,227,238-246`.
- tilelang 0.1.15 headers are written against **CANN 9.2+/930 API**: on a bare shell (9.1.1) every kernel compile fails (`asc_pack_to_high`, `asc_set_gm2l1_nz_para` …) ⇒ device scripts must source the repo env script. `known-issues.md:345-375`.
- Known tilelang defect: `numeric_limits.h` missing `<bit>` → `no member named 'bit_cast'`; patched by `scripts/patch-tilelang-ascend-bitcast.py`. `hardware-checklist.md:369`.
- **Compile pathology**: split-KV dim=256 hangs 17 min+ in `AscendPassPipelineBody` (`pipeline.py:83`); dim=128 plain split also hangs in test specs; guarded by `TILEINFER_SKIP_SPLIT_COMPILE_HANG=1` (default). Splits stay opt-in (`TILEINFER_ALLOW_SPLIT_KV=1`) by **policy** (pre-warm cost), not safety. `known-issues.md:355-395,220-224`.
- Env vars in force: `ASCEND_RT_VISIBLE_DEVICES`, `TILELANG_CACHE_DIR`, `ACL_OP_INIT_MODE`, `LD_LIBRARY_PATH`, `TILEINFER_VLLM_STRICT|REGISTER_BACKEND|GRAPH_BRIDGE`, `TILEINFER_ACLGRAPH`, `TILEINFER_COMPILE`, `TILEINFER_OPS` (`gdn,moe`; `dense` off), `TILEINFER_OPS_CANDIDATES`, `TILEINFER_POOL_CAP`, `TILEINFER_ASCEND_TARGET`, `TILEINFER_SIMD_SOFTMAX`, `TILEINFER_DENSE_NN`, `TILEINFER_ALLOW_SPLIT_KV`, `TILEINFER_METRICS_PATH`, `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`. `scripts/env-ascend102.sh`, `platform.py`, `plugin.py:19-40`, `worker.py:232-233`.
- `TILELANG_CACHE_DIR` must be a writable **local** dir, identical for pre-warm and serve, outside the repo. `hardware-checklist.md:34,92-97`, `scripts/env-ascend102.sh:29-35`.
- Kernel-module rules: **no `from __future__ import annotations`** (PEP 563 breaks the eager builder; shape constants must be computed inside the jit fn). `architecture.md:228,283-284`.
- ABI: passing the real page count instead of the pool-sized/zero-padded tensor raises `violates packed ABI constraint`. `known-issues.md:360-362`.
- pto-isa limit: `T.assign` on a small Accumulator tile fails at M=2/4, compiles at M=16 ⇒ keep the first GEMM's M ≥ 16. `architecture.md:229`.

## G. 算子注册与重绑定（本项目不走 aclnn）/ op rebinding
- Mechanism is **not** `torch.ops`/aclnn registration: ops are taken over by (a) rebinding the **module attribute** plus every consumer namespace listed in `aliases` (upstream uses by-value imports ⇒ one missing alias silently keeps the Triton op), and (b) `CustomOp.register`/`register_oot` semantics — an OOT platform dispatches to **`forward_oot`**, and an op that only defines `forward_native` quietly runs the pure-torch path. `notes/vllm-attention-contract.md:357-392`, `extending.md:224-248,352-366`.
- Takeover must be idempotent, read `TILEINFER_OPS` **per call**, and carry `__tileinfer_*` markers; `abi_ok=False` rows are deliberate "reported, not patched" evidence (e.g. `chunk_local_cumsum`). `extending.md:352-366`.
- Direct `torch.ops._moe_C.moe_align_block_size` is unconditional upstream — a vendor-namespace op with no NPU replacement. `notes/op-coverage-qwen36.md:130`.
- Loading `vllm_ascend_C` in a process that exits normally ⇒ CANN teardown **double-free, exit 134** (upstream); mitigated in pytest only. `known-issues.md:320-343`.
- Registries are per-process and hooks carry no config argument ⇒ model facts must come from a module-level `active_profile()`. `extending.md:39-40,171-173`.

## H. vLLM 0.23 必须实现的契约点 / engine contract
- **Entry points (exact strings)**: `vllm.platform_plugins` → `tileinfer.vllm.platform:register`; `vllm.general_plugins` → `tileinfer.vllm.plugin:install`. `register()` returns a **class-path string or `None`** (None when no visible NPU → vLLM falls back to CUDA/CPU); it is invoked **twice** (must be cheap, side-effect free); two active OOT platform plugins = hard error; the platform class is instantiated **with no args**. `hardware-checklist.md:127-137,304-305`, `platform.py:1-20`.
- **`general_plugins` install order is contractual**: patch registry → `AttentionBackendEnum.CUSTOM` registration → graph bridge → `ops.install.install_ops` → model registration; one INFO banner `TileInfer vLLM plugin` per process. `plugin.py:19-40`, `hardware-checklist.md:60-62`.
- **`Platform` ABC is ~70–78 methods**: 10 implemented / 6 must-override / 3 unverified / 56 default-fine; the inventory is re-derived from the live base class so a point release fails loudly. `platform.py:22-45`, `notes/vllm-api-0.23.md:12-35`.
- **`check_and_update_config` order is fixed**: resolve profile → `set_active_profile` → reject features → `validate_block_size` → align mamba pages; it writes `cache_config.block_size` and `parallel_config.worker_cls` (only when the sentinel `"auto"`, `config/parallel.py:248`; consumed at `worker_base.py:249-253`). `extending.md:198`, `platform.py:995-1068`, `hardware-checklist.md:308`.
- **`get_attn_backend_cls(selected_backend, attn_selector_config, num_heads=None) -> str`** returns a class path (never a class) — the `CUSTOM` registry entry is only needed for the by-name `--attention-backend custom` path (otherwise upstream raises "must be registered before use"); verify with `AttentionBackendEnum.CUSTOM.get_class()` identity. `notes/vllm-attention-contract.md:305-307,327-351`, `plugin.py:42-59`.
- **`AttentionBackend`** = 4 abstract methods (`get_name`, `get_impl_cls`, `get_builder_cls`, `get_kv_cache_shape`) + `get_supported_kernel_block_sizes`, `get_kv_cache_block_dim`, `get_kv_cache_stride_order`, `get_supported_head_sizes`; class attrs `supported_dtypes`, `supported_kv_cache_dtypes`, `forward_includes_kv_cache_update=True`. 注意：`supports_kv_cache_connector` 在 vLLM 0.23.0 的 `v1/attention/backend.py` 里**不存在**（已在安装包里全文检索确认），只有更新的版本才有——不要按记忆加这个属性。`notes/vllm-attention-contract.md:39-116`.
- **`AttentionImpl.forward(layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None)`**; `output` is required (not Optional); must set `self.kv_cache_dtype`; must **not** call `super().__init__()` (its body raises); reject `output_scale`/`output_block_scale` explicitly. `notes/vllm-attention-contract.md:129-200`.
- **Metadata builder**: `__init__(kv_cache_spec, layer_names, vllm_config, device)`, `build(common_prefix_len, common_attn_metadata, fast_build=False)`, `build_for_cudagraph_capture`, `_cudagraph_support`, `reorder_batch_threshold=None`, `supports_update_block_table=False`. `CommonAttentionMetadata.seq_lens` **already includes this step's tokens**; decode ≡ `max_query_len > 1`; `PAD_SLOT_ID=-1`, `NULL_BLOCK_ID=0`. `notes/vllm-attention-contract.md:265-294,204-261`.
- Unstated upstream (must be pinned locally): `slot_mapping` dtype (needs **int64**, no in-graph cast), `block_table_tensor` dtype (**int32**), q/k/v/output layout (`[tokens, heads*dim]` vs `[tokens, heads, dim]` — accept both), and the call sites of `validate_configuration` / `build_for_cudagraph_capture`. `notes/vllm-attention-contract.md:415-436`.
- **Worker hooks (four, fixed names)**: `init_device` → `determine_available_memory` → `initialize_from_config` → `compile_or_warm_up_model`; implemented by subclassing `vllm.v1.worker.gpu_worker.Worker` **and** `gpu_model_runner.GPUModelRunner`; the V2 model runner must be refused. `hardware-checklist.md:11-21,360`, `worker.py:152-161`, `status.md:22`.
- **KV spec / hybrid**: `support_hybrid_kv_cache()` must return `True` (the ABC default `False` makes vLLM disable the hybrid manager, then reject an attention+mamba mix); page geometry is the platform's via `update_block_size_for_backend` / `_align_hybrid_block_size` / `unify_kv_cache_spec_page_size`; `register_custom_kv_cache_specs` stays `pass` (both spec classes are vLLM's); the 10+30 layer split is the model's and is never enumerated by the platform. `platform.py:48-56,1200-1283`, `notes/vllm-api-0.23.md:28-35`.
- **Compile / warm-up**: `compile_or_warm_up_model` is the pre-`capture_model` hook (pre-warm decode plans for the config-derived bucket set — `attempts/layers/buckets/plans/failed` — and the place to run `_disable_cuda_only_compilation_passes`, set `get_pass_manager_cls`/`get_compile_backend`/`import_kernels`, and re-apply `cudagraph_mode`). `worker.py:177-203,708-712`, `platform.py:1635-1740`, `plan-vllm-native-backend.md:120`.
- Also required for start-up: `get_device_name`, `get_device_total_memory`, `set_device`, `num_compute_units`, `get_current_memory_usage`, `manual_seed_all`, `is_pin_memory_available`, `check_if_supports_dtype`, `empty_cache`, `synchronize`, `get_device_communicator_cls` (HCCL; `dist_backend` is `''` and **unverified**). `platform.py:1069-1324`, `worker.py:212-214`.
- `get_global_graph_pool` → `graph_pool_handle()` is the graph-pool seam the ABC reaches through `__getattr__`. `graphs.py:41-44`, `hardware-checklist.md:607`.

## I. 不能再踩的坑（含已作废结论）/ gotchas
- **已作废: "graph capture is unreachable on 950."** v1 ran it in-service (6 plans READY, `capture_calls=40`, `fallbacks=0`, logprobs bitwise equal). `status.md:57`, `plan-vllm-native-backend.md:74-77`.
- **已作废: `EE1016` as a platform dead end.** `plan-vllm-native-backend.md:76-77`.
- **已作废: "the PTO cube→vector hand-off cannot work on this SKU" as a blocker** — true only for the *fork*; the official wheel runs `T.dual_copy`+`T.SimtVF` on the same card. The hang also fails identically on CANN 9.1.0/9.1.1/9.2.0-beta.2 ⇒ **a CANN version change is not the fix**. `upstream-issue-pto-cv-hang.md:5-10`, `architecture.md:340-356`.
- **已作废: "kernel memory corruption"** — it was a bad gather index; Ascend doesn't fault on OOB gathers. `known-issues.md:59-114`.
- **Retracted: KI-2's first "resolved"** — the real defect was a registry race across threads, not an import failure. `known-issues.md:121-159`.
- **Retracted: the `block_q // 2` half-M-tile workaround** (gone; the full `block_q` tile is used). `known-issues.md:258-267`.
- **Wrong:** `accept_output_buffer` is **not** a vLLM 0.23 API (keep only as a forward-compatible hook); the port-plan table that listed it is wrong. `notes/vllm-attention-contract.md:22-35`.
- **Retracted: "GDN has a pure-torch fallback"** (the gate is ROCm-only). `notes/vllm-api-0.23.md:61-81`.
- Stale: handover's "worker = 骨架" line; the `222 passed` test count (now `263`); the old "KV pool must be contiguous" failure (closed by declaring block-first stride order). `hardware-checklist.md:196-199,367,704-705`, `status.md:95-110`.
- **Never quote performance from README/old docs**; cross-window drift reaches 2×, only paired same-window ratios count. `README.md:23-24`, `status.md:47-57`, `plan-vllm-native-backend.md:151-155`.
- The native backend's probe JSON (`TILEINFER_METRICS_PATH`) is **not wired**: a missing file is a TODO, not "capture didn't run". `hardware-checklist.md:596-600`.
- Fully removing `vllm_ascend` means owning the ops + worker surfaces (C4+C2 exceed the whole codebase) — decide scope before adopting these facts. `integration-vllm-native.md:259-274`.

## J. 明确未核实（是问题，不是事实）/ unverified
- Whether `general_plugins` loads **before** `check_and_update_config` (unrecorded in the local vLLM subset; benign — the platform then forces eager). `plugin.py:95-104`.
- `dist_backend`/HCCL on a single card; whether `torch.npu.set_device(torch.device)` accepts a device object; whether `torch.npu.memory_stats` populates `allocated_bytes.all.peak`; whether the AI-core attribute exists across torch_npu versions; whether `pin_memory()` works. `worker.py:208-249`, `platform.py:1166-1181`.
- Whether `validate_configuration` / `build_for_cudagraph_capture` are actually called at start-up (inferred). `notes/vllm-attention-contract.md:429-433`.
- Whether the CUDA→NPU bridge alone is sufficient for capture (runner-side call sites unverified beyond `:6547/:6565`). `worker.py:218-220`.
- Version drift inside the docs: `notes/dsv41-facts.md:26` records torch 2.10.0 / CANN "25.7.RC1" for the same box that elsewhere is 2.13 / 9.3.0 — treat numbers as per-snapshot.
