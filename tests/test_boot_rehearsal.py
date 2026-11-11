"""Boot rehearsal: the whole Qwen3.6 start-up walked end to end, **without vLLM and without an NPU**.

What this file is for
---------------------
Every other contract test checks one seam.  This one walks the sequence the engine actually
performs and asserts the observable outcome of each step, in order:

========================================  ======================================================
step                                      asserted outcome
========================================  ======================================================
``plugin.install()``                      returns True, reports all four install steps
platform selected                         ``platform.register()`` is the class path vLLM resolves
``check_and_update_config``               accepts the Qwen3.6 shape: block_size aligned to 2048,
                                          ``worker_cls`` set, eager forced while capture is off
``worker.init_device``                    device selected, ``TileInferModelRunner`` constructed
                                          (the ``num_compute_units`` seam closed), snapshot taken
``determine_available_memory``            the accounting runs and returns the computed bytes
``initialize_from_config``                the KV-spec path is reached with 40 groups
                                          (10 attention + 30 mamba), pools bound to the layers
``compile_or_warm_up_model``              the decode plans are pre-warmed **before** capture,
                                          capture is skipped with a counted reason, sampler last
attention builder ``build()``             a ``TileInferMetadata`` for a fake
                                          ``CommonAttentionMetadata``
impl ``forward()``                        KV written to the pool, ragged metadata rebuilt, the
                                          plan cache hit, output copied back
========================================  ======================================================

A missing step cannot pass silently: every step appends to ``reached`` and the test asserts the
full list at the end, and each step's counters/log line is checked where it happens.  The three
"no silent swallow" refusals (foreign metadata, cascade prefix, an unavailable kernel backend) are
asserted too, because those are the paths that would otherwise degrade into plausible garbage.

Why the fakes look like this
----------------------------
* ``tests/fake_vllm.py`` provides the vLLM *names* (shared with the other contract tests).
* ``tests/test_platform_contract.py`` owns the ``vllm.platforms.Platform`` fake; this file
  **imports that module** instead of installing a competing base class.  A second ``Platform``
  object would break that file's own assertions whenever this file runs first (alphabetically it
  does), because ``TileInferPlatform`` would be bound to the other base.
* The model runner is the one piece replaced *after* ``init_device``: the runner is the hardware
  seam (it owns the KV tensors and the layer modules), so the test installs a recording runner that
  binds pools exactly where ``GPUModelRunner.initialize_kv_cache_tensors`` does
  (``gpu_model_runner.py:7248-7253``).  The real ``TileInferModelRunner`` is still constructed by
  ``init_device`` and asserted, so the production seam is exercised.
* The decode *kernel* backend is replaced through the impl's own availability probe
  (``attn/impl.py:_decode_attn_or_none``).  Everything above it - the plan key, the plan cache, the
  bucket state, the ragged rebuild, the output staging - is the real code.

Also pinned here (cheap, and they are the same "does the launcher agree with the engine" question):
the ``scripts/serve_qwen36.py`` plan arithmetic against ``serving/buckets.py``/``planner.py``, its
refusals, ``--plan-only``, and the launcher's plan against the worker's own derivation.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import pathlib
import re
import shutil
import subprocess
import sys
import types
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch

import fake_vllm

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_MODEL = ROOT / "runtime" / "test-model-qwen36"

#: The Qwen3.6 shape as ``config.json`` spells it (40 layers = 30 GDN + 10 full attention).  The
#: on-disk fixture under ``runtime/`` is what ``--plan-only`` is run against by hand; this dict is
#: the canonical content, recreated when ``runtime/`` was cleaned (that directory is gitignored).
FIXTURE_CONFIG: Dict[str, Any] = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 2048,
    "intermediate_size": 512,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "max_position_embeddings": 32768,
    "layer_types": [
        ("linear_attention", "linear_attention", "linear_attention", "full_attention")[
            index % 4
        ]
        for index in range(40)
    ],
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "shared_expert_intermediate_size": 512,
    "torch_dtype": "bfloat16",
}

#: ``ascend_platform.engine.*`` cannot be imported until a ``vllm`` exists; this is the same shared fake
#: the other contract tests install (cooperating: it only fills in names).
VLLM_MODE = fake_vllm.install()
#: The contract tests below pin the *shape* of vLLM 0.23's API against ``tests/fake_vllm.py``.
#: When a real ``vllm`` is importable, :func:`fake_vllm.install` deliberately refuses to inject
#: anything (mode ``"real"``) - so the recorder names this file reads straight out of
#: ``sys.modules`` (``kv_transfer_calls``, ``request_memory_calls``, ...) do not exist, and the
#: failure surfaces as a collection error rather than a test failure.  Skip instead: the CPU-only
#: CI venv (no vLLM, no NPU) is where this file is meant to run.
if VLLM_MODE == "real":
    pytest.skip(
        "contract test pins vLLM 0.23's API shape with tests/fake_vllm.py; a real vLLM is "
        "installed so the fake supplies nothing. Run it in a venv without vLLM (the CPU CI venv).",
        allow_module_level=True,
    )


# The platform-side names (``vllm.platforms.Platform``/``PlatformEnum``, ``vllm.config``, ...) and
# the ``ascend_platform.engine.platform`` import belong to tests/test_platform_contract.py.  Importing it
# here - rather than installing a second base class - is what keeps this file order-independent
# with that one.
import test_platform_contract as platform_contract  # noqa: E402,F401

from ascend_platform.runtime import buckets as buckets_mod  # noqa: E402
from ascend_platform.runtime import planner as planner_mod  # noqa: E402
from ascend_platform.engine import platform as platform_mod  # noqa: E402
from ascend_platform.engine import plugin as plugin_mod  # noqa: E402
from ascend_platform.engine import worker as worker_mod  # noqa: E402
from ascend_platform.engine.attention import impl as impl_mod  # noqa: E402
from ascend_platform.engine.attention.backend import TileInferAttentionBackend  # noqa: E402
from ascend_platform.engine.attention.metadata import TileInferMetadata  # noqa: E402
from ascend_platform.engine.attention.metadata import TileInferMetadataBuilder  # noqa: E402
from ascend_platform.engine.graphs import capture_path_ready  # noqa: E402
from ascend_platform.ops import install as install_mod  # noqa: E402
# The implementations live in the operator layer; the platform only names them, so the rehearsal
# resolves them through the same port the platform uses (contracts/ports.py).  Resolution is lazy:
# a bare venv runs everything except the two binding-identity assertions, which skip with a reason.
from ascend_platform.contracts.ports import operator_available, ops_module  # noqa: E402


def _ops_or_skip(name: str) -> Any:
    if not operator_available():
        pytest.skip(f"no operator layer installed; {name} cannot be resolved")
    return ops_module(name)
from ascend_platform.engine.platform import TileInferPlatform, WORKER_CLASS_PATH  # noqa: E402

CommonAttentionMetadata = sys.modules["vllm.v1.attention.backend"].CommonAttentionMetadata
KV_TRANSFER_CALLS = sys.modules["vllm.distributed.kv_transfer"].kv_transfer_calls

#: The target model's *servable* shape, shrunk so ten layers of pools fit in a CPU test's memory
#: (the real Qwen3.6 has 16 q / 2 kv heads at head_dim 256; the contract is identical at these
#: numbers, and `check_head_layout` enforces the same rules).
NUM_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 128
BLOCK_SIZE = 2048
MAX_MODEL_LEN = 2048
MAX_NUM_SEQS = 4
MAX_NUM_BATCHED_TOKENS = 8
NUM_BLOCKS = 5  # 5 engine pages x 16 sub-pages = 80 kernel pages -> declared cap 128
DTYPE = torch.bfloat16
DEVICE = torch.device("cpu")

#: Fixed so the capture-skip assertion is hardware-independent (on a device box the real verdict
#: depends on the torch_npu build, which this rehearsal does not have).
CAPTURE_SKIP_REASON = "graph capture is not ready in the boot rehearsal (no Ascend device)"

#: The steps every run must reach, in order (see the table in the module docstring).
STEPS: Tuple[str, ...] = (
    "plugin.install",
    "platform.selected",
    "platform.config",
    "worker.init_device",
    "worker.determine_available_memory",
    "worker.initialize_from_config",
    "worker.compile_or_warm_up_model",
    "attention.builder.build",
    "attention.impl.forward",
)


# =============================================================================================
# Fake engine pieces: the vLLM types the worker/test reads, kept as small as the step needs
# =============================================================================================


class _Spec:
    """A stand-in for a real ``KVCacheSpec``: the fields this path reads.

    ``mamba_type`` is ``MambaSpec``'s discriminator (``v1/kv_cache_interface.py:608-613``) and is
    what the mamba state-update gate reads; the rehearsal's 30 GDN layers answer ``GDN_ATTN``
    exactly like ``vllm/model_executor/layers/mamba/gdn/base.py:49-51``.
    """

    def __init__(
        self,
        kind: str,
        *,
        mamba_type: Optional[str] = None,
        block_size: int = BLOCK_SIZE,
        num_kv_heads: int = NUM_KV_HEADS,
        head_size: int = HEAD_DIM,
    ) -> None:
        self.kind = kind
        self.mamba_type = mamba_type
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size


class _Group:
    def __init__(self, spec: _Spec, layer_names: Sequence[str]) -> None:
        self.kv_cache_spec = spec
        self.layer_names = tuple(layer_names)


class _KVCacheConfig:
    """``vllm.v1.kv_cache_interface.KVCacheConfig``, reduced to the fields the hooks read."""

    def __init__(self, num_blocks: int, groups: Sequence[_Group], needs_zeroing: bool = False):
        self.num_blocks = num_blocks
        self.kv_cache_groups = tuple(groups)
        self.needs_kv_cache_zeroing = needs_zeroing


class _AttentionLayer:
    """One attention module: the worker discovers it through ``impl``, the runner binds ``kv_cache``."""

    def __init__(self, impl: Any) -> None:
        self.impl = impl
        self.kv_cache: Optional[torch.Tensor] = None


class _CompilationMode:
    NONE = 0
    VLLM_COMPILE = 1


class _CUDAGraphMode:
    NONE = 0
    PIECEWISE = 1
    FULL = 2


class _CompilationTimes:
    """``vllm.v1.worker.worker_base.CompilationTimes`` (gpu_worker.py:747-750)."""

    def __init__(self, language_model: Any = None, encoder: Any = None) -> None:
        self.language_model = language_model
        self.encoder = encoder


class _RecordingRunner:
    """The model runner as the worker sees it: records every call, binds pools like vLLM does."""

    def __init__(self, *, events: Optional[List[Any]] = None) -> None:
        self.calls: List[Any] = []
        self.events: List[Any] = events if events is not None else []
        self.model_memory_usage = 1000
        self.is_pooling_model = False
        self.lora_config = object()
        self.device = DEVICE
        self.vllm_config: Any = None
        self.kv_caches: List[torch.Tensor] = []
        self.compilation_config = types.SimpleNamespace(static_forward_context={})

    # -- memory / warm-up ----------------------------------------------------------------
    def profile_run(self) -> None:
        self.calls.append(("profile_run",))
        self.events.append(("profile_run",))

    def _dummy_run(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(("_dummy_run", args, kwargs))
        self.events.append(("dummy_run", kwargs.get("num_tokens")))
        return (object(), object())

    def _dummy_sampler_run(self, hidden_states: Any = None) -> None:
        self.calls.append(("_dummy_sampler_run",))
        self.events.append(("sampler",))

    def _dummy_pooler_run(self, hidden_states: Any = None) -> None:
        self.calls.append(("_dummy_pooler_run",))

    def maybe_remove_all_loras(self, lora_config: Any) -> None:
        self.calls.append(("maybe_remove_all_loras", lora_config))

    def capture_model(self) -> int:
        self.calls.append(("capture_model",))
        return 4096

    def profile_cudagraph_memory(self) -> int:
        self.calls.append(("profile_cudagraph_memory",))
        return 0

    # -- KV path -------------------------------------------------------------------------
    def initialize_kv_cache(self, kv_cache_config: Any, is_profiling: bool = False) -> None:
        self.calls.append(("initialize_kv_cache", kv_cache_config))
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if getattr(spec, "kind", "") != "attention":
                continue  # mamba state is the model's own; no TileInfer pool
            for name in group.layer_names:
                # The rehearsal's stand-in for the runner's allocation is the legacy K/V-first
                # fused form, which the worker's pre-warm gate still accepts (a vLLM point release
                # could hand it back).  The hybrid, block-first physical layout the backend
                # actually declares is covered by
                # test_the_prewarm_gate_recognises_a_block_first_pool and end-to-end by
                # test_a_hybrid_block_major_pool_serves_a_decode_step below.
                pool = torch.empty(
                    (2, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=DTYPE
                )
                self.kv_caches.append(pool)
                module = self.compilation_config.static_forward_context.get(name)
                if module is not None:
                    module.kv_cache = pool
                self.calls.append(("kv_pool", name))

    def _init_kv_zero_meta(self) -> None:
        self.calls.append(("_init_kv_zero_meta",))

    def init_routed_experts_capturer(self) -> None:  # pragma: no cover - flag is off
        self.calls.append(("init_routed_experts_capturer",))

    # -- inspection ----------------------------------------------------------------------
    def called(self, name: str) -> bool:
        return any(isinstance(call, tuple) and call[:1] == (name,) for call in self.calls)

    def count(self, name: str) -> int:
        return sum(1 for call in self.calls if isinstance(call, tuple) and call[:1] == (name,))


class _FakeBatchAttention:
    """The decode kernel backend: records plan/run and fills the output with a sentinel.

    Only the *kernel* is fake.  ``impl`` builds the plan key and the cache entry itself, so the
    rehearsal proves the worker drives ``attn/impl.py``'s own path rather than a parallel one.
    """

    name = "fake-tileinfer-decode"

    def __init__(self, events: List[Any]) -> None:
        self.events = events
        self.plan_kwargs: List[Dict[str, Any]] = []
        self.runs: List[Dict[str, Any]] = []

    def plan(self, **kwargs: Any) -> Any:
        bucket = int(kwargs["kv_indptr"].numel()) - 1
        self.events.append(("plan", bucket))
        self.plan_kwargs.append(kwargs)
        return types.SimpleNamespace(bucket=bucket, kwargs=kwargs, backend=self.name)

    def run(self, q: Any, k_cache: Any, v_cache: Any, out: Any = None, plan: Any = None, **kw: Any):
        self.runs.append({"q": q, "out": out, "plan": plan, **kw})
        if out is not None:
            out.fill_(7.0)
        return out


# =============================================================================================
# Small builders
# =============================================================================================


def _fake_npu(**overrides: Any) -> types.SimpleNamespace:
    """``torch.npu`` as the worker uses it, with constant memory numbers (deterministic math)."""
    api: Dict[str, Any] = {
        "device_count": lambda: 1,
        "set_device": lambda device: None,
        "empty_cache": lambda: None,
        "mem_get_info": lambda device=None: (100_000_000, 200_000_000),
        "memory_stats": lambda device=None: {"allocated_bytes.all.peak": 1_000},
        "max_memory_allocated": lambda device=None: 1_000,
        "get_device_properties": lambda index=0: types.SimpleNamespace(num_aicore=40),
    }
    api.update(overrides)
    return types.SimpleNamespace(**api)


def _fake_module(monkeypatch: pytest.MonkeyPatch, name: str, attrs: Dict[str, Any]) -> Any:
    """Create/augment ``name`` (and its parent packages) for one test, restoring afterwards."""
    parts = name.split(".")
    for depth in range(1, len(parts) + 1):
        full = ".".join(parts[:depth])
        module = sys.modules.get(full)
        if module is None:
            module = types.ModuleType(full)
            monkeypatch.setitem(sys.modules, full, module)
            if depth > 1:
                parent = sys.modules[".".join(parts[: depth - 1])]
                monkeypatch.setattr(parent, parts[depth - 1], module, raising=False)
        if not hasattr(module, "__path__"):
            monkeypatch.setattr(module, "__path__", [], raising=False)
    module = sys.modules[name]
    for key, value in attrs.items():
        monkeypatch.setattr(module, key, value, raising=False)
    return module


def _new_impl() -> Any:
    return impl_mod.TileInferAttentionImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_DIM,
        scale=float(HEAD_DIM) ** -0.5,
        num_kv_heads=NUM_KV_HEADS,
        kv_cache_dtype="auto",
    )


def _hybrid_kv_config() -> Tuple[_KVCacheConfig, Tuple[str, ...], Tuple[str, ...]]:
    """40 groups: 10 attention + 30 mamba, with unique layer names and fake specs."""
    attention_names: List[str] = []
    mamba_names: List[str] = []
    groups: List[_Group] = []
    for index in range(10):
        name = f"model.layers.{index * 4 + 3}.self_attn"
        attention_names.append(name)
        groups.append(_Group(_Spec("attention"), [name]))
    for index in range(30):
        name = f"model.layers.{index}.linear_attn"
        mamba_names.append(name)
        groups.append(_Group(_Spec("mamba", mamba_type="GDN_ATTN"), [name]))
    kv_config = _KVCacheConfig(NUM_BLOCKS, groups, needs_zeroing=True)
    return kv_config, tuple(attention_names), tuple(mamba_names)


def _install_attention_layers(
    runner: _RecordingRunner, attention_names: Sequence[str], mamba_names: Sequence[str]
) -> Dict[str, Any]:
    """Populate ``static_forward_context`` the way the real runner does (attention + mamba)."""
    layers: Dict[str, Any] = {}
    for name in attention_names:
        layers[name] = _AttentionLayer(_new_impl())
    for name in mamba_names:
        layers[name] = types.SimpleNamespace()  # no TileInfer impl: the GDN layer's own state
    runner.compilation_config.static_forward_context = layers
    return layers


def _compilation_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        mode=_CompilationMode.NONE,
        compile_sizes=None,
        cudagraph_mode=_CUDAGraphMode.NONE,
        cudagraph_capture_sizes=None,
        get_compile_ranges=lambda: [],
        kv_cache_memory_bytes=None,
        compilation_time=1.5,
        encoder_compilation_time=2.5,
        static_forward_context={},
    )


def _new_worker(
    runner: _RecordingRunner,
    *,
    block_size: int = BLOCK_SIZE,
    max_model_len: int = MAX_MODEL_LEN,
    max_num_seqs: int = MAX_NUM_SEQS,
    max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS,
    num_blocks: int = NUM_BLOCKS,
    **attrs: Any,
) -> Any:
    worker = object.__new__(worker_mod.NPUWorker)
    worker.device = DEVICE
    worker.model_runner = runner
    compilation_config = attrs.pop("compilation_config", None) or _compilation_config()
    worker.compilation_config = compilation_config
    worker.vllm_config = types.SimpleNamespace(compilation_config=compilation_config)
    worker.cache_config = types.SimpleNamespace(
        block_size=block_size,
        num_gpu_blocks=num_blocks,
        gpu_memory_utilization=0.5,
        kv_cache_memory_bytes=None,
    )
    worker.model_config = types.SimpleNamespace(
        max_model_len=max_model_len,
        dtype=DTYPE,
        seed=0,
        enforce_eager=True,
        is_hybrid=True,
        enable_return_routed_experts=False,
    )
    worker.scheduler_config = types.SimpleNamespace(
        max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens
    )
    for name, value in attrs.items():
        setattr(worker, name, value)
    return worker


def _install_fake_decode_backend(
    monkeypatch: pytest.MonkeyPatch, events: List[Any]
) -> _FakeBatchAttention:
    backend = _FakeBatchAttention(events)
    # The factory takes the pool/query dtype (it compiles the kernel variant from it).
    monkeypatch.setattr(impl_mod, "_decode_attn_or_none", lambda dtype=None: backend)
    return backend


def _prepared_worker(
    monkeypatch: pytest.MonkeyPatch,
    events: List[Any],
    *,
    bind_pools: bool = True,
) -> Tuple[Any, _RecordingRunner, Dict[str, Any], _FakeBatchAttention, _KVCacheConfig]:
    """Worker + runner + layers (pools bound unless ``bind_pools`` is False) + fake kernel."""
    runner = _RecordingRunner(events=events)
    kv_config, attention_names, mamba_names = _hybrid_kv_config()
    layers = _install_attention_layers(runner, attention_names, mamba_names)
    if bind_pools:
        runner.initialize_kv_cache(kv_config)  # the seam gpu_model_runner.py:7248 uses
    worker = _new_worker(runner)
    backend = _install_fake_decode_backend(monkeypatch, events)
    return worker, runner, layers, backend, kv_config


def _common_metadata(num_reqs: int = 2) -> Any:
    """A decode-step ``CommonAttentionMetadata``: one token per request, one engine page column."""
    query_lens = [1] * num_reqs
    start = [0]
    for length in query_lens:
        start.append(start[-1] + length)
    start_t = torch.tensor(start, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=start_t,
        query_start_loc_cpu=start_t,
        seq_lens=torch.tensor([3, 4][:num_reqs], dtype=torch.int32),
        num_reqs=num_reqs,
        num_actual_tokens=num_reqs,
        max_query_len=1,
        max_seq_len=4,
        block_table_tensor=torch.zeros((num_reqs, 1), dtype=torch.int32),
        slot_mapping=torch.arange(num_reqs, dtype=torch.int64),
    )


def _builder(max_num_seqs: int = MAX_NUM_SEQS) -> TileInferMetadataBuilder:
    return TileInferAttentionBackend.get_builder_cls()(
        kv_cache_spec=_Spec("attention"),
        layer_names=["model.layers.3.self_attn"],
        vllm_config=types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(max_num_seqs=max_num_seqs)
        ),
        device=DEVICE,
    )


def _block_first_pool(num_blocks: int = NUM_BLOCKS) -> torch.Tensor:
    """The pool a hybrid run gets: logical block-first, physically ``[2][blocks][...]``.

    Exactly what ``GPUModelRunner._reshape_kv_cache_tensors`` builds from this backend's own
    declarations (``gpu_model_runner.py:7084-7133``): allocate in
    ``get_kv_cache_stride_order()`` and permute the view back to ``get_kv_cache_shape()``.
    """
    shape = TileInferAttentionBackend.get_kv_cache_shape(
        num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM
    )
    order = TileInferAttentionBackend.get_kv_cache_stride_order()
    physical = torch.zeros(tuple(shape[i] for i in order), dtype=DTYPE)
    return physical.permute(*[order.index(i) for i in range(len(order))])


# =============================================================================================
# Fixtures
# =============================================================================================


@pytest.fixture(scope="session", autouse=True)
def _ensure_model_fixture() -> Any:
    """Recreate ``runtime/test-model-qwen36`` when it is missing (``runtime/`` is gitignored).

    It is a *synthetic* ``config.json`` - no weights, no tokenizer - whose shape is the one the card
    model really has (40 layers = 30 GDN + 10 full attention, 16 heads, 2 KV heads, head_dim 256).
    The suite must not depend on a 67 GB checkpoint, so the rehearsal builds its own model directory.

    The directory is removed again when this fixture created it: a test run should not leave a
    "qwen3.6 config" lying in the workspace for the next reader to wonder about.  A fixture that was
    already there (someone ran ``--plan-only`` by hand) is left untouched.
    """
    config_path = FIXTURE_MODEL / "config.json"
    created = not config_path.is_file()
    if created:
        FIXTURE_MODEL.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(FIXTURE_CONFIG, indent=2) + "\n", encoding="utf-8")
    try:
        yield FIXTURE_MODEL
    finally:
        if created:
            shutil.rmtree(FIXTURE_MODEL, ignore_errors=True)


@pytest.fixture(autouse=True)
def _fresh_runtime_state() -> Any:
    """Every test starts with an empty plan cache and zeroed impl counters."""
    impl_mod.reset_runtime_state()
    impl_mod.reset_counters()
    try:
        yield
    finally:
        impl_mod.reset_runtime_state()
        impl_mod.reset_counters()


@pytest.fixture
def scratch_dir() -> Any:
    """A writable directory for throwaway model configs.

    ``tmp_path`` is not usable in this session: the sandbox denies writes under the system TEMP
    directory (pytest's own basetemp cannot even be created), while the workspace is writable.
    """
    path = ROOT / "runtime" / "rehearsal-tmp" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def isolated_plugin() -> Any:
    """Snapshot/restore ``plugin._STATE`` so install() cannot leak into another test file."""
    snapshot = dict(plugin_mod._STATE)
    try:
        yield plugin_mod
    finally:
        plugin_mod._STATE.clear()
        plugin_mod._STATE.update(snapshot)


@pytest.fixture
def boot_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The vLLM/torch_npu seams ``init_device``/``compile_or_warm_up_model`` reach for."""
    events: List[Any] = []
    npu = _fake_npu()
    monkeypatch.setattr(worker_mod, "_npu_module", lambda what: npu)

    platforms = sys.modules["vllm.platforms"]
    current_platform = TileInferPlatform()
    # Real vLLM's Platform ABC carries ``dist_backend`` (``interface.py``; TileInferPlatform leaves
    # it at '' - UNVERIFIED_NOTES['dist_backend']); the minimal fake base in
    # tests/test_platform_contract.py does not, so the rehearsal supplies the same value.
    current_platform.dist_backend = ""
    monkeypatch.setattr(platforms, "current_platform", current_platform, raising=False)

    gpu_worker = sys.modules["vllm.v1.worker.gpu_worker"]
    monkeypatch.setattr(
        gpu_worker,
        "init_worker_distributed_environment",
        lambda *args, **kwargs: events.append(("dist",)),
        raising=False,
    )
    _fake_module(
        monkeypatch,
        "vllm.utils.torch_utils",
        {"set_random_seed": lambda seed: events.append(("seed", seed))},
    )
    _fake_module(
        monkeypatch,
        "vllm.v1.worker.workspace",
        {
            "init_workspace_manager": lambda device, n: events.append(
                ("workspace", str(device), n)
            )
        },
    )
    _fake_module(
        monkeypatch,
        "vllm.v1.utils",
        {"report_usage_stats": lambda config: events.append(("usage",))},
    )

    # compile_or_warm_up_model's own imports (the CUDA-only branches are not reached).
    _fake_module(monkeypatch, "vllm.config.compilation", {"CompilationMode": _CompilationMode})
    config_mod = sys.modules["vllm.config"]
    monkeypatch.setattr(config_mod, "CUDAGraphMode", _CUDAGraphMode, raising=False)
    _fake_module(
        monkeypatch,
        "vllm.distributed.parallel_state",
        {"get_pp_group": lambda: types.SimpleNamespace(is_last_rank=True)},
    )
    frozen: List[str] = []
    _fake_module(
        monkeypatch,
        "vllm.utils.gc_utils",
        {
            "freeze_gc_heap": lambda: frozen.append("freeze"),
            "maybe_attach_gc_debug_callback": lambda: frozen.append("debug"),
        },
    )
    _fake_module(monkeypatch, "vllm.v1.worker.worker_base", {"CompilationTimes": _CompilationTimes})

    monkeypatch.setattr(
        worker_mod.graphs, "capture_path_ready", lambda: (False, CAPTURE_SKIP_REASON)
    )
    return types.SimpleNamespace(events=events, npu=npu, frozen=frozen)


# =============================================================================================
# 1. the boot rehearsal
# =============================================================================================


def test_boot_rehearsal_walks_the_whole_startup_sequence(
    monkeypatch: pytest.MonkeyPatch,
    boot_env: Any,
    isolated_plugin: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reached: List[str] = []

    def step(name: str) -> None:
        assert name not in reached, f"step {name!r} ran twice"
        reached.append(name)

    # --- 1. plugin.install() ------------------------------------------------------------
    assert plugin_mod.install() is True
    status = plugin_mod.status()
    assert status["plugin"]["installed"] is True
    assert status["vllm"]["available"] is True
    assert status["patches"]["summary"], "the install report must say what the patches did"
    assert status["backend"]["detail"], "the backend step must report its outcome"
    assert status["graphs"]["reason"], "the graph verdict must carry a reason"
    step("plugin.install")

    # --- 2. platform selected -----------------------------------------------------------
    monkeypatch.setattr(platform_mod, "_npu_available", lambda: True)
    assert platform_mod.register() == "ascend_platform.engine.platform.TileInferPlatform"
    step("platform.selected")

    # --- 3. check_and_update_config accepts the Qwen3.6 shape ----------------------------
    config = types.SimpleNamespace(
        speculative_config=None,
        cache_config=types.SimpleNamespace(
            cache_dtype="auto", block_size=128, user_specified_block_size=False
        ),
        model_config=types.SimpleNamespace(
            enable_sleep_mode=False, use_mla=False, is_hybrid=True, enforce_eager=False
        ),
        parallel_config=types.SimpleNamespace(worker_cls="auto"),
    )
    TileInferPlatform.check_and_update_config(config)
    assert config.cache_config.block_size == platform_mod.HYBRID_BLOCK_SIZE == 2048
    assert config.parallel_config.worker_cls == WORKER_CLASS_PATH
    ready, _reason = capture_path_ready()
    assert config.model_config.enforce_eager is (not ready), (
        "capture readiness and enforce_eager must agree (platform.py:_force_eager_until_capture_lands)"
    )
    selector = types.SimpleNamespace(
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        use_non_causal=False,
        use_batch_invariant=False,
        use_kv_connector=False,
        attn_type="decoder",
    )
    assert (
        TileInferPlatform.get_attn_backend_cls(None, selector)
        == platform_mod.ATTN_BACKEND_CLASS_PATH
    )
    # A pinned foreign backend is refused, not resolved to a vLLM one.
    with pytest.raises(ValueError, match="attention-backend"):
        TileInferPlatform.get_attn_backend_cls("flash_attn", selector)
    # A hybrid model with an operator-pinned 128-token page is refused (not silently aligned).
    pinned = types.SimpleNamespace(
        speculative_config=None,
        cache_config=types.SimpleNamespace(
            cache_dtype="auto", block_size=128, user_specified_block_size=True
        ),
        model_config=types.SimpleNamespace(
            enable_sleep_mode=False, use_mla=False, is_hybrid=True, enforce_eager=False
        ),
        parallel_config=types.SimpleNamespace(worker_cls="auto"),
    )
    with pytest.raises(ValueError, match="multiple of 2048"):
        TileInferPlatform.check_and_update_config(pinned)
    step("platform.config")

    # --- 4. worker.init_device ----------------------------------------------------------
    worker = object.__new__(worker_mod.NPUWorker)
    worker.local_rank = 0
    worker.rank = 0
    worker.distributed_init_method = None
    worker.use_v2_model_runner = False
    worker.device_config = types.SimpleNamespace(device_type="npu")
    worker.parallel_config = types.SimpleNamespace(
        distributed_executor_backend=None,
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        data_parallel_rank_local=None,
        data_parallel_index=0,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        local_world_size=1,
        enable_dbo=False,
    )
    worker.cache_config = types.SimpleNamespace(
        block_size=BLOCK_SIZE,
        num_gpu_blocks=None,
        gpu_memory_utilization=0.5,
        kv_cache_memory_bytes=None,
    )
    worker.model_config = types.SimpleNamespace(
        max_model_len=MAX_MODEL_LEN,
        dtype=DTYPE,
        seed=0,
        enforce_eager=True,
        is_hybrid=True,
        enable_return_routed_experts=False,
    )
    worker.scheduler_config = types.SimpleNamespace(
        max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS
    )
    worker.compilation_config = _compilation_config()
    worker.vllm_config = types.SimpleNamespace(
        compilation_config=worker.compilation_config, parallel_config=worker.parallel_config
    )

    worker.init_device()
    assert worker.device.type in worker_mod.NPU_DEVICE_TYPES
    # The real runner subclass was constructed: the num_compute_units seam is closed.
    assert isinstance(worker.model_runner, worker_mod.TileInferModelRunner)
    assert worker.model_runner.num_sms == 40
    assert worker.init_snapshot == worker_mod.NpuMemorySnapshot(
        total_memory=200_000_000,
        free_memory=100_000_000,
        torch_peak=1_000,
        device_=worker.device,
    )
    assert worker.requested_memory == 100_000_000
    event_names = [event[0] for event in boot_env.events]
    assert "dist" in event_names and "workspace" in event_names and "usage" in event_names
    step("worker.init_device")

    # From here the runner is the recording fake: it owns the KV tensors and the layer modules,
    # i.e. the hardware seam - which is exactly what this rehearsal cannot have.
    runner = _RecordingRunner(events=boot_env.events)
    runner.vllm_config = worker.vllm_config
    worker.model_runner = runner

    # --- 5. determine_available_memory --------------------------------------------------
    available = worker.determine_available_memory()
    # requested 100M - weights 1000 - activation/non-torch 0 - graph pool 0
    assert available == 99_999_000
    assert worker.available_kv_cache_memory_bytes == available
    assert runner.called("profile_run")
    step("worker.determine_available_memory")

    # --- 6. initialize_from_config: the KV spec path, 40 groups -------------------------
    kv_config, attention_names, mamba_names = _hybrid_kv_config()
    layers = _install_attention_layers(runner, attention_names, mamba_names)
    worker.initialize_from_config(kv_config)

    # --- 6a. the mamba state-update decision --------------------------------------------
    # Qwen3.6 is 30 GDN layers: vLLM's Mamba SSU dispatch is a no-op for mamba_type=GDN_ATTN,
    # so "our backend selected" here means our ported packed-decode op, not an SSU class.
    assert worker.mamba_state_path["decision"] == "gdn-fla"
    assert worker.mamba_state_path["gdn_layers"] == 30
    assert worker.mamba_state_path["ssu_layers"] == 0
    assert worker.mamba_state_path["types"] == {"GDN_ATTN": 30}

    kinds = [group.kv_cache_spec.kind for group in kv_config.kv_cache_groups]
    assert len(kv_config.kv_cache_groups) == 40
    assert kinds.count("attention") == 10 and kinds.count("mamba") == 30
    assert worker.cache_config.num_gpu_blocks == kv_config.num_blocks
    assert KV_TRANSFER_CALLS[-1] == (worker.vllm_config, kv_config)
    assert ("initialize_kv_cache", kv_config) in runner.calls
    assert runner.count("kv_pool") == 10
    assert runner.called("_init_kv_zero_meta")  # needs_kv_cache_zeroing was set
    for name in attention_names:
        assert isinstance(layers[name].kv_cache, torch.Tensor)
        assert tuple(layers[name].kv_cache.shape) == (
        2,
        NUM_BLOCKS,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    for name in mamba_names:
        assert not hasattr(layers[name], "kv_cache")
    step("worker.initialize_from_config")

    # --- 7. compile_or_warm_up_model: prewarm before capture, sampler after --------------
    backend = _install_fake_decode_backend(monkeypatch, boot_env.events)
    real_capture = worker.capture_model

    def _recorded_capture() -> int:
        boot_env.events.append(("capture",))
        return real_capture()

    monkeypatch.setattr(worker, "capture_model", _recorded_capture, raising=False)

    with caplog.at_level(logging.INFO, logger="ascend_platform.engine.worker"):
        times = worker.compile_or_warm_up_model()

    buckets = planner_mod.buckets_up_to(min(MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS))
    assert worker.prewarm_stats["bucket_set"] == buckets == (1, 2, 4)
    assert worker.prewarm_stats["fitted_bucket_set"] == buckets
    assert worker.prewarm_stats["attempts"] == 1
    assert worker.prewarm_stats["layers"] == 10
    assert worker.prewarm_stats["buckets"] == len(buckets)
    assert worker.prewarm_stats["plans"] == len(buckets)
    assert worker.prewarm_stats["failed"] == 0
    assert worker.prewarm_stats["skipped_reason"] == ""
    assert worker.prewarm_stats["width"] == 16
    assert worker.prewarm_stats["engine_width"] == 1
    assert worker.prewarm_stats["block_size"] == BLOCK_SIZE
    assert worker.prewarm_stats["max_num_reqs"] == MAX_NUM_SEQS
    assert worker.prewarm_stats["pool_cap"] == 128  # next_pow2(5 blocks x 16 sub-pages)
    assert len(impl_mod.PLAN_CACHE) == len(buckets)
    assert impl_mod.IMPL_COUNTERS.get("plan_builds") == len(buckets)
    assert impl_mod.IMPL_COUNTERS.get("state_builds") == 10 * len(buckets)

    # ordering: every plan is built before the capture attempt, and the sampler dummy run
    # stays after capture (gpu_worker.py:708-728).
    plan_positions = [index for index, event in enumerate(boot_env.events) if event[0] == "plan"]
    capture_position = boot_env.events.index(("capture",))
    sampler_position = boot_env.events.index(("sampler",))
    assert plan_positions and max(plan_positions) < capture_position < sampler_position
    assert [bucket for _, bucket in (event for event in boot_env.events if event[0] == "plan")] == [
        1,
        2,
        4,
    ]

    # capture was skipped, counted, and explained - never silently missing.
    assert worker.capture_stats["attempts"] == 1
    assert worker.capture_stats["skipped"] == 1
    assert worker.capture_stats["captured"] == 0
    assert worker.capture_stats["last_skip_reason"] == CAPTURE_SKIP_REASON
    assert not runner.called("capture_model")
    assert any(
        "pre-warmed 3 decode plan" in record.getMessage() for record in caplog.records
    ), [record.getMessage() for record in caplog.records]

    # the sampler / pooled run and the GC freeze are the same order as the GPU worker's.
    sampler_calls = [call for call in runner.calls if call[:1] == ("_dummy_run",)]
    assert sampler_calls and sampler_calls[-1][2].get("num_tokens") == min(
        MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS
    )
    assert runner.called("maybe_remove_all_loras")
    assert runner.called("_dummy_sampler_run")
    assert not runner.called("_dummy_pooler_run")
    assert boot_env.frozen == ["freeze", "debug"]
    assert isinstance(times, _CompilationTimes)
    assert (times.language_model, times.encoder) == (1.5, 2.5)
    step("worker.compile_or_warm_up_model")

    # --- 8. attention builder.build() on a fake CommonAttentionMetadata ------------------
    builder = _builder()
    meta = builder.build(0, _common_metadata(num_reqs=2))
    assert isinstance(meta, TileInferMetadata)
    assert (meta.num_reqs, meta.num_actual_tokens, meta.max_query_len) == (2, 2, 1)
    assert meta.bucket == 2 and meta.is_decode is True and meta.block_size == BLOCK_SIZE
    step("attention.builder.build")

    # --- 9. impl.forward() on fake tensors ----------------------------------------------
    impl = layers[attention_names[0]].impl
    kv_cache = layers[attention_names[0]].kv_cache
    assert 2 in impl._states, "bucket 2's state must have been prepared by the pre-warm"
    states_before = impl_mod.IMPL_COUNTERS.get("state_builds")

    q = torch.randn(2, NUM_HEADS, HEAD_DIM, dtype=DTYPE)
    k = torch.randn(2, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
    v = torch.randn(2, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
    output = torch.zeros(2, NUM_HEADS * HEAD_DIM, dtype=DTYPE)
    result = impl.forward(None, q, k, v, kv_cache, meta, output)

    assert result is output
    assert torch.equal(
        output.view(2, NUM_HEADS, HEAD_DIM), torch.full((2, NUM_HEADS, HEAD_DIM), 7.0, dtype=DTYPE)
    ), "the kernel's output was not copied back into vLLM's output buffer"
    # (a) the KV write happened before attention: the two slots hold this step's K.
    flat_pool = kv_cache[0].view(-1, NUM_KV_HEADS, HEAD_DIM)
    assert torch.equal(flat_pool[0], k[0]) and torch.equal(flat_pool[1], k[1])
    # (b) the ragged triple was rebuilt from seq_lens [3, 4] -> one page each.
    run = backend.runs[-1]
    assert run["kv_indptr"].tolist() == [0, 1, 2]
    assert run["kv_last_page_len"].tolist() == [3, 4]
    assert run["kv_indices"].numel() == 128
    # (c) the plan was a cache hit, not a compile and not a degraded step.
    assert run["plan"] is impl_mod.PLAN_CACHE.get(impl._plan_key(impl._states[2]))
    assert impl_mod.IMPL_COUNTERS.get("decode_steps") == 1
    assert impl_mod.IMPL_COUNTERS.get("kv_writes") == 1
    assert impl_mod.IMPL_COUNTERS.get("plan_not_ready") == 0
    assert impl_mod.IMPL_COUNTERS.get("plan_failed") == 0
    assert impl_mod.IMPL_COUNTERS.get("prefill_reference") == 0
    assert impl_mod.IMPL_COUNTERS.get("state_builds") == states_before  # no allocation in the step
    step("attention.impl.forward")

    # --- "no step silently swallows an exception" --------------------------------------
    # Each refusal is raised *and* counted, so "the engine never got here" is distinguishable.
    before = impl_mod.IMPL_COUNTERS.get("metadata_mismatch")
    with pytest.raises(TypeError, match="TileInferMetadata"):
        impl.forward(None, q, k, v, kv_cache, object(), output)
    assert impl_mod.IMPL_COUNTERS.get("metadata_mismatch") == before + 1
    with pytest.raises(ValueError, match="common_prefix_len"):
        builder.build(64, _common_metadata())
    with pytest.raises(ValueError, match="max_num_seqs"):
        builder.build(0, _common_metadata(num_reqs=MAX_NUM_SEQS + 1))

    # The whole sequence ran, in order, once.  A missing step fails here.
    assert tuple(reached) == STEPS


# =============================================================================================
# 2. the pre-warm's own contract: idempotent, skip-able, shape-only
# =============================================================================================


def test_the_decode_plan_prewarm_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, boot_env: Any
) -> None:
    worker, _runner, _layers, backend, _kv_config = _prepared_worker(monkeypatch, boot_env.events)

    assert worker._prewarm_decode_plans() == 3
    first = dict(worker.prewarm_stats)
    assert first["plans"] == 3 and first["failed"] == 0
    assert len(impl_mod.PLAN_CACHE) == 3

    assert worker._prewarm_decode_plans() == 0, "a second pre-warm must build nothing"
    assert len(impl_mod.PLAN_CACHE) == 3
    assert worker.prewarm_stats["plans"] == 3  # cumulative: still the same three plans
    assert worker.prewarm_stats["attempts"] == 2
    assert impl_mod.IMPL_COUNTERS.get("plan_builds") == 3
    assert impl_mod.IMPL_COUNTERS.get("state_builds") == 30
    assert len(backend.plan_kwargs) == 3


def test_the_prewarm_skips_with_a_logged_reason_when_there_is_no_device(
    monkeypatch: pytest.MonkeyPatch, boot_env: Any, caplog: pytest.LogCaptureFixture
) -> None:
    worker, _runner, layers, _backend, _kv_config = _prepared_worker(monkeypatch, boot_env.events)
    monkeypatch.setattr(impl_mod, "_decode_attn_or_none", lambda dtype=None: None)  # no kernel

    with caplog.at_level(logging.WARNING, logger="ascend_platform.engine.worker"):
        assert worker._prewarm_decode_plans() == 0  # must not raise
    reason = worker.prewarm_stats["skipped_reason"]
    assert "kernel backend is unavailable" in reason
    assert worker.prewarm_stats["plans"] == 0
    assert len(impl_mod.PLAN_CACHE) == 0
    for name, layer in layers.items():
        if hasattr(layer, "impl"):
            assert layer.impl._states == {}, f"{name} allocated a state without a device"
    assert any("pre-warm skipped" in record.getMessage() for record in caplog.records)


def test_the_shape_only_prewarm_keeps_the_plans_and_lets_the_first_step_bind_the_pool(
    monkeypatch: pytest.MonkeyPatch, boot_env: Any
) -> None:
    """The documented ordering constraint: no bound pool yet -> compile from shape, drop state."""
    worker, _runner, layers, _backend, _kv_config = _prepared_worker(
        monkeypatch, boot_env.events, bind_pools=False
    )
    assert worker._prewarm_decode_plans() == 3
    impl = layers["model.layers.3.self_attn"].impl
    assert impl._states == {}, (
        "the synthetic state must be dropped: the first real step would otherwise hit the impl's "
        "data_ptr check (attn/impl.py:_state_for)"
    )
    assert len(impl_mod.PLAN_CACHE) == 3

    # The engine binds the real pool afterwards (as initialize_kv_cache does); the first step
    # must allocate its own state and still hit the plan that was compiled from the shape.
    pool = torch.empty((2, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM), dtype=DTYPE)
    layers["model.layers.3.self_attn"].kv_cache = pool
    meta = _builder().build(0, _common_metadata(num_reqs=2))
    q = torch.randn(2, NUM_HEADS, HEAD_DIM, dtype=DTYPE)
    k = torch.randn(2, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
    v = torch.randn(2, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
    output = torch.zeros(2, NUM_HEADS * HEAD_DIM, dtype=DTYPE)

    result = impl.forward(None, q, k, v, pool, meta, output)
    assert result is output
    assert torch.equal(
        output.view(2, NUM_HEADS, HEAD_DIM),
        torch.full((2, NUM_HEADS, HEAD_DIM), 7.0, dtype=DTYPE),
    ), "the kernel's output was not copied back into vLLM's output buffer"
    assert 2 in impl._states
    assert impl._states[2].k_pool.data_ptr() == buckets_mod.pool_sub_view(
        pool[0], buckets_mod.sub_pages(BLOCK_SIZE)
    ).data_ptr(), "the first step must bind the layer's own pool, not the placeholder"
    assert impl_mod.IMPL_COUNTERS.get("plan_builds") == 3, "the first step must not recompile"
    assert impl_mod.IMPL_COUNTERS.get("plan_not_ready") == 0


def test_a_hybrid_block_major_pool_serves_a_decode_step(
    monkeypatch: pytest.MonkeyPatch, boot_env: Any
) -> None:
    """The pool a *hybrid* run actually gets: block-first, physically K/V-outermost.

    `get_kv_cache_shape` puts ``num_blocks`` in dim 0, so vLLM's hybrid layout rewrite skips the
    pool (``gpu_model_runner.py:7190-7191``) instead of interleaving K and V per block; the
    ``get_kv_cache_stride_order`` declaration makes the physical order ``[2][blocks][...]``, so the
    logical tensor is a non-contiguous permutation and ``kv_cache[:, 0]`` is a *contiguous*
    ``[blocks, page, heads, dim]`` half.  One eager step must write this step's K into the K half
    (and V into the V half) and hand the kernel a zero-copy 128-token page view of exactly that
    memory.  Writing the wrong half, or writing at ``(offset, block)``, would still produce a
    plausible-looking pool - hence the placement assertions below.
    """
    worker, _runner, layers, _backend, _kv_config = _prepared_worker(monkeypatch, boot_env.events)
    assert worker._prewarm_decode_plans() == 3  # the plan is capacity-keyed, not pointer-keyed
    impl = layers["model.layers.3.self_attn"].impl
    impl._states.clear()  # a different pool below: the prepared state names the old data_ptr

    # Exactly what `GPUModelRunner._reshape_kv_cache_tensors` builds from this backend's own
    # declarations (gpu_model_runner.py:7084-7133): the logical shape is block-first, the physical
    # order comes from `get_kv_cache_stride_order`, and the view is permuted back.
    shape = TileInferAttentionBackend.get_kv_cache_shape(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM
    )
    pool = _block_first_pool()
    assert not pool.is_contiguous(), "the hybrid pool is a permuted, non-contiguous view"
    assert tuple(pool.shape) == shape
    assert pool[:, 0].is_contiguous() and pool[:, 1].is_contiguous()
    assert pool[:, 0].data_ptr() == pool.data_ptr()
    layers["model.layers.3.self_attn"].kv_cache = pool

    meta = _builder().build(0, _common_metadata(num_reqs=2))  # slots [0, 1]
    q = torch.randn(2, NUM_HEADS, HEAD_DIM, dtype=DTYPE)
    k = torch.randn(2, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
    v = torch.randn(2, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
    output = torch.zeros(2, NUM_HEADS * HEAD_DIM, dtype=DTYPE)

    result = impl.forward(None, q, k, v, pool, meta, output)
    assert result is output
    assert torch.equal(
        output.view(2, NUM_HEADS, HEAD_DIM),
        torch.full((2, NUM_HEADS, HEAD_DIM), 7.0, dtype=DTYPE),
    ), "the kernel's output was not copied back into vLLM's output buffer"

    # K landed in the K half and V in the V half, at the flat slots the metadata named...
    k_flat = pool[:, 0].view(-1, NUM_KV_HEADS, HEAD_DIM)
    v_flat = pool[:, 1].view(-1, NUM_KV_HEADS, HEAD_DIM)
    assert torch.equal(k_flat[0], k[0]) and torch.equal(k_flat[1], k[1])
    assert torch.equal(v_flat[0], v[0]) and torch.equal(v_flat[1], v[1])
    # ...nothing else was touched (the physical buffer started zeroed), so a block/offset swap or
    # a dropped K/V index cannot pass unnoticed.
    assert int(torch.count_nonzero(k_flat[2:])) == 0
    assert int(torch.count_nonzero(v_flat[2:])) == 0

    # (b) the kernel got the contiguous 128-token page view of that same memory - zero copy.
    state = impl._states[2]
    assert state.k_pool.is_contiguous()
    assert state.k_pool.data_ptr() == pool[:, 0].data_ptr()
    assert state.v_pool.data_ptr() == pool[:, 1].data_ptr()
    assert tuple(state.k_pool.shape) == (
        NUM_BLOCKS * buckets_mod.sub_pages(BLOCK_SIZE),
        buckets_mod.KERNEL_PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    # the plan was a cache hit (the capacity is unchanged by the layout), not a recompile.
    assert impl_mod.IMPL_COUNTERS.get("plan_builds") == 3
    assert impl_mod.IMPL_COUNTERS.get("plan_not_ready") == 0
    assert impl_mod.IMPL_COUNTERS.get("decode_steps") == 1


def test_the_prewarm_gate_recognises_a_block_first_pool(
    monkeypatch: pytest.MonkeyPatch, boot_env: Any
) -> None:
    """R2: the pool gate discovers the K/V axis instead of assuming ``(2, N, ...)``.

    Before the layout change the gate was ``pool.shape[0] == 2``; a block-first pool has
    ``num_blocks`` there, so it silently fell into the shape-only path (and ``pool[0]``, block 0's
    ``(2, page, ...)`` slice, would have been the wrong half).  Both orders must now produce the
    *same* declared capacity and therefore the *same* plan key.
    """
    worker, _runner, layers, _backend, _kv_config = _prepared_worker(monkeypatch, boot_env.events)
    assert worker._prewarm_decode_plans() == 3
    impl = layers["model.layers.3.self_attn"].impl
    legacy_keys = {bucket: impl._plan_key(impl._states[bucket]) for bucket in (1, 2, 4)}

    pool = _block_first_pool()
    for layer in layers.values():
        if hasattr(layer, "impl"):
            layer.kv_cache = pool
    sub = buckets_mod.sub_pages(BLOCK_SIZE)

    got = worker._prewarm_layer_pool(
        layers["model.layers.3.self_attn"], {"block_size": BLOCK_SIZE}, sub
    )
    assert got["synthetic"] is False, "a block-first pool must not fall into the shape-only path"
    assert got["pages"] == NUM_BLOCKS * sub
    assert got["k_pool"].data_ptr() == pool[:, 0].data_ptr()
    assert got["v_pool"].data_ptr() == pool[:, 1].data_ptr()

    # The K half's page count - and only that - feeds `declared_pool_cap`, so the plan key is
    # unchanged: the re-prewarm against the block-first pool must build nothing.
    assert worker._prewarm_decode_plans() == 0
    assert worker.prewarm_stats["failed"] == 0
    assert len(impl_mod.PLAN_CACHE) == 3
    for bucket in (1, 2, 4):
        assert impl._states[bucket].k_pool.data_ptr() == pool[:, 0].data_ptr()
        assert impl._plan_key(impl._states[bucket]) == legacy_keys[bucket]


def test_a_bucket_that_does_not_fit_the_pool_is_trimmed_counted_and_logged(
    monkeypatch: pytest.MonkeyPatch, boot_env: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The impl's staging table needs ``bucket*width < cap-1``; an impossible bucket is refused
    up front (logged) instead of being handed to the kernel path as a guaranteed failure."""
    worker, _runner, _layers, _backend, _kv_config = _prepared_worker(monkeypatch, boot_env.events)
    worker.model_config.max_model_len = 16384  # 8 engine pages -> 128 kernel pages of table width

    with caplog.at_level(logging.WARNING, logger="ascend_platform.engine.worker"):
        assert worker._prewarm_decode_plans() == 0
    assert worker.prewarm_stats["buckets"] == 3
    assert worker.prewarm_stats["fitted_bucket_set"] == ()
    assert worker.prewarm_stats["plans"] == 0
    assert "do not fit its KV pool" in " ".join(
        record.getMessage() for record in caplog.records
    )
    assert "no decode plan could be built" in worker.prewarm_stats["skipped_reason"]
    assert len(impl_mod.PLAN_CACHE) == 0


def test_a_bucket_state_bound_to_a_stale_pool_is_rebuilt(
    monkeypatch: pytest.MonkeyPatch, boot_env: Any
) -> None:
    """vLLM's profiling pass builds a throwaway pool and clears it; a state must not outlive it."""
    worker, _runner, layers, _backend, _kv_config = _prepared_worker(monkeypatch, boot_env.events)
    assert worker._prewarm_decode_plans() == 3
    impl = layers["model.layers.3.self_attn"].impl
    replacement = torch.empty_like(layers["model.layers.3.self_attn"].kv_cache)
    layers["model.layers.3.self_attn"].kv_cache = replacement

    assert worker._prewarm_decode_plans() == 0  # the plans are keyed by capacity, not pointer
    expected = buckets_mod.pool_sub_view(
        replacement[0], buckets_mod.sub_pages(BLOCK_SIZE)
    ).data_ptr()
    for bucket in (1, 2, 4):
        assert impl._states[bucket].k_pool.data_ptr() == expected


def test_the_prewarm_skips_before_the_kv_pool_exists(
    monkeypatch: pytest.MonkeyPatch, boot_env: Any
) -> None:
    """The ordering constraint, stated as a skip: no pool -> no declared capacity -> no plan."""
    worker, _runner, _layers, _backend, _kv_config = _prepared_worker(
        monkeypatch, boot_env.events, bind_pools=False
    )
    worker.cache_config.num_gpu_blocks = 0
    assert worker._prewarm_decode_plans() == 0
    assert "KV pool has not been allocated yet" in worker.prewarm_stats["skipped_reason"]
    assert len(impl_mod.PLAN_CACHE) == 0

# =============================================================================================
# Not migrated: the launcher contract
# =============================================================================================
#
# The source repo pinned ``scripts/serve_qwen36.py`` here: its plan arithmetic against
# ``serving/buckets.py``, its refusals, ``--plan-only``, and its argv against the engine's own
# derivation.  The launcher is a *deployment recipe*, not platform semantics (docs/migration-from-
# tileinfer.md §4), and this repo does not ship ``scripts/``.  Those 15 assertions stay with the
# launcher's owner; what the platform owes them is the unchanged arithmetic they pin, which
# ``test_boot_rehearsal.py::test_the_plan_*`` and ``tests/test_plan_cache.py`` already cover.


@pytest.mark.skipif(operator_available(), reason="this pins the behaviour with NO operator layer installed")
def test_a_missing_operator_layer_degrades_to_a_report_instead_of_a_failed_boot(
    boot_env: Any, isolated_plugin: Any
) -> None:
    """The fallback the plan promised: no operator layer ⇒ the engine still boots, loudly.

    "拔掉算子层能起服务" is easy to read as "vendor kernels take over", which is not what happens on
    Ascend: there is no vLLM-side GDN implementation to fall back to, so the first forward is the
    thing that fails.  What must hold — and is pinned here — is that *start-up* survives: the plugin
    installs, the ops step records a readable reason, and nothing raises before the model runs.
    """
    assert plugin_mod.install() is True
    ops = plugin_mod.status()["ops"]
    assert ops, "the ops step must report something even when it could not run"
    assert ops["bound"] == [], "nothing can be bound without a layer"
    detail = ops["detail"]
    assert "OperatorLayerUnavailable" in detail or "no operator layer" in detail, (
        f"the reason must name the missing layer, not just 'failed': {detail!r}"
    )
    # The summary has to say *which* layer it looked for and where to point it: "0 bound" alone
    # reads as "there was nothing to bind", which is the one thing it does not mean.
    # "unavailable", not "missing": the same field now also carries a version *mismatch* (the layer
    # is there and speaks a different contract), which "missing" would describe wrongly.
    assert "operator layer unavailable" in detail
    assert "ascend_attention" in detail and "ASCEND_PLATFORM_OPS" in detail
