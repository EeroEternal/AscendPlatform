"""The first real ``DeviceProfile``: Ascend950PR (the only card this backend has been validated on).

Every number here is measured or read out of the source repo; the ``evidence`` field keeps the
provenance so a second card can be added by copying this file and changing numbers, not by editing
the platform.
"""

from __future__ import annotations

from .profile import DeviceProfile, EnvRequirement, register_device_profile

#: 131072 MB of HBM as reported by ``torch.npu.get_device_properties(0).total_memory``.
HBM_BYTES = 131072 * 1024 * 1024

#: One attention page: 2 (K+V) x 2 kv_heads x 256 head_dim x 2 B = 2048 B/token, x 2048 tokens.
ATTENTION_PAGE_BYTES = 2 * 2 * 256 * 2 * 2048

#: The state page is padded to exactly one attention page (vLLM compares pages across groups).
STATE_PAGE_BYTES = 4194304

ASCEND950PR = DeviceProfile(
    name="Ascend950PR",
    vendor="ascend",
    dispatch_key="NPU",
    hbm_bytes=HBM_BYTES,
    compute_units=56,
    cube_units=28,
    vector_units=56,
    kernel_page_size=128,
    engine_page_multiple=2048,
    state_page_bytes=STATE_PAGE_BYTES,
    dtypes=("bfloat16", "float16"),
    kv_cache_dtypes=("auto", "bfloat16", "float16"),
    layouts=("ND", "NZ"),
    head_sizes=(128, 256),
    min_gqa_group=2,
    capabilities=("graph_capture", "hybrid_kv", "mamba_state"),
    env=(
        EnvRequirement(
            name="ACL_OP_INIT_MODE",
            expected="1",
            required=True,
            failure_mode=(
                "wrong numbers, not an error: CANN silently upgrades the mode and disables aclops"
            ),
            source="docs/hardware-checklist.md, architecture.md 'Environment'",
        ),
        EnvRequirement(
            name="TILELANG_CACHE_DIR",
            required=False,
            failure_mode=(
                "every kernel recompiles in-process (~70 s holding the GIL), which looks like an "
                "engine hang; the cache dir must also be identical for pre-warm and serve"
            ),
            source="docs/hardware-checklist.md step 1.3",
        ),
    ),
    name_patterns=("950", "Ascend950"),
    evidence=(
        "measured on Ascend950PR_9579 with torch_npu 2.10.0.post4 / 2.13.0: "
        "multi_processor_count=56, cube_core_num=28, vector_core_num=56, 131072 MB HBM; "
        "page geometry from docs/hardware-checklist.md step 6"
    ),
).validate()

register_device_profile(ASCEND950PR)
