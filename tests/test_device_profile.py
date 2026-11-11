"""The device seam: one frozen record per card, validated at registration, extensible by a file.

These tests are the reason the seam exists.  In the source repo the same numbers lived in three
modules and two docs, so "which card is this?" could only be answered by reading code.  Here it is a
record, and adding a card is one :func:`register_device_profile` call — test
``test_a_second_card_is_one_registration`` is that claim, executable.
"""

from __future__ import annotations

import pathlib

import types

import pytest

from ascend_platform.device import (
    DEFAULT_DEVICE_PROFILE,
    DeviceProfile,
    EnvRequirement,
    detect_device_profile,
    device_profile,
    known_device_profiles,
    register_device_profile,
    unregister_device_profile,
)


def _profile(**overrides):
    base = dict(
        name="test-card",
        hbm_bytes=1024,
        compute_units=8,
        kernel_page_size=128,
        engine_page_multiple=2048,
        state_page_bytes=4096,
        capabilities=("graph_capture",),
    )
    base.update(overrides)
    return DeviceProfile(**base)


# ---------------------------------------------------------------------------------------------
# The record validates its own coherence
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"kernel_page_size": 100}, "power of two"),
        ({"kernel_page_size": 0}, "positive power of two"),
        ({"engine_page_multiple": 1000}, "multiple of kernel_page_size"),
        ({"dtypes": ()}, "at least one compute dtype"),
        ({"head_sizes": ()}, "at least one head size"),
        ({"min_gqa_group": 0}, "min_gqa_group"),
    ],
)
def test_validate_rejects_an_incoherent_profile(overrides, message):
    with pytest.raises(ValueError, match=message):
        _profile(**overrides).validate()


def test_validate_returns_self_so_it_can_be_chained_at_the_call_site():
    profile = _profile()
    assert profile.validate() is profile


# ---------------------------------------------------------------------------------------------
# The queries the platform actually gates on
# ---------------------------------------------------------------------------------------------


def test_the_page_geometry_query_is_what_the_kv_pool_will_be_reviewed_through():
    profile = _profile()
    profile.check_engine_page(2048)  # a whole number of 128-token kernel pages
    profile.check_engine_page(128)
    with pytest.raises(ValueError, match="not a multiple of the 128-token kernel page"):
        profile.check_engine_page(1024 + 1)


def test_dtype_and_layout_queries_read_the_declared_surface():
    profile = _profile(dtypes=("bfloat16",), kv_cache_dtypes=("auto",), layouts=("ND",))
    bf16 = types.SimpleNamespace(name="bfloat16")
    fp16 = types.SimpleNamespace(name="float16")
    assert profile.supports_dtype(bf16) and not profile.supports_dtype(fp16)
    assert profile.supports_dtype("bfloat16"), "a bare name is accepted too"
    assert profile.supports_kv_cache_dtype("auto") and not profile.supports_kv_cache_dtype("fp8")
    assert profile.has("graph_capture") and not profile.has("mamba_state")


def test_as_dict_is_json_shaped_and_keeps_the_env_requirements():
    profile = _profile(
        env=(EnvRequirement(name="ACL_OP_INIT_MODE", expected="1", failure_mode="wrong numbers"),)
    )
    data = profile.as_dict()
    assert data["name"] == "test-card"
    assert data["env"] == [
        {
            "name": "ACL_OP_INIT_MODE",
            "expected": "1",
            "required": True,
            "failure_mode": "wrong numbers",
            "source": "",
        }
    ]


# ---------------------------------------------------------------------------------------------
# The registry: closed by name, open by call
# ---------------------------------------------------------------------------------------------


def test_a_second_card_is_one_registration():
    """The extension story, executable: a new card must not need a new branch in the platform."""
    profile = _profile(name="hypothetical-second-card", compute_units=16)
    register_device_profile(profile)
    try:
        assert known_device_profiles() == tuple(sorted(known_device_profiles()))
        assert device_profile("hypothetical-second-card") is profile
    finally:
        unregister_device_profile("hypothetical-second-card")
    assert "hypothetical-second-card" not in known_device_profiles()


def test_a_duplicate_name_is_refused_unless_replacement_is_asked_for():
    profile = _profile(name="duplicate-card")
    register_device_profile(profile)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_device_profile(_profile(name="duplicate-card", compute_units=99))
        replacement = _profile(name="duplicate-card", compute_units=99)
        register_device_profile(replacement, replace_existing=True)
        assert device_profile("duplicate-card").compute_units == 99
    finally:
        unregister_device_profile("duplicate-card")


def test_an_unknown_name_raises_with_the_known_names_in_the_message():
    with pytest.raises(KeyError) as excinfo:
        device_profile("no-such-card")
    assert "Ascend950PR" in str(excinfo.value)


def test_registration_validates_before_storing():
    with pytest.raises(ValueError, match="power of two"):
        register_device_profile(_profile(name="broken-card", kernel_page_size=100))
    assert "broken-card" not in known_device_profiles()


# ---------------------------------------------------------------------------------------------
# Detection: never raises, never invents numbers
# ---------------------------------------------------------------------------------------------


def test_detection_matches_the_reference_card_by_name():
    assert detect_device_profile(device_name="Ascend950PR_9579").name == "Ascend950PR"


def test_detection_returns_a_coherent_unknown_card_rather_than_the_reference_numbers():
    profile = detect_device_profile(device_name="Ascend9999PR")
    assert profile.name == "Ascend9999PR"
    assert profile.kernel_page_size == DEFAULT_DEVICE_PROFILE.kernel_page_size
    assert "unprofiled" in profile.evidence
    assert profile.name_patterns == ("Ascend9999PR",)


def test_detection_without_a_device_yields_the_fallback_profile():
    assert detect_device_profile(npu=object()) is DEFAULT_DEVICE_PROFILE


def test_the_fallback_profile_is_itself_coherent():
    DEFAULT_DEVICE_PROFILE.validate()
    assert DEFAULT_DEVICE_PROFILE.engine_page_multiple % DEFAULT_DEVICE_PROFILE.kernel_page_size == 0


# ---------------------------------------------------------------------------------------------
# The reference card's numbers, pinned
# ---------------------------------------------------------------------------------------------


def test_the_reference_card_publishes_the_measured_numbers():
    profile = device_profile("Ascend950PR")
    assert profile.compute_units == 56
    assert profile.cube_units == 28 and profile.vector_units == 56
    assert profile.hbm_bytes == 131072 * 1024 * 1024
    assert profile.kernel_page_size == 128
    assert profile.engine_page_multiple == 2048
    assert profile.state_page_bytes == 4194304
    assert profile.head_sizes == (128, 256)
    assert profile.min_gqa_group == 2
    assert profile.dtypes == ("bfloat16", "float16")


def test_the_reference_card_declares_the_capabilities_the_platform_gates_on():
    profile = device_profile("Ascend950PR")
    assert profile.has("graph_capture")
    assert profile.has("hybrid_kv")
    assert profile.has("mamba_state")
    # A capability that is absent must be refused, not degraded: nothing may claim pin_memory
    # support while the device path cannot allocate pinned NPU memory.
    assert not profile.has("pin_memory")


def test_the_state_page_is_exactly_one_attention_page():
    """vLLM compares page sizes across the attention and state groups; a mismatch is a start-up error."""
    profile = device_profile("Ascend950PR")
    attention_page = 2 * 2 * 256 * 2 * profile.engine_page_multiple
    assert attention_page == profile.state_page_bytes


def test_the_environment_requirements_name_the_failure_mode():
    profile = device_profile("Ascend950PR")
    acl = next(item for item in profile.env if item.name == "ACL_OP_INIT_MODE")
    assert acl.required and acl.expected == "1"
    assert "wrong numbers" in acl.failure_mode, (
        "the interesting part is how it fails: unset ACL_OP_INIT_MODE produces wrong numerics, "
        "which is why the check cannot be an assert on an exception"
    )
    assert all(item.failure_mode for item in profile.env)


def test_every_registered_profile_is_coherent():
    for name in known_device_profiles():
        device_profile(name).validate()


# ---------------------------------------------------------------------------------------------
# The read points: the profile and the modules that still carry a literal must agree
# ---------------------------------------------------------------------------------------------
#
# Two kinds of number ended up in ``DeviceProfile`` and they must not be confused:
#
# * **card facts** (HBM, compute units, dtypes, capabilities) — they differ per card, and
#   ``engine/`` reads them through ``device_profile()`` at run time;
# * **kernel geometry** (128-token page, 2048-token hybrid page) — it is fixed by the operator
#   build, not by the card, so deriving it per card would be false indirection.  It is recorded in
#   the profile and *asserted equal* here, which catches a literal that drifts without pretending
#   to be a per-card value.


def test_the_kernel_page_is_the_same_number_in_the_profile_and_in_the_serving_arithmetic():
    from ascend_platform.runtime import buckets

    assert buckets.KERNEL_PAGE_SIZE == DEFAULT_DEVICE_PROFILE.kernel_page_size == 128, (
        "the 128-token page is the kernels' BC tile: runtime/buckets.py and the profile must not "
        "drift apart, or sub-page mapping silently reads the wrong rows"
    )


def test_the_hybrid_page_is_the_same_number_in_the_profile_the_model_profile_and_the_platform():
    from ascend_platform.profiles import active_profile

    assert DEFAULT_DEVICE_PROFILE.engine_page_multiple == 2048
    assert active_profile().block_size_policy.required_multiple(hybrid=True) == (
        DEFAULT_DEVICE_PROFILE.engine_page_multiple
    ), "the model profile's hybrid page rule and the device profile must agree"

    try:
        from ascend_platform.engine import platform as platform_mod
    except ImportError as exc:  # engine/ needs vLLM; this file is also run without it
        pytest.skip(f"engine.platform is not importable here: {exc}")
    assert platform_mod.HYBRID_BLOCK_SIZE == DEFAULT_DEVICE_PROFILE.engine_page_multiple, (
        "the hybrid engine page is what the mamba state page forces; the platform's boot-time "
        "alias and the device profile must agree"
    )


# =============================================================================================
# "adding a card is one module plus one line" — the *line* is the part people forget
# =============================================================================================

DEVICE_PKG = pathlib.Path(__file__).resolve().parents[1] / "src" / "ascend_platform" / "device"
#: Modules that are infrastructure, not a card.
INFRASTRUCTURE = {"__init__", "profile"}


def _card_modules():
    return sorted(
        path for path in DEVICE_PKG.glob("*.py") if path.stem not in INFRASTRUCTURE
    )


def test_the_card_inventory_is_not_empty():
    assert _card_modules(), "no card modules found; the scan is broken"


def test_every_card_module_is_imported_by_the_package():
    """A card file that nobody imports is a profile that never registers — and nothing fails.

    That is the failure mode the docs used to describe as "one file, no other edit": the module
    would sit there, ``detect_device_profile()`` would fall back to the reference numbers, and the
    second box would silently run with the first box's HBM/unit counts.
    """
    init = (DEVICE_PKG / "__init__.py").read_text(encoding="utf-8")
    missing = [
        path.name for path in _card_modules() if f"from . import {path.stem}" not in init
    ]
    assert not missing, (
        "these card modules are not imported by device/__init__.py, so they never register: "
        f"{missing}. Add `from . import <name>` (the import's side effect *is* the registration)."
    )


def test_every_card_module_registers_a_profile():
    """The other half: a module that is imported but only defines helpers is not a card."""
    silent = [
        path.name
        for path in _card_modules()
        if "register_device_profile(" not in path.read_text(encoding="utf-8")
    ]
    assert not silent, f"card modules that never call register_device_profile(): {silent}"
