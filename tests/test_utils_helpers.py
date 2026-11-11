"""The device helpers every layer goes through — pure logic, so it can be pinned on a CPU box.

``ascend_platform.utils`` is deliberately tiny and boring: it is the one place that decides "NPU or
CPU", and it must work on a laptop with no ``torch_npu`` at all (that property is what lets the whole
package be imported and tested without hardware).  Boring code still rots, and the two functions
that matter are the ones whose failures are silent:

* :func:`quantize_capacity` feeds compile keys.  A regression there does not raise — it makes every
  capacity a new compile key, which shows up as a serving process that stalls minutes into a run
  (``docs/known-issues.md``).  So the powers of two are pinned explicitly, including "never shrink".
* :func:`resolve_device` is the only place allowed to *refuse*: asking for an NPU on a box without
  one must raise with a sentence a user can act on, not fall back to CPU and produce a wrong answer
  on the wrong device.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ascend_platform import utils


@pytest.mark.parametrize(
    "given,expected",
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 4),
        (4, 4),
        (5, 8),
        (7, 8),
        (8, 8),
        (1000, 1024),
        (1024, 1024),
        (1025, 2048),
        (-3, 1),
        ("9", 16),
    ],
)
def test_quantize_capacity_rounds_up_to_a_power_of_two(given, expected):
    assert utils.quantize_capacity(given) == expected


def test_quantize_capacity_never_shrinks_and_is_idempotent():
    """The property the compile cache depends on: one bucket, and re-quantising is a no-op."""
    for n in range(0, 5000):
        bucket = utils.quantize_capacity(n)
        assert bucket >= n
        assert bucket & (bucket - 1) == 0, f"{n} -> {bucket} is not a power of two"
        assert utils.quantize_capacity(bucket) == bucket


def test_resolve_device_defaults_to_cpu_without_an_npu():
    if utils.is_npu_available():  # a real device box: the NPU branch is the interesting one
        assert utils.resolve_device(None) == torch.device("npu", 0)
        pytest.skip("this machine has an NPU; the CPU fallback cannot be observed here")
    assert utils.resolve_device(None) == torch.device("cpu")
    assert utils.resolve_device("auto") == torch.device("cpu")


def test_resolve_device_honours_an_explicit_device_object_and_name():
    given = torch.device("cpu", 0)
    assert utils.resolve_device(given) is given, "an explicit torch.device must pass through"
    assert utils.resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_refuses_an_npu_that_is_not_there():
    """Falling back to CPU here would silently compute on the wrong device."""
    if utils.is_npu_available():
        pytest.skip("this machine has an NPU, so the refusal branch is unreachable")
    with pytest.raises(RuntimeError) as caught:
        utils.resolve_device("npu")
    message = str(caught.value)
    assert "NPU" in message and "unavailable" in message
    assert "torch_npu" in message, "the message must name what to install"


def test_device_name_is_readable_on_cpu():
    assert utils.device_name(torch.device("cpu")) == "cpu:0"
    assert utils.device_name(torch.device("cpu", 3)) == "cpu:3"


def test_default_dtype_is_the_fast_path_dtype():
    assert utils.default_dtype(torch.device("cpu")) is torch.float16


def test_synchronize_is_a_no_op_on_cpu():
    utils.synchronize()  # None: nothing to wait for
    utils.synchronize(torch.device("cpu"))  # explicit CPU: same answer


def test_is_npu_available_and_device_count_agree():
    """The count is derived from availability, so a mismatch means one of the two swallowed an error."""
    if utils.is_npu_available():
        assert utils.npu_device_count() >= 1
    else:
        assert utils.npu_device_count() == 0
