"""What the hardware *is*, as data.

``DeviceProfile`` is the single place device constants may live: HBM size, core counts, page
geometry, supported dtypes/layouts, capability bits, and the environment requirements that make the
toolchain behave.  Platform code reads a profile; it never reads a literal.

Adding a second card = adding one module that builds a :class:`DeviceProfile` and registers it,
plus one line here (``from . import <card>``) whose side effect is the registration.  The import is
deliberately explicit rather than discovered: the dependency graph stays static, which is what
``_acl`` and the import-whitelist tests read.  ``tests/test_device_profile.py`` fails when a module
in this directory is missing from the list below.
"""

from __future__ import annotations

from .profile import (
    DeviceProfile,
    EnvRequirement,
    detect_device_profile,
    device_profile,
    known_device_profiles,
    register_device_profile,
    unregister_device_profile,
    DEFAULT_DEVICE_PROFILE,
)
from . import ascend950  # noqa: F401  (registers the first real profile)

__all__ = [
    "DeviceProfile",
    "EnvRequirement",
    "detect_device_profile",
    "device_profile",
    "known_device_profiles",
    "register_device_profile",
    "unregister_device_profile",
    "DEFAULT_DEVICE_PROFILE",
]
