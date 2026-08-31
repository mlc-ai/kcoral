"""Resolve a physical GPU id to its UUID, through NVML over ctypes.

CUDA numbers only the devices a process may open, so hiding a card from other
users renumbers the rest and a physical id stops naming the same GPU. NVML
enumerates every card regardless, so pinning a worker by UUID survives that.
"""

from __future__ import annotations

import ctypes
from functools import cache
from typing import Any

_NVML_SUCCESS = 0
_UUID_BUFFER_SIZE = 96  # NVML_DEVICE_UUID_V2_BUFFER_SIZE


class _NVMLAPI:
    """The small NVML surface needed to name a GPU by identity."""

    def __init__(self, library: Any) -> None:
        if library.nvmlInit_v2() != _NVML_SUCCESS:
            raise OSError("nvmlInit_v2 failed")
        self._handle_by_index = library.nvmlDeviceGetHandleByIndex_v2
        self._device_uuid = library.nvmlDeviceGetUUID

    def device_uuid(self, gpu_id: int) -> str | None:
        handle = ctypes.c_void_p()
        if self._handle_by_index(ctypes.c_uint(gpu_id), ctypes.byref(handle)) != _NVML_SUCCESS:
            return None
        buffer = ctypes.create_string_buffer(_UUID_BUFFER_SIZE)
        if self._device_uuid(handle, buffer, _UUID_BUFFER_SIZE) != _NVML_SUCCESS:
            return None
        return buffer.value.decode("utf-8", errors="replace")


@cache
def _nvml_api() -> _NVMLAPI | None:
    """Load NVML once per process; None where it is absent or unusable."""
    for candidate in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
        try:
            return _NVMLAPI(ctypes.CDLL(candidate))
        except (AttributeError, OSError):
            continue
    return None


def device_uuid(gpu_id: int) -> str | None:
    """The UUID of a physical GPU, or None if NVML cannot name it."""
    api = _nvml_api()
    return None if api is None else api.device_uuid(gpu_id)


def uuid_key(value: str | None) -> str | None:
    """A UUID in comparable form: NVML spells it ``GPU-<hex>``, torch ``<hex>``."""
    if value is None:
        return None
    return value.strip().removeprefix("GPU-").lower() or None
