"""Native NVIDIA device identity and CUDA runtime error APIs."""

from __future__ import annotations

import ctypes
import ctypes.util
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
        self._device_minor_number = library.nvmlDeviceGetMinorNumber

    def device_minor_number(self, gpu_id: int) -> int | None:
        handle = ctypes.c_void_p()
        if self._handle_by_index(ctypes.c_uint(gpu_id), ctypes.byref(handle)) != _NVML_SUCCESS:
            return None
        minor = ctypes.c_uint()
        if self._device_minor_number(handle, ctypes.byref(minor)) != _NVML_SUCCESS:
            return None
        return minor.value

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


def device_minor_number(gpu_id: int) -> int | None:
    """Resolve an NVML index to the Linux /dev/nvidiaN device number.

    NVIDIA containers can expose index 0 while retaining a host device node
    such as /dev/nvidia6. UUID selection and filesystem mounts must identify
    the same card even when these numbers differ.
    """
    api = _nvml_api()
    return None if api is None else api.device_minor_number(gpu_id)


def uuid_key(value: str | None) -> str | None:
    """A UUID in comparable form: NVML spells it ``GPU-<hex>``, torch ``<hex>``."""
    if value is None:
        return None
    return value.strip().removeprefix("GPU-").lower() or None


class _CUDAErrorAPI:
    """The small libcudart surface needed to inspect CUDA's last-error slot."""

    def __init__(self, library: Any) -> None:
        self._get_last_error = library.cudaGetLastError
        self._get_last_error.argtypes = []
        self._get_last_error.restype = ctypes.c_int
        self._get_error_name = library.cudaGetErrorName
        self._get_error_name.argtypes = [ctypes.c_int]
        self._get_error_name.restype = ctypes.c_char_p
        self._get_error_string = library.cudaGetErrorString
        self._get_error_string.argtypes = [ctypes.c_int]
        self._get_error_string.restype = ctypes.c_char_p

    def take_last_error(self) -> str | None:
        code = self._get_last_error()
        if code == 0:
            return None
        name = _decode_cuda_error(self._get_error_name(code), "cudaErrorUnknown")
        description = _decode_cuda_error(self._get_error_string(code), "unknown error")
        return f"CUDA error {name} ({code}): {description}"


def _decode_cuda_error(value: bytes | None, fallback: str) -> str:
    return value.decode("utf-8", errors="replace") if value is not None else fallback


@cache
def _cuda_error_api() -> _CUDAErrorAPI:
    """Load the same CUDA runtime torch uses, once per worker process."""
    candidates: list[str] = []
    discovered = ctypes.util.find_library("cudart")
    if discovered is not None:
        candidates.append(discovered)
    try:
        import torch

        if torch.version.cuda:
            candidates.append(f"libcudart.so.{torch.version.cuda.split('.', 1)[0]}")
    except Exception:
        pass
    candidates.extend(["libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0"])

    failures: list[str] = []
    for candidate in dict.fromkeys(candidates):
        try:
            return _CUDAErrorAPI(ctypes.CDLL(candidate))
        except (AttributeError, OSError) as exc:
            failures.append(f"{candidate}: {exc}")
    raise RuntimeError("could not load libcudart to inspect CUDA errors: " + "; ".join(failures))
