from __future__ import annotations

import ctypes
from typing import Any


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_Deleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _Deleter),
]

_OWNERS: dict[int, _RawTensorProducer] = {}


@_Deleter
def _release(managed: ctypes.POINTER(_DLManagedTensor)) -> None:
    _OWNERS.pop(ctypes.addressof(managed.contents), None)


_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_GetPointer.restype = ctypes.c_void_p


class _RawTensorProducer:
    def __init__(self, data: bytes, dtype: Any, shape: tuple[int, ...]) -> None:
        self._backing = bytearray(data) if data else bytearray(1)
        self._buffer = (ctypes.c_ubyte * len(self._backing)).from_buffer(self._backing)
        self._shape = (ctypes.c_int64 * len(shape))(*shape) if shape else None
        shape_pointer = (
            ctypes.cast(self._shape, ctypes.POINTER(ctypes.c_int64))
            if self._shape is not None
            else None
        )
        self._managed = _DLManagedTensor(
            _DLTensor(
                ctypes.addressof(self._buffer),
                _DLDevice(1, 0),
                len(shape),
                _DLDataType(int(dtype.type_code), int(dtype.bits), int(dtype.lanes)),
                shape_pointer,
                None,
                0,
            ),
            None,
            _release,
        )
        _OWNERS[ctypes.addressof(self._managed)] = self

    def __dlpack_device__(self) -> tuple[int, int]:
        return (1, 0)

    def __dlpack__(self, stream: int | None = None) -> Any:
        del stream
        return _PyCapsule_New(ctypes.addressof(self._managed), b"dltensor", None)


def tensor_from_bytes(data: bytes, dtype_name: str, shape: tuple[int, ...]) -> Any:
    import tvm_ffi

    dtype = tvm_ffi.dtype(dtype_name)
    producer = _RawTensorProducer(data, dtype, shape)
    try:
        return tvm_ffi.from_dlpack(producer, require_contiguous=True)
    except Exception:
        _OWNERS.pop(ctypes.addressof(producer._managed), None)
        raise


def cpu_tensor_bytes(tensor: Any, size: int) -> bytes:
    capsule = tensor.__dlpack__()
    pointer = _PyCapsule_GetPointer(capsule, b"dltensor")
    if not pointer:
        raise ValueError("DLPack producer returned an invalid capsule")
    managed = ctypes.cast(pointer, ctypes.POINTER(_DLManagedTensor)).contents
    dl_tensor = managed.dl_tensor
    if dl_tensor.device.device_type != 1:
        raise ValueError("tensor is not in CPU memory")
    return ctypes.string_at(int(dl_tensor.data) + int(dl_tensor.byte_offset), size)
