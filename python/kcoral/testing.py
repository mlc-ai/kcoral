"""GPU-free runtime used by the protocol and worker tests."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import ExecutionError
from .gpu_runtime import resolve_entry  # shared so the double cannot drift; touches no GPU


class _UnsharedGPU:
    """A lease over a GPU nothing else can reach, for calling ``execute`` with no
    worker around it. Acquiring and releasing are nothing."""

    held = False

    def acquire(self) -> None:
        pass

    def release(self) -> None:
        pass


UNSHARED_GPU = _UnsharedGPU()


class _Opaque:
    pass


@dataclass
class _FakeTensor:
    data: bytes
    dtype: str
    shape: list[int]


def _opaque(*_args: Any) -> _Opaque:
    return _Opaque()


def _structural(*_args: Any) -> dict[str, Any]:
    return {"ok": True, "values": [None, False, 7, 1.5, "text"]}


def _binary(*_args: Any) -> bytes:
    return b"binary-result"


def _pid(*_args: Any) -> int:
    return os.getpid()


def _crash(*_args: Any):
    os._exit(1)


def _crash_after_output(*_args: Any):
    """Exit the way a native fault does: a message on the descriptor, then gone
    before anything can send it back over the pipe."""
    os.write(2, b"fatal: simulated device-side assert\n")
    os._exit(1)


def _sleep(seconds: Any = 0.0, *_args: Any) -> dict[str, float]:
    time.sleep(float(seconds))
    return {"slept": float(seconds)}


_BUILTINS: dict[str, Callable] = {
    "builtin.opaque": _opaque,
    "builtin.structural": _structural,
    "builtin.binary": _binary,
    "builtin.crash": _crash,
    "builtin.crash_after_output": _crash_after_output,
    "builtin.sleep": _sleep,
    "builtin.cpu_sleep": _sleep,  # same work, declared not to need the GPU
}


class FakeRuntime:
    def __init__(self) -> None:
        self._poisoned = False
        self._last_error: str | None = None

    def load_module(self, source: str, entry: str | None = None, language: str = "python") -> Any:
        assert language == "python", "the fake runtime has no compiler"
        namespace: dict[str, Any] = {}
        try:
            exec(compile(source, "<uploaded>", "exec"), namespace)
        except SyntaxError as exc:
            raise ExecutionError("parse", str(exc)) from exc
        except Exception as exc:
            raise ExecutionError("parse", f"{type(exc).__name__}: {exc}") from exc
        return resolve_entry(namespace, source, entry)

    def target(self) -> dict[str, str]:
        return {"arch": "fake"}

    def versions(self) -> dict[str, str]:
        return {}

    def device_uuid(self) -> str | None:
        return None

    def load_library(self, data: bytes, entry: str | None = None) -> Any:
        raise ExecutionError("unavailable", "the fake runtime cannot load a library")

    def get_function(self, module: Any, name: str) -> Callable:
        try:
            fn = module[name] if isinstance(module, dict) else getattr(module, name)
        except (KeyError, AttributeError) as exc:
            raise ExecutionError("compile", f"the module exports no function {name!r}") from exc
        if not callable(fn):
            raise ExecutionError("runtime", f"module member {name!r} is not callable")
        return fn

    def load_tensor(self, data: bytes, dtype: str, shape: list[int]) -> _FakeTensor:
        return _FakeTensor(data=data, dtype=dtype, shape=shape)

    def export_tensor(self, value: Any) -> tuple[str, list[int], bytes] | None:
        if not isinstance(value, _FakeTensor):
            return None
        return value.dtype, value.shape, value.data

    def builtin(self, name: str) -> Callable:
        if name == "builtin.poison":
            return self._poison
        if name == "builtin.stale_cuda_error":
            return self._stale_cuda_error
        if name == "builtin.stale_cuda_error_unavailable":
            return self._stale_cuda_error_unavailable
        fn = _BUILTINS.get(name)
        if fn is None:
            raise ExecutionError("runtime", f"unknown function: {name!r}")
        return fn

    def cpu_only_builtins(self) -> frozenset[str]:
        return frozenset({"builtin.cpu_sleep"})

    def synchronize(self) -> None:
        pass

    def take_last_error(self) -> str | None:
        error = self._last_error
        self._last_error = None
        return error

    def reset(self) -> None:
        if self._poisoned:
            raise RuntimeError("simulated poisoned GPU context")

    def _poison(self) -> None:
        self._poisoned = True
        raise ExecutionError("runtime", "simulated illegal memory access")

    def _stale_cuda_error(self) -> None:
        self._last_error = "CUDA error cudaErrorInvalidValue (1): invalid argument"

    def _stale_cuda_error_unavailable(self) -> None:
        self._stale_cuda_error()
        raise ExecutionError("unavailable", "CUPTI recorded no GPU activity")


def fake_runtime_factory() -> FakeRuntime:
    return FakeRuntime()
