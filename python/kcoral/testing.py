"""GPU-free runtime used by the protocol and worker tests."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .engine import Runtime, execute
from .errors import ExecutionError, GPUAccessViolation
from .lease import Lease
from .python_module import LoadedPythonModule
from .schemas import Program, ProgramOutcome


def execute_for_test(
    program: Program,
    runtime: Runtime,
    lease: Lease,
    **kwargs: Any,
) -> ProgramOutcome:
    """Own a temporary workspace for an engine test running without a worker."""
    with tempfile.TemporaryDirectory(prefix="kcoral-test-") as workspace_dir:
        return execute(program, runtime, lease, workspace_dir=workspace_dir, **kwargs)


class _UnsharedGPU:
    """A lease over a GPU nothing else can reach, for calling ``execute`` with no
    worker around it. Acquiring and releasing are nothing."""

    held = False

    def acquire(self) -> None:
        pass

    def release(self) -> None:
        pass


UNSHARED_GPU = _UnsharedGPU()


@dataclass
class _FakeTensor:
    data: bytes
    dtype: str
    shape: list[int]


_watched_cuda_calls: list[str] | None = None  # recorded while a forbid_gpu guard is up


def simulate_cuda_call(name: str = "cudaMalloc") -> None:
    """What uploaded test source calls in place of a CUDA API call."""
    if _watched_cuda_calls is not None:
        _watched_cuda_calls.append(name)


class FakeRuntime:
    def __init__(self) -> None:
        self._poisoned = False
        self._last_error: str | None = None

    def load_module(self, source: str) -> Any:
        namespace: dict[str, Any] = {"_test_runtime": self}
        try:
            exec(compile(source, "<uploaded>", "exec"), namespace)
        except SyntaxError as exc:
            raise ExecutionError("parse", str(exc)) from exc
        except Exception as exc:
            raise ExecutionError("parse", f"{type(exc).__name__}: {exc}") from exc
        return LoadedPythonModule(namespace=namespace)

    def target(self) -> dict[str, str]:
        return {"arch": "fake"}

    def versions(self) -> dict[str, str]:
        return {}

    def device_uuid(self) -> str | None:
        return None

    def load_library(self, data: bytes) -> Any:
        raise ExecutionError("unavailable", "the fake runtime cannot load a library")

    def get_function(self, module: Any, name: str) -> Any:
        if isinstance(module, LoadedPythonModule):
            try:
                return module.namespace[name]
            except KeyError:
                raise ExecutionError(
                    "parse", f"the uploaded Python module defines no name {name!r}"
                ) from None
        try:
            fn = module[name] if isinstance(module, dict) else getattr(module, name)
        except (KeyError, AttributeError) as exc:
            raise ExecutionError("compile", f"the module exports no function {name!r}") from exc
        return fn

    def load_tensor(self, data: bytes, dtype: str, shape: list[int]) -> _FakeTensor:
        return _FakeTensor(data=data, dtype=dtype, shape=shape)

    def export_tensor(self, value: Any) -> tuple[str, list[int], bytes] | None:
        if not isinstance(value, _FakeTensor):
            return None
        return value.dtype, value.shape, value.data

    @contextmanager
    def forbid_gpu(self) -> Iterator[None]:
        global _watched_cuda_calls
        _watched_cuda_calls = calls = []
        try:
            yield
        finally:
            _watched_cuda_calls = None
            if calls:
                raise GPUAccessViolation(calls[0], "<uploaded>:1 in main", "", time.monotonic_ns())

    def synchronize(self) -> None:
        pass

    def prepare_to_release_gpu(self) -> None:
        self.synchronize()

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
