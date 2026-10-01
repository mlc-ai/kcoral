"""CPU runtime for compiling CUDA source without initializing a GPU."""

from __future__ import annotations

import gc
import importlib.metadata
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from . import process_state
from .errors import ExecutionError
from .python_module import LoadedPythonModule, materialize_module


class CPURuntime:
    """Runtime exposed by ``--device cpu`` workers.

    CPU workers accept source files and can return a compiled shared object.
    Uploaded Python supplies the compilation harness. GPU tensors and uploaded
    libraries remain unavailable; constructing this runtime imports no GPU library.
    """

    def __init__(self) -> None:
        self._seeded_fnames: list[str] = []
        self._process_state = process_state.snapshot()

    def load_module(self, source: str) -> Any:
        return materialize_module(source, self._seeded_fnames)

    def load_library(self, data: bytes) -> Any:
        raise ExecutionError("unavailable", "library uploads require a GPU worker")

    def get_function(self, module: Any, name: str) -> Any:
        if isinstance(module, LoadedPythonModule):
            return module.get_function(name)
        raise ExecutionError("unavailable", "get_function expects an uploaded Python module")

    def target(self) -> dict[str, str]:
        """CPU workers compile for the architecture supplied by the client."""
        return {}

    def versions(self) -> dict[str, str]:
        versions: dict[str, str] = {}
        try:
            versions["tvm_ffi"] = importlib.metadata.version("apache-tvm-ffi")
        except importlib.metadata.PackageNotFoundError:
            pass
        return versions

    def device_uuid(self) -> str | None:
        return None

    def load_tensor(self, data: bytes, dtype: str, shape: list[int]) -> Any:
        raise ExecutionError("unavailable", "tensor uploads require a GPU worker")

    def export_tensor(self, value: Any) -> tuple[str, list[int], bytes] | None:
        return None

    def forbid_gpu(self) -> AbstractContextManager[None]:
        return nullcontext()  # there is no GPU here to touch

    def synchronize(self) -> None:
        pass

    def take_last_error(self) -> str | None:
        return None

    def reset(self) -> None:
        import linecache

        for fname in self._seeded_fnames:
            linecache.cache.pop(fname, None)
        self._seeded_fnames.clear()
        gc.collect()
        process_state.restore(self._process_state)


def cpu_runtime_factory() -> CPURuntime:
    """Picklable factory used by spawned CPU workers."""
    return CPURuntime()
