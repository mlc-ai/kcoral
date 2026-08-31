"""CPU runtime for compiling CUDA source without initializing a GPU."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from typing import Any

from . import builtin_ops
from .errors import ExecutionError

_COMPILE_BUILTIN = "builtin.compile_cuda_binary"


class CPURuntime:
    """Runtime exposed by ``--device cpu`` workers.

    CPU workers accept CUDA source and can return a compiled shared object. GPU
    tensors, uploaded libraries, Python modules, and execution builtins remain
    unavailable so constructing this runtime never imports or initializes a GPU
    library.
    """

    def __init__(self) -> None:
        self._builtins = builtin_ops.snapshot_registry()

    def load_module(self, source: str, language: str = "python") -> Any:
        if language == "cuda":
            return builtin_ops.CUDAModule(source=source)
        raise ExecutionError("unavailable", "Python modules are unavailable in CPU mode")

    def load_library(self, data: bytes) -> Any:
        raise ExecutionError("unavailable", "library uploads require a GPU worker")

    def get_function(self, module: Any, name: str) -> Any:
        if isinstance(module, builtin_ops.CUDAModule):
            return module.get_function(name)
        raise ExecutionError("unavailable", "only CUDA source modules are available in CPU mode")

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

    def builtin(self, name: str) -> Callable:
        fn = builtin_ops.resolve(name)
        if fn is None:
            raise ExecutionError("runtime", f"unknown function: {name!r}")
        if name != _COMPILE_BUILTIN:
            raise ExecutionError("unavailable", f"{name} is unavailable in CPU mode")
        return fn

    def cpu_only_builtins(self) -> frozenset[str]:
        return frozenset({_COMPILE_BUILTIN})

    def synchronize(self) -> None:
        pass

    def take_last_error(self) -> str | None:
        return None

    def reset(self) -> None:
        builtin_ops.restore_registry(self._builtins)


def cpu_runtime_factory() -> CPURuntime:
    """Picklable factory used by spawned CPU workers."""
    return CPURuntime()
