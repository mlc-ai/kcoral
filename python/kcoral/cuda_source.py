"""CUDA source text and function names selected by get_function."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ExecutionError


@dataclass(frozen=True)
class CUDAModule:
    """Uploaded CUDA C source, optionally with a selected exported function."""

    source: str
    name: str | None = None

    def get_function(self, name: str) -> CUDAModule:
        if not name.isidentifier():
            raise ExecutionError("parse", "a CUDA function name must be an identifier")
        if name == "main":
            raise ExecutionError("parse", "C++ reserves 'main'; name the function otherwise")
        return CUDAModule(source=self.source, name=name)
