"""Compiling CUDA C uploads through TVM FFI. Needs nvcc, a host C++ compiler,
and ninja; without them ``compile_cuda`` reports ``unavailable``."""

from __future__ import annotations

import functools
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any

from ..errors import ExecutionError
from ._common import short
from ._registry import register_builtin


@dataclass(frozen=True)
class CUDASource:
    """CUDA C source text and the name of the function it exports."""

    source: str
    entry: str


@register_builtin("compile_cuda", cpu_only=True)
def compile_cuda(src: Any, cfg: Any = None) -> Any:
    """Build a CUDA C upload into its exported function, caching the build on
    disk. cfg: ``extra_cuda_cflags``."""
    if not isinstance(src, CUDASource):
        raise ExecutionError(
            "compile", "compile_cuda expects a module upload whose language is 'cuda'"
        )
    options = cfg if cfg is not None else {}
    if not isinstance(options, dict):
        raise ExecutionError("compile", "compile_cuda options must be a dict")
    cuda_cflags = _string_list(options, "extra_cuda_cflags")
    _require_cuda_toolchain()
    import tvm_ffi.cpp

    # Selects the GPU's arch-specific target unless the operator already pinned one.
    os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", _cuda_arch_list())
    try:
        mod = tvm_ffi.cpp.load_inline(
            name=f"upload_{src.entry}",
            cuda_sources=src.source,
            functions=src.entry,
            extra_cuda_cflags=cuda_cflags or None,
        )
    except RuntimeError as exc:  # nvcc/ptxas diagnostics, surfaced by ninja
        raise ExecutionError("compile", short(_diagnostics(str(exc)))) from exc
    return getattr(mod, src.entry)


# --- helpers ----------------------------------------------------------------

# nvcc writes "file(9): error: ...", ptxas "..., line 9; error   : ...", gcc
# "file:9:1: error: ...".
_DIAGNOSTIC = re.compile(r"\berror\b\s*:", re.IGNORECASE)


def _diagnostics(text: str) -> str:
    """Compiler output from its first diagnostic on, dropping the ninja invocation
    ahead of it so the truncation budget is spent on errors, not command lines."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _DIAGNOSTIC.search(line):
            return "\n".join(lines[index:])
    return text


def _require_cuda_toolchain() -> None:
    """Report ``unavailable`` for a missing build program, rather than letting a
    bare FileNotFoundError surface. Lookups mirror the ones ``tvm_ffi.cpp`` makes."""
    try:
        import tvm_ffi.cpp  # noqa: F401
    except ImportError as exc:
        raise ExecutionError(
            "unavailable", "server-side CUDA C compilation requires tvm_ffi.cpp"
        ) from exc
    missing = [tool for tool in ("ninja", os.environ.get("CXX", "c++")) if not shutil.which(tool)]
    if not _has_nvcc():
        missing.append("nvcc")
    if missing:
        raise ExecutionError(
            "unavailable",
            f"server-side CUDA C compilation needs {', '.join(missing)}, "
            "which this server does not have",
        )


def _has_nvcc() -> bool:
    if shutil.which("nvcc"):
        return True
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    return os.path.exists(os.path.join(cuda_home, "bin", "nvcc"))


@functools.lru_cache(maxsize=1)
def _cuda_arch_list() -> str:
    """``TVM_FFI_CUDA_ARCH_LIST`` value for the visible GPU. Hopper onward gets the
    arch-specific target (``sm_100a``), without which tcgen05 and wgmma fail to build."""
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}a" if major >= 9 else f"{major}.{minor}"


def _string_list(options: dict, key: str) -> list[str]:
    value = options.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ExecutionError("compile", f"{key!r} must be a list of strings")
    return value
