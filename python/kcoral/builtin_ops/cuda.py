"""Compiling CUDA C uploads through TVM FFI. Needs nvcc, a host C++ compiler,
and ninja; without them ``compile_cuda`` reports ``unavailable``."""

from __future__ import annotations

import functools
import os
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..deferred import DeferredGPUResult
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
    options = _validate_compile_request(src, cfg, "compile_cuda")
    cuda_cflags = _string_list(options, "extra_cuda_cflags")
    library_path = _build_cuda(src, cuda_cflags)
    # Building is host-only and may overlap another worker's benchmark. Loading
    # the shared object registers its CUDA fatbinary, so defer that small phase
    # until the engine has reacquired this GPU's lease.
    return DeferredGPUResult(lambda: _load_compiled_entry(library_path, src.entry))


@register_builtin("compile_cuda_binary", cpu_only=True)
def compile_cuda_binary(src: Any, cfg: Any = None) -> bytes:
    """Build a CUDA C upload for an explicit GPU architecture and return its
    shared-object bytes. cfg: ``arch`` and optional ``extra_cuda_cflags``."""
    options = _validate_compile_request(src, cfg, "compile_cuda_binary")
    arch = options.get("arch")
    if not isinstance(arch, str):
        raise ExecutionError("compile", "compile_cuda_binary option 'arch' must be a string")
    arch_list = _tvm_ffi_arch(arch)
    cuda_cflags = _string_list(options, "extra_cuda_cflags")
    library_path = _build_cuda(src, cuda_cflags, arch_list=arch_list)
    try:
        return Path(library_path).read_bytes()
    except OSError as exc:
        raise ExecutionError("compile", f"cannot read compiled CUDA library: {short(exc)}") from exc


# --- helpers ----------------------------------------------------------------


def _validate_compile_request(src: Any, cfg: Any, builtin: str) -> dict:
    if not isinstance(src, CUDASource):
        raise ExecutionError(
            "compile", f"{builtin} expects a module upload whose language is 'cuda'"
        )
    options = cfg if cfg is not None else {}
    if not isinstance(options, dict):
        raise ExecutionError("compile", f"{builtin} options must be a dict")
    return options


def _build_cuda(src: CUDASource, cuda_cflags: list[str], arch_list: str | None = None) -> str:
    _require_cuda_toolchain()
    import tvm_ffi.cpp

    # A GPU worker discovers its visible device once. A CPU worker temporarily
    # overrides the setting with the target supplied by the client.
    if arch_list is None and "TVM_FFI_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TVM_FFI_CUDA_ARCH_LIST"] = _cuda_arch_list()
    source_manages_exports = _declares_tvm_ffi_macro(src.source)
    try:
        with _cuda_arch_override(arch_list):
            return tvm_ffi.cpp.build_inline(
                name=f"upload_{src.entry}",
                cuda_sources=src.source,
                functions=None if source_manages_exports else src.entry,
                extra_cuda_cflags=cuda_cflags or None,
                backend="cuda",
            )
    except RuntimeError as exc:  # nvcc/ptxas diagnostics, surfaced by ninja
        raise ExecutionError("compile", short(_diagnostics(str(exc)))) from exc


@contextmanager
def _cuda_arch_override(arch_list: str | None) -> Iterator[None]:
    if arch_list is None:
        yield
        return
    key = "TVM_FFI_CUDA_ARCH_LIST"
    previous = os.environ.get(key)
    os.environ[key] = arch_list
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


_CUDA_ARCH = re.compile(r"sm_(\d+)(\d)(a?)")


def _tvm_ffi_arch(arch: str) -> str:
    """Convert a health target such as ``sm_100a`` to TVM FFI's ``10.0a``."""
    match = _CUDA_ARCH.fullmatch(arch)
    if match is None:
        raise ExecutionError("compile", "compile_cuda_binary option 'arch' must look like 'sm_90a'")
    major, minor, suffix = match.groups()
    return f"{int(major)}.{minor}{suffix}"


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


def _declares_tvm_ffi_macro(source: str) -> bool:
    """Whether the source manages its own TVM FFI export boundary."""
    return bool(re.search(r"(?m)^[ \t]*TVM_FFI_DLL_EXPORT_TYPED_FUNC[ \t]*\(", source))


def _compiled_entry(mod: Any, entry: str):
    if hasattr(mod, entry):
        return getattr(mod, entry)
    try:
        return mod.get_function(entry)
    except AttributeError as exc:
        raise ExecutionError("compile", f"compiled module has no exported entry {entry!r}") from exc


def _load_compiled_entry(library_path: str, entry: str):
    import tvm_ffi

    try:
        mod = tvm_ffi.load_module(library_path)
        return _compiled_entry(mod, entry)
    except ExecutionError:
        raise
    except Exception as exc:
        raise ExecutionError("compile", f"cannot load compiled CUDA module: {short(exc)}") from exc


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
