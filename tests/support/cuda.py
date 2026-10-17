"""Compiling CUDA C uploads through TVM FFI. Needs nvcc, a host C++ compiler,
and ninja; without them ``compile_cuda`` reports ``unavailable``. On ROCm
workers the source is HIP C++, built with ROCm's clang for the ``gfx*`` target."""

from __future__ import annotations

import functools
import os
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from kcoral.errors import ExecutionError
from kcoral.support import vendor

from ._common import short


def compile_cuda(src: Any, cfg: Any = None) -> Any:
    """Build a CUDA C upload into its exported function, caching the build on
    disk. cfg: ``functions`` and optional ``extra_cuda_cflags``."""
    options = _validate_compile_request(src, cfg, "compile_cuda")
    cuda_cflags = _string_list(options, "extra_cuda_cflags")
    library_path = _build_cuda(src, options["functions"], cuda_cflags)
    # This function includes CUDA module loading, so callers must hold the lease.
    return _load_compiled_function(library_path, options["functions"][0])


def compile_cuda_binary(src: Any, cfg: Any = None) -> bytes:
    """Build a CUDA C upload for an explicit GPU architecture and return its
    shared-object bytes. cfg: ``functions``, ``arch`` and optional ``extra_cuda_cflags``."""
    options = _validate_compile_request(src, cfg, "compile_cuda_binary")
    arch = options.get("arch")
    if not isinstance(arch, str):
        raise ExecutionError("compile", "compile_cuda_binary option 'arch' must be a string")
    arch_list = _tvm_ffi_arch(arch)  # validated against the worker's backend
    cuda_cflags = _string_list(options, "extra_cuda_cflags")
    library_path = _build_cuda(src, options["functions"], cuda_cflags, arch_list=arch_list)
    try:
        return Path(library_path).read_bytes()
    except OSError as exc:
        raise ExecutionError("compile", f"cannot read compiled CUDA library: {short(exc)}") from exc


# --- helpers ----------------------------------------------------------------


def _validate_compile_request(src: Any, cfg: Any, harness: str) -> dict:
    if not isinstance(src, str):
        raise ExecutionError("compile", f"{harness} expects a source file path")
    options = cfg if cfg is not None else {}
    if not isinstance(options, dict):
        raise ExecutionError("compile", f"{harness} options must be a dict")
    functions = _string_list(options, "functions")
    if not functions:
        raise ExecutionError("compile", f"{harness} requires exported function names")
    return options


def _build_cuda(
    source_path: str, functions: list[str], cuda_cflags: list[str], arch_list: str | None = None
) -> str:
    _require_cuda_toolchain()
    import tvm_ffi.cpp

    # A GPU worker discovers its visible device once. A CPU worker temporarily
    # overrides the setting with the target supplied by the client.
    if arch_list is None and _arch_variable() not in os.environ:
        os.environ[_arch_variable()] = _cuda_arch_list()
    try:
        source = Path(source_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ExecutionError("compile", f"cannot read CUDA source: {short(exc)}") from exc
    source_manages_exports = _declares_tvm_ffi_macro(source)
    try:
        with _cuda_arch_override(arch_list):
            return tvm_ffi.cpp.build_inline(
                name="upload_" + "_".join(functions),
                cuda_sources=_portable_source(source),
                functions=None if source_manages_exports else functions,
                extra_cuda_cflags=cuda_cflags or None,
                backend=_backend(),
            )
    except RuntimeError as exc:  # nvcc/ptxas or clang diagnostics, surfaced by ninja
        raise ExecutionError("compile", short(_diagnostics(str(exc)))) from exc


@contextmanager
def _cuda_arch_override(arch_list: str | None) -> Iterator[None]:
    if arch_list is None:
        yield
        return
    key = _arch_variable()
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
_ROCM_ARCH = re.compile(r"gfx[0-9a-f]+")


def _portable_source(source: str) -> str:
    """HIP C++ needs the runtime header that nvcc includes implicitly; it defines
    ``blockIdx`` and the ``<<<...>>>`` launch. Also locate ROCm for the link."""
    if _backend() != "hip":
        return source
    _set_rocm_home()
    if "hip/hip_runtime.h" in source:
        return source
    return "#include <hip/hip_runtime.h>\n" + source


def _set_rocm_home() -> None:
    """TVM FFI links ``-lamdhip64`` from ROCM_HOME, else from hipcc's parent
    directory. With ROCm installed as Python wheels (``rocm-sdk``), hipcc is a
    launcher in /usr/local/bin and the libraries live in ``_rocm_sdk_devel``."""
    if os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or os.path.isdir("/opt/rocm"):
        return
    try:
        import _rocm_sdk_devel
    except ImportError:
        return
    os.environ["ROCM_HOME"] = os.path.dirname(_rocm_sdk_devel.__file__)


def _backend() -> str:
    """TVM FFI's name for the worker's GPU compiler backend."""
    return "hip" if vendor.backend() == "rocm" else "cuda"


def _arch_variable() -> str:
    return "TVM_FFI_ROCM_ARCH_LIST" if _backend() == "hip" else "TVM_FFI_CUDA_ARCH_LIST"


def _tvm_ffi_arch(arch: str) -> str:
    """Convert a health target such as ``sm_100a`` to TVM FFI's ``10.0a``; a ROCm
    target such as ``gfx942`` is already in TVM FFI's spelling."""
    if _backend() == "hip":
        if _ROCM_ARCH.fullmatch(arch) is None:
            raise ExecutionError(
                "compile", "compile_cuda_binary option 'arch' must look like 'gfx942'"
            )
        return arch
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


def _compiled_function(mod: Any, name: str):
    if hasattr(mod, name):
        return getattr(mod, name)
    try:
        return mod.get_function(name)
    except AttributeError as exc:
        raise ExecutionError(
            "compile", f"compiled module has no exported function {name!r}"
        ) from exc


def _load_compiled_function(library_path: str, name: str):
    import tvm_ffi

    try:
        mod = tvm_ffi.load_module(library_path)
        return _compiled_function(mod, name)
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
    if _backend() == "hip":
        if not _has_hip_compiler():
            missing.append("ROCm clang (hipcc)")
    elif not _has_nvcc():
        missing.append("nvcc")
    if missing:
        language = "HIP C++" if _backend() == "hip" else "CUDA C"
        raise ExecutionError(
            "unavailable",
            f"server-side {language} compilation needs {', '.join(missing)}, "
            "which this server does not have",
        )


def _has_nvcc() -> bool:
    if shutil.which("nvcc"):
        return True
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    return os.path.exists(os.path.join(cuda_home, "bin", "nvcc"))


def _has_hip_compiler() -> bool:
    """TVM FFI finds ROCm's clang through hipcc, ROCM_HOME/ROCM_PATH, or
    /opt/rocm; ROCm PyTorch wheels may also carry it in ``_rocm_sdk_devel``."""
    if shutil.which("hipcc"):
        return True
    roots = [os.environ.get("ROCM_HOME"), os.environ.get("ROCM_PATH"), "/opt/rocm"]
    if any(root and os.path.exists(os.path.join(root, "llvm", "bin", "clang++")) for root in roots):
        return True
    try:
        import _rocm_sdk_devel  # noqa: F401
    except ImportError:
        return False
    return True


@functools.lru_cache(maxsize=1)
def _cuda_arch_list() -> str:
    """``TVM_FFI_CUDA_ARCH_LIST`` value for the visible GPU. Hopper onward gets the
    arch-specific target (``sm_100a``), without which tcgen05 and wgmma fail to build.
    On ROCm, the ``TVM_FFI_ROCM_ARCH_LIST`` value: the ``gfx*`` architecture."""
    import torch

    if _backend() == "hip":
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]

    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}a" if major >= 9 else f"{major}.{minor}"


def _string_list(options: dict, key: str) -> list[str]:
    value = options.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ExecutionError("compile", f"{key!r} must be a list of strings")
    return value
