"""The GPU runtime: materialize uploads, resolve builtins.

The one :class:`Runtime` that touches a GPU; it runs inside the worker process
and has no compiler of its own — compiling and running kernels are builtins (see
:mod:`kcoral.builtin_ops`). torch is imported lazily, so importing this
module touches no GPU.
"""

from __future__ import annotations

import ast
import ctypes
import ctypes.util
import gc
import hashlib
import importlib.metadata
import linecache
import os
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from . import builtin_ops, process_state
from .errors import ExecutionError

# The name an uploaded module's entry object takes when the upload names none.
ENTRY_POINT = "main"

# Libraries already dlopened by this worker, keyed by the SHA-256 of their bytes.
# Purely a memoization — every request carries the bytes it needs. Bounded because
# each entry holds a loaded GPU module; an evicted one stays mapped for as long as
# the functions it produced are reachable.
_LOADED_LIBRARIES: OrderedDict[str, Any] = OrderedDict()
_LOADED_LIBRARIES_LIMIT = 32
_LIBRARY_DIR: Path | None = None
_LOADERS_READY = False


@dataclass(frozen=True)
class LoadedLibrary:
    """A request-local view of one cached TVM-FFI module."""

    digest: str
    module: Any


@dataclass(frozen=True)
class LoadedFunction:
    """A callable that keeps its defining module alive and propagates Torch's stream."""

    owner: LoadedLibrary
    function: Callable

    def __call__(self, *args: Any) -> Any:
        import tvm_ffi

        # Tensor conversion also propagates a stream, but a launcher may accept
        # only scalar handles and still call TVMFFIEnvGetStream.
        with tvm_ffi.use_torch_stream():
            return self.function(*args)


class GPURuntime:
    """Construct inside the worker after its GPU is pinned. Building it checks that
    torch and tvm_ffi import (the mandatory deps), so a broken environment fails at
    startup, not mid-run. Full tvm is optional — only ``compile_tirx`` needs it, and
    it fails gracefully when tvm is absent."""

    def __init__(self) -> None:
        _require_torch_and_ffi()
        self._seeded_fnames: list[str] = []  # linecache keys to clear on reset
        _warm_up()
        self._builtins = builtin_ops.snapshot_registry()
        self._process_state = process_state.snapshot()
        self._request_libraries: list[LoadedLibrary] = []

    def load_module(self, source: str, entry: str | None = None, language: str = "python") -> Any:
        if language == "cuda":
            # Nothing runs here: `builtin.compile_cuda` turns the text into a module.
            assert entry is not None
            return builtin_ops.CUDASource(source=source, entry=entry)
        return self._materialize_module(source, entry)

    def load_library(self, data: bytes, entry: str | None = None) -> Any:
        library = _materialize_library(data)
        # `env` is cleared before reset, but objects returned by the DSO may be
        # collected through cycles. Keep its code mapped through that collection.
        self._request_libraries.append(library)
        return library if entry is None else self.get_function(library, entry)

    def get_function(self, module: Any, name: str) -> LoadedFunction:
        if not isinstance(module, LoadedLibrary):
            raise ExecutionError("runtime", "get_function expects a library module handle")
        try:
            exists = module.module.implements_function(name)
        except Exception as exc:
            raise ExecutionError("runtime", f"cannot inspect the uploaded library: {exc}") from exc
        if not exists:
            raise ExecutionError("compile", f"the uploaded library exports no function {name!r}")
        try:
            function = module.module.get_function(name)
        except Exception as exc:
            raise ExecutionError(
                "compile", f"cannot bind function {name!r} from the uploaded library: {exc}"
            ) from exc
        return LoadedFunction(owner=module, function=function)

    def target(self) -> dict[str, str]:
        """What a client must compile a library for."""
        return describe_target()

    def versions(self) -> dict[str, str]:
        """The worker's library versions, for a client comparing its own."""
        return describe_versions()

    def device_uuid(self) -> str | None:
        """The card this worker actually got, for the parent to check."""
        return describe_device_uuid()

    def load_tensor(self, data: bytes, dtype: str, shape: list[int]) -> Any:
        return _materialize_tensor(data, dtype, shape)

    def export_tensor(self, value: Any) -> tuple[str, list[int], bytes] | None:
        import torch

        if not isinstance(value, torch.Tensor):
            return None
        tensor = value.detach().cpu().contiguous()
        dtype = str(tensor.dtype).removeprefix("torch.")
        shape = [int(dimension) for dimension in tensor.shape]
        data = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return dtype, shape, data

    def builtin(self, name: str) -> Callable:
        fn = builtin_ops.resolve(name)
        if fn is None:
            raise ExecutionError("runtime", f"unknown function: {name!r}")
        return fn

    def cpu_only_builtins(self) -> frozenset[str]:
        return builtin_ops.cpu_only_builtins()

    def synchronize(self) -> None:
        """Drain the GPU, so no kernel of this request is still running when the
        lease is given up and another worker starts measuring."""
        import torch

        # Do not hide a poisoned context. The engine preserves the request's
        # instruction error while the parent replaces this worker process.
        torch.cuda.synchronize()

    def take_last_error(self) -> str | None:
        """Consume CUDA's thread-local last error, if one is pending.

        Some invalid launches do not enqueue work and therefore are not reported
        by ``synchronize``. Leaving that error behind makes an unrelated CUDA API
        call in the next request report it instead.
        """
        return _cuda_error_api().take_last_error()

    def reset(self) -> None:
        import torch

        builtin_ops.restore_registry(self._builtins)
        process_state.restore(self._process_state)
        for fname in self._seeded_fnames:
            linecache.cache.pop(fname, None)
        self._seeded_fnames.clear()
        # An uploaded module's namespace is a reference cycle (its functions'
        # `__globals__` point back at it), so only a collection frees the
        # module-scope tensors `empty_cache()` would otherwise find still live.
        gc.collect()
        self._request_libraries.clear()
        # CUDA errors are sticky within a process. Surface one so the parent can
        # replace this worker instead of returning its context to the pool.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _materialize_module(self, source: str, entry: str | None) -> Any:
        # A kernel is re-read from its source text at compile time, so seed
        # linecache. Key by content hash so two functions in one program don't
        # overwrite each other's source.
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
        fname = f"<uploaded:{digest}>"
        linecache.cache[fname] = (len(source), None, source.splitlines(True), fname)
        self._seeded_fnames.append(fname)
        ns: dict = {}
        try:
            # Trusted only because the worker is a GPU-pinned, crash-isolated process.
            exec(compile(source, fname, "exec"), ns)
        except SyntaxError as exc:
            raise ExecutionError("parse", f"syntax error: {exc}") from exc
        except Exception as exc:
            raise ExecutionError("parse", f"{type(exc).__name__}: {exc}") from exc
        return resolve_entry(ns, source, entry)


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


def resolve_entry(namespace: dict, source: str, entry: str | None) -> Any:
    """Pick the entry object out of an uploaded module's executed namespace.

    An explicit ``entry`` wins, then ``main``, then the sole top-level definition.
    Several definitions and no ``main`` is ambiguous, so the error names the
    candidates instead of guessing. The result is deliberately not checked for
    callability: a decorator may bind a handle a builtin consumes rather than one
    ``run`` calls.
    """
    if entry is not None:
        try:
            return namespace[entry]
        except KeyError:
            raise ExecutionError("parse", f"source does not define {entry!r}") from None
    if ENTRY_POINT in namespace:
        return namespace[ENTRY_POINT]
    candidates = [name for name in _top_level_definitions(source) if name in namespace]
    if len(candidates) == 1:
        return namespace[candidates[0]]
    if not candidates:
        raise ExecutionError("parse", "source defines no top-level function or class")
    raise ExecutionError(
        "parse",
        f"source defines top-level names {', '.join(repr(name) for name in candidates)}; "
        f"name one {ENTRY_POINT!r} or set 'entry' on the upload",
    )


def _top_level_definitions(source: str) -> list[str]:
    """Names bound by a top-level ``def``/``async def``/``class``, in source order.

    Imports and assignments are excluded, so a module-level constant beside one
    kernel does not make the entry ambiguous.
    """
    names: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name not in names:
                names.append(node.name)
    return names


def _warm_up() -> None:
    """Pay the once-per-worker costs here rather than inside the first request,
    which would hold the GPU throughout: creating the CUDA context, and importing
    tvm and the CuTe DSL runtime, all cost more than a request should. Best-effort,
    since the optional deps may be absent."""
    try:
        import torch

        torch.cuda.synchronize()  # creates this process's CUDA context
    except Exception:
        pass
    _register_library_loaders()  # imports tvm and preloads the CuTe DSL runtime
    try:
        # tvm reuses the context torch just made, but still opens its own CUDA
        # device API on first use.
        import tvm

        tvm.runtime.empty((1,), "float32", tvm.cuda(0))
    except Exception:
        pass  # tvm is optional
    # The imports above live as long as the worker; freezing them out of the
    # collector's reach is what keeps the `gc.collect()` in `reset()` cheap.
    gc.collect()
    gc.freeze()


def _require_torch_and_ffi() -> None:
    try:
        import torch  # noqa: F401
        import tvm_ffi  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment misconfiguration
        raise RuntimeError(
            "The GPU runtime requires torch and tvm_ffi in the worker environment "
            f"(full tvm is optional). Importing them failed: {exc!r}"
        ) from exc


def _materialize_tensor(data: bytes, dtype_name: str, shape: list[int]) -> Any:
    import torch

    try:
        dtype = builtin_ops.torch_dtype(dtype_name)
        if not data:
            return torch.empty(shape, dtype=dtype, device="cuda")
        return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape).to("cuda")
    except ExecutionError:
        raise
    except Exception as exc:
        raise ExecutionError("runtime", f"malformed tensor: {exc}") from exc


def describe_target() -> dict[str, str]:
    """The compilation target of the visible GPU."""
    import torch

    major, minor = torch.cuda.get_device_capability()
    # Cache the compiler spelling while worker startup is serialized. A later
    # cpu_only compile must not query the CUDA driver after dropping its lease.
    os.environ.setdefault(
        "TVM_FFI_CUDA_ARCH_LIST", f"{major}.{minor}a" if major >= 9 else f"{major}.{minor}"
    )
    return {"arch": f"sm_{major}{minor}a" if major >= 9 else f"sm_{major}{minor}"}


def describe_versions() -> dict[str, str]:
    """Versions a client may want to match; optional dependencies are absent."""
    import torch

    versions = {"torch": torch.__version__}
    if torch.version.cuda:
        versions["cuda"] = torch.version.cuda
    for name in ("tvm", "tvm_ffi", "triton"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # optional, or no version attribute
            pass
    # By distribution metadata rather than import: nothing here calls these, and
    # importing flashinfer costs about a second.
    optional = (("cutlass", "nvidia-cutlass-dsl"), ("flashinfer", "flashinfer-python"))
    for key, distribution in optional:
        try:
            versions[key] = importlib.metadata.version(distribution)
        except Exception:  # not installed
            pass
    return versions


def describe_device_uuid() -> str | None:
    """The visible GPU's UUID; None where torch cannot name one."""
    import torch

    try:
        return str(torch.cuda.get_device_properties(0).uuid)
    except Exception:  # no such attribute on older torch, or no visible device
        return None


def _materialize_library(data: bytes) -> LoadedLibrary:
    """Load a prebuilt shared object as a module.

    Loading needs a path, so the bytes go to a file unlinked once dlopen maps it.
    Function binding is separate so one upload can expose multiple entry points.
    """
    import tvm_ffi

    _register_library_loaders()
    digest = hashlib.sha256(data).hexdigest()
    cached = _LOADED_LIBRARIES.get(digest)
    if cached is not None:
        _LOADED_LIBRARIES.move_to_end(digest)
    else:
        path = _library_dir() / f"{digest}.so"
        path.write_bytes(data)
        try:
            cached = tvm_ffi.load_module(str(path))
        except Exception as exc:
            raise ExecutionError("compile", f"cannot load the uploaded library: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)  # the mapping outlives the file
        _LOADED_LIBRARIES[digest] = cached
        if len(_LOADED_LIBRARIES) > _LOADED_LIBRARIES_LIMIT:
            _LOADED_LIBRARIES.popitem(last=False)
    return LoadedLibrary(digest=digest, module=cached)


def _register_library_loaders() -> None:
    """Unpacking an ``export_library`` blob needs the loader the TVM CUDA runtime
    registers. Both this and the CuTe DSL preload cost an import, so they happen on
    the first library upload rather than at startup."""
    global _LOADERS_READY
    if _LOADERS_READY:
        return
    try:
        import tvm  # noqa: F401
    except Exception:
        pass  # tvm is optional; only embedded-blob libraries depend on it
    _preload_cute_dsl_runtime()
    _LOADERS_READY = True


def _preload_cute_dsl_runtime() -> None:
    """A CuTeDSL export needs ``libcute_dsl_runtime.so``, which no loader search
    path covers. Loading it by absolute path is enough: dlopen then resolves the
    dependency against the loaded SONAME."""
    try:
        from cutlass.runtime import find_runtime_libraries

        for path in find_runtime_libraries(enable_tvm_ffi=False):
            if Path(path).exists():
                ctypes.CDLL(path)
    except Exception:
        pass  # cutlass is optional; only CuTeDSL exports depend on it


def _library_dir() -> Path:
    global _LIBRARY_DIR
    if _LIBRARY_DIR is None:
        _LIBRARY_DIR = Path(tempfile.mkdtemp(prefix="kcoral-lib-"))
    return _LIBRARY_DIR


class _GPURuntimeFactory:
    """Two-phase, picklable factory used by spawned GPU workers."""

    def prepare(self) -> Callable[[], GPURuntime]:
        """Import mandatory dependencies without creating a CUDA context."""
        _require_torch_and_ffi()
        import torch

        if torch.cuda.is_initialized():  # guard future changes to preparation
            raise RuntimeError("GPU runtime preparation unexpectedly initialized CUDA")
        return GPURuntime

    def __call__(self) -> GPURuntime:
        return GPURuntime()


gpu_runtime_factory = _GPURuntimeFactory()
