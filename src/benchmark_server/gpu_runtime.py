"""The GPU runtime: materialize uploads, resolve builtins.

The one :class:`Runtime` that touches a GPU; it runs inside the worker process
and has no compiler of its own — compiling and running kernels are builtins (see
:mod:`benchmark_server.builtin_ops`). torch is imported lazily, so importing this
module touches no GPU.
"""

from __future__ import annotations

import ast
import hashlib
import linecache
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import builtin_ops
from .errors import ExecutionError

# The name an uploaded module's entry object takes when the upload names none.
ENTRY_POINT = "main"

# Libraries already dlopened by this worker, keyed by the SHA-256 of their bytes.
# Purely a memoization — every request carries the bytes it needs. Entries live as
# long as the worker: a dlopened object cannot be unmapped while values it produced
# are still reachable.
_LOADED_LIBRARIES: dict[str, Any] = {}
_LIBRARY_DIR: Path | None = None
_LOADERS_READY = False


class GPURuntime:
    """Construct inside the worker after its GPU is pinned. Building it checks that
    torch and tvm_ffi import (the mandatory deps), so a broken environment fails at
    startup, not mid-run. Full tvm is optional — only ``compile_tirx`` needs it, and
    it fails gracefully when tvm is absent."""

    def __init__(self) -> None:
        _require_torch_and_ffi()
        self._seeded_fnames: list[str] = []  # linecache keys to clear on reset

    def load_module(self, source: str, entry: str | None = None, language: str = "python") -> Any:
        if language == "cuda":
            # Nothing runs here: `builtin.compile_cuda` turns the text into a module.
            assert entry is not None
            return builtin_ops.CUDASource(source=source, entry=entry)
        return self._materialize_module(source, entry)

    def load_library(self, data: bytes, entry: str) -> Any:
        return _materialize_library(data, entry)

    def target(self) -> dict[str, str]:
        """What a client must compile a library for."""
        return describe_target()

    def versions(self) -> dict[str, str]:
        """The worker's library versions, for a client comparing its own."""
        return describe_versions()

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

    def reset(self) -> None:
        import torch

        for fname in self._seeded_fnames:
            linecache.cache.pop(fname, None)
        self._seeded_fnames.clear()
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass  # a poisoned CUDA context is handled at the worker level

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
    return {"arch": f"sm_{major}{minor}a" if major >= 9 else f"sm_{major}{minor}"}


def describe_versions() -> dict[str, str]:
    """Versions a client may want to match; optional dependencies are absent."""
    import torch

    versions = {"torch": torch.__version__}
    if torch.version.cuda:
        versions["cuda"] = torch.version.cuda
    for name in ("tvm", "tvm_ffi"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # optional, or no version attribute
            pass
    return versions


def _materialize_library(data: bytes, entry: str) -> Any:
    """Load a prebuilt shared object and bind the function it exports as ``entry``.
    Loading needs a path, so the bytes go to a file unlinked once dlopen maps it."""
    import tvm_ffi

    _register_library_loaders()
    digest = hashlib.sha256(data).hexdigest()
    cached = _LOADED_LIBRARIES.get(digest)
    if cached is None:
        path = _library_dir() / f"{digest}.so"
        path.write_bytes(data)
        try:
            cached = tvm_ffi.load_module(str(path))
        except Exception as exc:
            raise ExecutionError("compile", f"cannot load the uploaded library: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)  # the mapping outlives the file
        _LOADED_LIBRARIES[digest] = cached
    if not cached.implements_function(entry):
        raise ExecutionError("compile", f"the uploaded library exports no function {entry!r}")
    return getattr(cached, entry)


def _register_library_loaders() -> None:
    """Unpacking an ``export_library`` blob needs the loader the TVM CUDA runtime
    registers, and nothing else in a library-only program imports tvm."""
    global _LOADERS_READY
    if _LOADERS_READY:
        return
    try:
        import tvm  # noqa: F401
    except Exception:
        pass  # tvm is optional; only embedded-blob libraries depend on it
    _LOADERS_READY = True


def _library_dir() -> Path:
    global _LIBRARY_DIR
    if _LIBRARY_DIR is None:
        _LIBRARY_DIR = Path(tempfile.mkdtemp(prefix="benchmark-server-lib-"))
    return _LIBRARY_DIR


def gpu_runtime_factory() -> GPURuntime:
    """Picklable factory so a spawned worker can build the runtime."""
    return GPURuntime()
