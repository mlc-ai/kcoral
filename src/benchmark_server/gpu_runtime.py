"""The GPU runtime: materialize uploads, resolve builtins.

The one :class:`Runtime` that touches a GPU; it runs inside the worker process
and has no compiler of its own — compiling and running kernels are builtins (see
:mod:`benchmark_server.builtin_ops`). torch is imported lazily, so importing this
module touches no GPU.
"""

from __future__ import annotations

import hashlib
import json
import linecache
from collections.abc import Callable
from typing import Any

from . import builtin_ops
from .errors import ExecutionError
from .packages import load_package_entry

# An uploaded function module defines its entry object under this name.
ENTRY_POINT = "main"


class GPURuntime:
    """Construct inside the worker after its GPU is pinned. Building it checks that
    torch and tvm_ffi import (the mandatory deps), so a broken environment fails at
    startup, not mid-run. Full tvm is optional — only ``compile_tirx`` needs it, and
    it fails gracefully when tvm is absent."""

    def __init__(self) -> None:
        _require_torch_and_ffi()
        self._seeded_fnames: list[str] = []  # linecache keys to clear on reset
        self._package_cleanups: list[Callable[[], None]] = []

    def materialize(self, kind: str, data: bytes) -> Any:
        if kind == "function":
            return self._materialize_function(data)
        if kind == "tensor":
            return _materialize_tensor(data)
        if kind == "package":
            entry, cleanup = load_package_entry(data)
            self._package_cleanups.append(cleanup)
            return entry
        if kind == "object":
            return json.loads(data.decode("utf-8"))
        raise ExecutionError("runtime", f"kind {kind!r} not supported")

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
        for cleanup in self._package_cleanups:
            cleanup()
        self._package_cleanups.clear()
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass  # a poisoned CUDA context is handled at the worker level

    def _materialize_function(self, data: bytes) -> Any:
        source = data.decode("utf-8")
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
        if ENTRY_POINT not in ns:
            raise ExecutionError("parse", f"source must define {ENTRY_POINT!r}")
        return ns[ENTRY_POINT]


def _require_torch_and_ffi() -> None:
    try:
        import torch  # noqa: F401
        import tvm_ffi  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment misconfiguration
        raise RuntimeError(
            "The GPU runtime requires torch and tvm_ffi in the worker environment "
            f"(full tvm is optional). Importing them failed: {exc!r}"
        ) from exc


def _materialize_tensor(data: bytes) -> Any:
    import torch

    header, _, raw = data.partition(b"\x00")
    try:
        meta = json.loads(header.decode("utf-8"))
        dtype = builtin_ops.torch_dtype(meta["dtype"])
        shape = [int(d) for d in meta["shape"]]
        return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape).to("cuda")
    except ExecutionError:
        raise
    except Exception as exc:
        raise ExecutionError("runtime", f"malformed tensor: {exc}") from exc


def gpu_runtime_factory() -> GPURuntime:
    """Picklable factory so a spawned worker can build the runtime."""
    return GPURuntime()
