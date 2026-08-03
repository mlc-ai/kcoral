"""The GPU runtime: materialize uploads, resolve builtins.

The one :class:`Runtime` that touches a GPU; it runs inside the worker process
and has no compiler of its own — compiling and running kernels are builtins (see
:mod:`benchmark_server.builtin_ops`). torch is imported lazily, so importing this
module touches no GPU.
"""

from __future__ import annotations

import hashlib
import linecache
from collections.abc import Callable
from typing import Any

from . import builtin_ops
from .errors import ExecutionError

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

    def load_module(self, source: str) -> Any:
        return self._materialize_module(source)

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

    def _materialize_module(self, source: str) -> Any:
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
        # Deliberately not a callable check: a ``@T.jit`` kernel is a TIRJit, which
        # defines no ``__call__`` and is meant for ``builtin.compile_tirx`` rather
        # than direct invocation. Calling a non-callable handle is caught at run
        # time, and compiling a non-kernel is caught by the builtin.
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


def gpu_runtime_factory() -> GPURuntime:
    """Picklable factory so a spawned worker can build the runtime."""
    return GPURuntime()
