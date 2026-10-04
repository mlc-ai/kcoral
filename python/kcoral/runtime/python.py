"""CPU execution, uploaded Python modules, and request state restoration."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import linecache
import os
import sys
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

from kcoral.errors import ExecutionError
from kcoral.support.platform import _environ

# Knobs read and written as an attribute, as (dotted path under ``torch``, name).
# ``fp32_precision`` rather than the ``allow_tf32`` that shadows it: the boolean
# cannot express "unset", so writing it back turns 'none' into 'ieee' and leaves
# ``get_float32_matmul_precision`` raising for every program that follows.
_TORCH_ATTRS = (
    ("backends.cuda.matmul", "fp32_precision"),
    ("backends.cudnn", "fp32_precision"),
    ("backends.cudnn.conv", "fp32_precision"),
    ("backends.cudnn.rnn", "fp32_precision"),
    ("backends.cuda.matmul", "allow_fp16_reduced_precision_reduction"),
    ("backends.cuda.matmul", "allow_bf16_reduced_precision_reduction"),
    ("backends.cudnn", "benchmark"),
    ("backends.cudnn", "deterministic"),
)


# Restore getter/setter pairs before attributes: ``set_float32_matmul_precision``
# writes through to the per-backend knobs, so the finer-grained ones must land last.
_TORCH_CALLS = (
    ("get_float32_matmul_precision", "set_float32_matmul_precision"),
    ("get_default_dtype", "set_default_dtype"),
    ("get_default_device", "set_default_device"),
    ("is_grad_enabled", "set_grad_enabled"),
    ("are_deterministic_algorithms_enabled", "use_deterministic_algorithms"),
)


# The RNG is deliberately not restored: rewinding it would hand every request the
# same random draw, a bigger change than the reseeding it would undo.


def snapshot() -> dict[str, Any]:
    """The settings as they stand, to hand back to :func:`restore` later.

    Capture PyTorch settings only if it is already loaded; taking a snapshot
    does not import PyTorch.

    Take it after the runtime's warm-up, not before: the warm-up sets
    ``CUTE_DSL_LIBS`` and its loaders run once, so an earlier snapshot would restore
    the variable away with nothing left to set it again.
    """
    return {"torch": _snapshot_torch(), "environ": _environ()}


def restore(snapshot: dict[str, Any]) -> None:
    """Put the settings back as ``snapshot`` found them."""
    if snapshot["torch"]:
        _restore_torch(snapshot["torch"])
    environ = snapshot["environ"]
    current = _environ()
    for name in set(current) - set(environ):
        os.unsetenv(name)  # reaches libc even for a name ``os.environ`` never saw
        os.environ.pop(name, None)
    for name, value in environ.items():
        if current.get(name) != value:
            os.environ[name] = value  # ``__setitem__`` putenv()s, so libc is fixed too


def _snapshot_torch() -> dict[str, Any]:
    torch = sys.modules.get("torch")
    if torch is None:
        return {}

    state: dict[str, Any] = {}
    for path, name in _TORCH_ATTRS:
        try:
            state[f"{path}.{name}"] = getattr(_reach(torch, path), name)
        except Exception:
            pass  # a torch build without this knob
    for getter, setter in _TORCH_CALLS:
        try:
            state[setter] = getattr(torch, getter)()
        except Exception:
            pass
    return state


def _restore_torch(state: dict[str, Any]) -> None:
    """Write back only the knobs that actually moved.

    Setting one to the value it already holds is not free: ``set_default_device``
    installs a torch function mode that intercepts every tensor creation afterwards.
    """
    import torch

    for getter, setter in _TORCH_CALLS:
        if setter in state:
            try:
                current = getattr(torch, getter)()
            except Exception:
                current = _UNREADABLE  # write back rather than assume it is already right
            if current != state[setter]:
                try:
                    getattr(torch, setter)(state[setter])
                except Exception:
                    pass
    for path, name in _TORCH_ATTRS:
        if f"{path}.{name}" in state:
            try:
                holder = _reach(torch, path)
                if getattr(holder, name) != state[f"{path}.{name}"]:
                    setattr(holder, name, state[f"{path}.{name}"])
            except Exception:
                pass  # a program may have made the knob itself unwritable


_UNREADABLE = object()


def _reach(root: Any, path: str) -> Any:
    for part in path.split("."):
        root = getattr(root, part)
    return root


@dataclass(frozen=True)
class LoadedPythonModule:
    namespace: dict[str, Any]

    def get_function(self, name: str) -> Any:
        try:
            return self.namespace[name]
        except KeyError:
            raise ExecutionError(
                "parse", f"the uploaded Python module defines no name {name!r}"
            ) from None


def materialize_module(source: str, seeded_fnames: list[str]) -> LoadedPythonModule:
    # A kernel is re-read from its source text at compile time, so seed
    # linecache. Key by content hash so two functions in one program don't
    # overwrite each other's source.
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]
    fname = f"<uploaded:{digest}>"
    linecache.cache[fname] = (len(source), None, source.splitlines(True), fname)
    seeded_fnames.append(fname)
    ns: dict = {}
    try:
        # Runs in the request worker process, with its configured device visibility.
        exec(compile(source, fname, "exec"), ns)
    except SyntaxError as exc:
        raise ExecutionError("parse", f"syntax error: {exc}") from exc
    except Exception as exc:
        raise ExecutionError("parse", f"{type(exc).__name__}: {exc}") from exc
    return LoadedPythonModule(namespace=ns)


class CPURuntime:
    """Runtime exposed by ``--device cpu`` workers.

    CPU workers accept source files and can return a compiled shared object.
    Uploaded Python supplies the compilation harness. GPU tensors and uploaded
    libraries remain unavailable; constructing this runtime imports no GPU library.
    """

    def __init__(self) -> None:
        self._seeded_fnames: list[str] = []
        self._process_state = snapshot()

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

    def prepare_to_release_gpu(self) -> None:
        self.synchronize()

    def take_last_error(self) -> str | None:
        return None

    def reset(self) -> None:
        import linecache

        for fname in self._seeded_fnames:
            linecache.cache.pop(fname, None)
        self._seeded_fnames.clear()
        gc.collect()
        restore(self._process_state)


def cpu_runtime_factory() -> CPURuntime:
    """Picklable factory used by spawned CPU workers."""
    return CPURuntime()
