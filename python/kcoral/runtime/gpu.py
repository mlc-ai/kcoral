"""The GPU runtime: materialize uploads, load uploaded modules.

The one :class:`Runtime` that touches a GPU; it runs inside the worker process
and executes the harness supplied by the client. torch is imported lazily, so importing this
module touches no GPU.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import hashlib
import importlib.metadata
import linecache
import os
import tempfile
import time
import traceback
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from kcoral.errors import ExecutionError, GPUAccessViolation
from kcoral.runtime import python as process_state
from kcoral.runtime.python import LoadedPythonModule, materialize_module
from kcoral.support import sandbox
from kcoral.support.cuda import _cuda_error_api

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

    module: Any


@dataclass(frozen=True)
class LoadedFunction:
    """A callable that keeps its defining module alive."""

    owner: LoadedLibrary
    function: Callable

    def __call__(self, *args: Any) -> Any:
        return self.function(*args)


class GPURuntime:
    """Construct inside the worker after its GPU is pinned. Building it checks that
    torch and tvm_ffi import (the mandatory deps), so a broken environment fails at
    startup, not mid-run. Full tvm is optional; uploaded TIRx harnesses and
    libraries containing TVM modules need it."""

    def __init__(self) -> None:
        _require_torch_and_ffi()
        self._seeded_fnames: list[str] = []  # linecache keys to clear on reset
        _warm_up()
        self._cupti_guard_used = False
        self._process_state = process_state.snapshot()
        self._request_libraries: list[LoadedLibrary] = []

    def load_module(self, source: str) -> Any:
        return materialize_module(source, self._seeded_fnames)

    def load_library(self, data: bytes) -> LoadedLibrary:
        library = _materialize_library(data)
        # `env` is cleared before reset, but objects returned by the DSO may be
        # collected through cycles. Keep its code mapped through that collection.
        self._request_libraries.append(library)
        return library

    def get_function(self, module: Any, name: str) -> Any:
        if isinstance(module, LoadedPythonModule):
            return module.get_function(name)
        if not isinstance(module, LoadedLibrary):
            raise ExecutionError(
                "runtime", "get_function expects an uploaded module or library handle"
            )
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

    @contextmanager
    def forbid_gpu(self) -> Iterator[None]:
        """Fail the guarded call once it has entered any CUDA runtime or driver API.
        Best effort: the call is seen from inside, and a child process not at all."""
        try:
            from cupti import cupti
        except ImportError as exc:
            raise ExecutionError(
                "unavailable",
                "verifying a cpu_only function needs cupti-python in the worker environment",
            ) from exc

        violation: GPUAccessViolation | None = None

        def record_first_call(_userdata, _domain, callback_id, callback_data) -> None:
            nonlocal violation
            if (
                violation is not None
                or callback_data.callback_site != cupti.ApiCallbackSite.API_ENTER
            ):
                return
            detected_at_ns = time.monotonic_ns()
            cuda_call = str(callback_data.function_name or f"CUDA callback {callback_id}")
            try:
                frames = traceback.extract_stack()[:-1]  # below this callback
                location, call_stack = _call_site(frames), "".join(traceback.format_list(frames))
            except Exception:  # nothing may escape into the CUDA call
                location, call_stack = "an unknown call site", ""
            violation = GPUAccessViolation(cuda_call, location, call_stack, detected_at_ns)

        domains = (cupti.CallbackDomain.RUNTIME_API, cupti.CallbackDomain.DRIVER_API)
        try:
            subscriber = cupti.subscribe(record_first_call, 0)
            self._cupti_guard_used = True
            for domain in domains:
                cupti.enable_domain(1, subscriber, domain)
        except cupti.cuptiError as exc:
            raise ExecutionError(
                "unavailable", f"CUPTI cannot watch for CUDA calls: {exc}"
            ) from exc
        try:
            yield
        finally:
            try:
                for domain in domains:
                    cupti.enable_domain(0, subscriber, domain)
            finally:
                cupti.unsubscribe(subscriber)
            if violation is not None:
                raise violation

    def synchronize(self) -> None:
        """Drain the GPU, so no kernel of this request is still running when the
        lease is given up and another worker starts measuring."""
        import torch

        # Do not hide a poisoned context. The engine preserves the request's
        # instruction error while the parent replaces this worker process.
        torch.cuda.synchronize()

    def prepare_to_release_gpu(self) -> None:
        """Release unused allocator cache so peers need not wait for final cleanup."""
        import torch

        self.synchronize()
        if torch.cuda.memory_reserved() > torch.cuda.memory_allocated():
            torch.cuda.empty_cache()
            self.synchronize()

    def take_last_error(self) -> str | None:
        """Consume CUDA's thread-local last error, if one is pending.

        Some invalid launches do not enqueue work and therefore are not reported
        by ``synchronize``. Leaving that error behind makes an unrelated CUDA API
        call in the next request report it instead.
        """
        return _cuda_error_api().take_last_error()

    def reset(self) -> None:
        import torch

        process_state.restore(self._process_state)
        for fname in self._seeded_fnames:
            linecache.cache.pop(fname, None)
        self._seeded_fnames.clear()
        # An uploaded module's namespace is a reference cycle (its functions'
        # `__globals__` point back at it), so only a collection frees the
        # module-scope tensors `empty_cache()` would otherwise find still live.
        gc.collect()
        self._request_libraries.clear()
        if sandbox.active():
            # Uploaded library mappings and static state must not survive a
            # request in a reused filesystem sandbox.
            _LOADED_LIBRARIES.clear()
            gc.collect()
        # CUDA errors are sticky within a process. Surface one so the parent can
        # replace this worker instead of returning its context to the pool.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if self._cupti_guard_used:
            # Unsubscribe stops callbacks but leaves CUPTI helper threads alive.
            # Finalize only after GPU work is drained.
            # The built-in timer already finalizes its own activity session.
            from cupti import cupti

            cupti.finalize()
            self._cupti_guard_used = False


def _call_site(frames: list[traceback.FrameSummary]) -> str:
    """Where the program's own code made the call: the innermost uploaded frame,
    else the innermost one outside this package."""
    package = str(Path(__file__).resolve().parents[1])
    for accept in (
        lambda frame: frame.filename.startswith("<uploaded:"),
        lambda frame: not frame.filename.startswith(package),
    ):
        for frame in reversed(frames):
            if accept(frame):
                return f"{frame.filename}:{frame.lineno} in {frame.name}"
    return "an unknown call site"


def _warm_up() -> None:
    """Initialize CUDA under the GPU lease; optional libraries are best-effort."""
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
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ExecutionError("runtime", f"unknown dtype: {dtype_name!r}")
        if not data:
            return torch.empty(shape, dtype=dtype, device="cuda")
        # The temporary host view is read only and never escapes this function.
        # CUDA receives independent storage; copying the bytes on CPU first
        # needlessly extends the GPU lease for large uploads.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given buffer is not writable", category=UserWarning
            )
            host = torch.frombuffer(data, dtype=dtype).reshape(shape)
        return host.to("cuda")
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
    """Return a copy of this worker's startup dependency metadata."""
    return dict(_describe_versions())


@cache
def _describe_versions() -> dict[str, str]:
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
    """Load shared-library bytes through TVM FFI.

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
            if sandbox.active():
                # TVM FFI otherwise retains its own process-lifetime reference,
                # even after our Python library cache has been cleared.
                cached = tvm_ffi.load_module(str(path), keep_module_alive=False)
            else:
                cached = tvm_ffi.load_module(str(path))
        except Exception as exc:
            raise ExecutionError("compile", f"cannot load the uploaded library: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)  # the mapping outlives the file
        _LOADED_LIBRARIES[digest] = cached
        if len(_LOADED_LIBRARIES) > _LOADED_LIBRARIES_LIMIT:
            _LOADED_LIBRARIES.popitem(last=False)
    return LoadedLibrary(module=cached)


def _register_library_loaders() -> None:
    """Load TVM and CuTe host libraries once, normally during worker preparation."""
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
    if sandbox.active():
        return Path(sandbox.WORKSPACE) / sandbox.PRIVATE / "libraries"
    if _LIBRARY_DIR is None:
        _LIBRARY_DIR = Path(tempfile.mkdtemp(prefix="kcoral-lib-"))
    return _LIBRARY_DIR


class _GPURuntimeFactory:
    """Two-phase, picklable factory used by spawned GPU workers."""

    def prepare(self) -> Callable[[], GPURuntime]:
        """Load host dependencies without a CUDA context, outside the GPU lease."""
        _require_torch_and_ffi()
        import torch

        # CuTe may query the driver version, but must not create a context.
        _register_library_loaders()
        describe_versions()
        # Keep imported objects out of GC scans during CUDA startup.
        gc.collect()
        gc.freeze()
        if torch.cuda.is_initialized():  # guard future changes to preparation
            raise RuntimeError("GPU runtime preparation unexpectedly initialized CUDA")
        return GPURuntime

    def __call__(self) -> GPURuntime:
        return GPURuntime()


gpu_runtime_factory = _GPURuntimeFactory()
