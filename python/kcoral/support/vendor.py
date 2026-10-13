"""The GPU vendor backend, and the device operations that differ by vendor.

``kcoral server --backend`` resolves the backend once and exports it, so every
worker the server spawns (sandboxed or not) inherits the same choice. Each
backend module (:mod:`kcoral.support.cuda`, :mod:`kcoral.support.rocm`)
provides the functions dispatched here.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import shutil
from functools import cache
from pathlib import Path
from types import ModuleType

BACKENDS = ("cuda", "rocm")
ENVIRONMENT = "KCORAL_GPU_BACKEND"


def detect() -> str:
    """The backend of the installed torch build, without importing torch.

    ROCm wheels carry a ``+rocm`` local version. Builds that do not (such as
    some vendor containers) are recognized by a KFD device and no NVIDIA one.
    """
    try:
        if "rocm" in importlib.metadata.version("torch"):
            return "rocm"
    except importlib.metadata.PackageNotFoundError:
        pass
    if Path("/dev/kfd").exists() and not Path("/dev/nvidiactl").exists():
        return "rocm"
    return "cuda"


def select(requested: str) -> str:
    """Resolve ``auto`` and export the choice to this process and its workers."""
    if requested not in ("auto", *BACKENDS):
        raise ValueError(f"unknown GPU backend {requested!r}; expected auto, cuda or rocm")
    name = detect() if requested == "auto" else requested
    os.environ[ENVIRONMENT] = name
    backend.cache_clear()
    return name


@cache
def backend() -> str:
    value = os.environ.get(ENVIRONMENT, "auto")
    if value == "auto":
        return detect()
    if value not in BACKENDS:
        raise ValueError(f"{ENVIRONMENT}={value!r} is not one of auto, cuda, rocm")
    return value


def module() -> ModuleType:
    return importlib.import_module(f"kcoral.support.{backend()}")


# Remote profiling tools by vendor (``kcoral run TOOL``), as found on PATH.
_PROFILERS = {
    "cuda": (("ncu",), ("compute-sanitizer",), ("run-iket",)),
    "rocm": (("rocprofv3",), ("rocprof-compute", "omniperf")),  # tool, then executables
}


def capabilities(device: str) -> dict[str, object]:
    """What a client can rely on from this server's GPU workers, for ``/health``.

    ``timing`` is the ``benchmark`` method; ``cpu_only_check`` says when a
    ``cpu_only`` function's GPU access is caught: ``at_entry`` for any CUDA call,
    or ``torch_at_entry`` where torch operations are stopped as they are made and
    other HIP calls are reported after the function returns (ROCm);
    ``profilers`` are the remote tools installed.
    """
    if device != "gpu":
        return {"backend": "cpu"}
    name = backend()
    rocm = name == "rocm"
    return {
        "backend": name,
        "timing": "trace" if rocm else "cupti",
        "cpu_only_check": "torch_at_entry" if rocm else "at_entry",
        "profilers": [names[0] for names in _PROFILERS[name] if any(map(shutil.which, names))],
    }


def require_gpu_sandbox_support(
    gpu_ids: list[int] | tuple[int, ...], readonly_paths: tuple[Path, ...] = ()
) -> None:
    """Refuse bubblewrap for ROCm GPU workers that could not use their GPU there,
    rather than start them unisolated or let them fail.

    ROCm learns which GPUs exist from the KFD topology in sysfs and, through
    ROCm 7.0, acts on all of them: given only its own render node, a sandboxed
    worker sets a memory policy on another GPU and fails (``hipErrorNoDevice``).
    libhsakmt's ``hsakmt_fmm_init_process_apertures`` skips unusable GPUs when
    acquiring VMs but not when setting memory policies. It works when a device
    cgroup hides the other GPUs (a container given only these GPUs) or with a
    libhsakmt that skips them there too, so probe each GPU in a real sandbox.
    """
    if backend() != "rocm":
        return
    for gpu in dict.fromkeys(gpu_ids):
        problem = module().sandbox_problem(gpu, tuple(readonly_paths))
        if problem is not None:
            raise RuntimeError(
                f"bubblewrap isolation cannot be used for ROCm GPU {gpu}: {problem}. "
                "ROCm acts on every GPU it can read in the KFD topology; run the server "
                "in a container given only its GPUs, use a libhsakmt that skips unusable "
                "GPUs, or start the server with --sandbox none"
            )


def require_matching_torch() -> None:
    """Refuse a worker whose torch was built for the other vendor."""
    import torch

    built = "rocm" if getattr(torch.version, "hip", None) else "cuda"
    if built != backend():
        raise RuntimeError(
            f"the {backend()} backend was selected, but torch {torch.__version__} is a "
            f"{built} build; install the matching torch or pass --backend {built}"
        )


def strict_identity() -> bool:
    """Whether a worker that cannot confirm its card must be refused."""
    return module().STRICT_IDENTITY


def visibility_variables() -> tuple[str, ...]:
    """Environment variables through which the server pins a worker's GPUs."""
    return module().VISIBILITY_VARIABLES


def runtime_paths() -> tuple[str, ...]:
    """Vendor toolkit directories a sandboxed worker may read."""
    return module().RUNTIME_PATHS


def device_selector(gpu_id: int) -> str:
    return module().device_selector(gpu_id)


def device_identity(gpu_id: int) -> str | None:
    return module().device_identity(gpu_id)


def identity_key(value: str | None) -> str | None:
    return module().identity_key(value)


def visible_devices_env(selector: str) -> dict[str, str | None]:
    """Variables to set (or, for None, remove) so a worker sees only ``selector``."""
    return module().visible_devices_env(selector)


def device_nodes(gpu_ids: tuple[int, ...]) -> list[Path]:
    return module().device_nodes(gpu_ids)


def describe_device_identity() -> str | None:
    """Inside a worker: the identity of the card it actually got."""
    return module().describe_device_identity()


def take_last_error() -> str | None:
    """Inside a worker: consume the runtime's pending last error, if any."""
    return module().take_last_error()
