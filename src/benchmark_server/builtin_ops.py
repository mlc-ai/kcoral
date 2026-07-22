"""Server builtins and the registry that holds them.

Each builtin is a module-level function registered as ``builtin.<name>`` by
:func:`register_builtin`; the runtime looks one up with :func:`resolve`. Extend
the server by adding a builtin, not a runtime. torch/tvm are imported lazily, so
importing this module touches no GPU. Builtins raise :class:`ExecutionError`
tagged with the failing stage (parse / compile / runtime / correctness), or
``unavailable`` when an optional dependency (tvm) is not installed.
"""

from __future__ import annotations

from typing import Any, Callable

from .errors import ExecutionError

_REGISTRY: dict[str, Callable] = {}


def register_builtin(name: str) -> Callable:
    """Register a module-level function as the builtin ``builtin.<name>``."""

    def decorator(fn: Callable) -> Callable:
        _REGISTRY["builtin." + name] = fn
        return fn

    return decorator


def resolve(name: str) -> "Callable | None":
    """The builtin registered under ``name`` (e.g. ``builtin.randn``), or None."""
    return _REGISTRY.get(name)


@register_builtin("randn")
def randn(spec: Any) -> Any:
    import torch

    shape, dtype, seed = _read_spec(spec)
    if not dtype.is_floating_point:
        raise ExecutionError("runtime", "randn requires a floating dtype")
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(int(seed))
    return torch.randn(shape, dtype=dtype, device="cuda", generator=gen)


@register_builtin("empty")
def empty(spec: Any) -> Any:
    import torch

    shape, dtype, _ = _read_spec(spec)
    return torch.empty(shape, dtype=dtype, device="cuda")


@register_builtin("zeros")
def zeros(spec: Any) -> Any:
    import torch

    shape, dtype, _ = _read_spec(spec)
    return torch.zeros(shape, dtype=dtype, device="cuda")


@register_builtin("compile_tirx")
def compile_tirx(fn: Any, bindings: Any = None) -> Any:
    try:
        import tvm
    except ImportError as exc:  # tvm is an optional server dependency
        raise ExecutionError(
            "unavailable",
            "server-side compilation requires tvm, which is not installed on this server",
        ) from exc

    if not hasattr(fn, "specialize"):
        raise ExecutionError("compile", "compile_tirx expects a @T.jit kernel handle")
    kwargs = bindings if isinstance(bindings, dict) else {}
    try:
        pf = fn.specialize(**kwargs)  # TIRX parse happens here
    except tvm.error.DiagnosticError as exc:
        raise ExecutionError("parse", _short(exc)) from exc
    try:
        mod = tvm.IRModule({"main": pf})
        return tvm.compile(mod, target=tvm.target.Target("cuda"), tir_pipeline="tirx")
    except tvm.error.InternalError as exc:  # lowering
        raise ExecutionError("compile", _short(exc)) from exc
    except RuntimeError as exc:  # codegen (nvcc/nvrtc)
        raise ExecutionError("compile", _short(exc)) from exc


@register_builtin("benchmark")
def benchmark(mod: Any, *rest: Any) -> dict:
    import torch

    tensors, cfg = _split_cfg(rest)
    if not callable(mod):
        raise ExecutionError("runtime", "benchmark expects a compiled module handle")
    warmup = int(cfg.get("warmup", 10))
    repeat = int(cfg.get("repeat", 30))

    def call() -> None:
        mod(*tensors)

    try:
        for _ in range(max(warmup, 1)):
            call()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            call()
        end.record()
        torch.cuda.synchronize()
    except RuntimeError as exc:  # a tvm run error or torch "CUDA error: ..."
        raise ExecutionError("runtime", _short(exc)) from exc
    return {"latency_ms": start.elapsed_time(end) / repeat, "warmup": warmup, "repeat": repeat}


@register_builtin("check_close")
def check_close(actual: Any, expected: Any, *rest: Any) -> dict:
    import torch

    _, cfg = _split_cfg(rest)
    rtol = float(cfg.get("rtol", 1e-2))
    atol = float(cfg.get("atol", 1e-3))
    try:
        torch.cuda.synchronize()
        a = actual.float()
        e = expected.float()
        max_abs = float((a - e).abs().max())
        passed = bool(torch.allclose(a, e, rtol=rtol, atol=atol))
    except RuntimeError as exc:
        raise ExecutionError("runtime", _short(exc)) from exc
    return {"passed": passed, "max_abs_err": max_abs, "rtol": rtol, "atol": atol}


@register_builtin("assert_close")
def assert_close(actual: Any, expected: Any, *rest: Any) -> dict:
    """Like ``check_close``, but a mismatch is a failure: it raises ``correctness``
    so the instruction FAILs and the rest of the program is skipped."""
    result = check_close(actual, expected, *rest)
    if not result["passed"]:
        raise ExecutionError(
            "correctness",
            f"outputs differ: max_abs_err={result['max_abs_err']} exceeds "
            f"atol={result['atol']}, rtol={result['rtol']}",
        )
    return result


# --- helpers ----------------------------------------------------------------


def torch_dtype(name: str) -> Any:
    """Resolve a dtype name (e.g. ``"float16"``) to the torch dtype."""
    import torch

    dt = getattr(torch, str(name), None)
    if not isinstance(dt, torch.dtype):
        raise ExecutionError("runtime", f"unknown dtype: {name!r}")
    return dt


def _read_spec(spec: Any) -> "tuple[list[int], Any, Any]":
    if not isinstance(spec, dict):
        raise ExecutionError("runtime", "expected a {shape, dtype} spec")
    try:
        shape = [int(d) for d in spec["shape"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ExecutionError("runtime", f"bad 'shape' in spec: {exc}") from exc
    return shape, torch_dtype(spec.get("dtype", "float16")), spec.get("seed")


def _split_cfg(args: tuple) -> "tuple[tuple, dict]":
    """Split a builtin's trailing config dict from its leading tensor args."""
    if args and isinstance(args[-1], dict):
        return args[:-1], args[-1]
    return args, {}


def _short(exc: Exception, limit: int = 600) -> str:
    text = str(exc).strip()
    return text if len(text) <= limit else text[:limit] + " …[truncated]"
