"""Compiling Triton kernels. Needs triton; without it ``compile_triton`` reports
``unavailable``."""

from __future__ import annotations

from typing import Any

from ..errors import ExecutionError
from ._common import short, split_cfg
from ._registry import register_builtin


@register_builtin("compile_triton", cpu_only=True)
def compile_triton(fn: Any, *rest: Any) -> Any:
    """Compile a ``@triton.jit`` kernel for the arguments it specializes on. cfg:
    ``grid``, one to three ints, plus any keyword the launch takes — ``num_warps``,
    ``num_stages``, a constexpr by name. Triton does its own on-disk caching."""
    try:
        import triton  # noqa: F401
    except ImportError as exc:  # triton is an optional server dependency
        raise ExecutionError(
            "unavailable",
            "server-side Triton compilation requires triton, which is not installed on this server",
        ) from exc

    args, cfg = split_cfg(rest)
    options = dict(cfg)
    grid = _grid(options.pop("grid", None))
    warmup = getattr(fn, "warmup", None)
    if warmup is None:
        raise ExecutionError("compile", "compile_triton expects a @triton.jit kernel")
    try:
        # Compiles and launches nothing: Triton gates the launch on `warmup=False`.
        # The tensors are read for dtype and pointer alignment, never dereferenced.
        warmup(*args, grid=grid, **options)  # grid is required here, but unused
    except Exception as exc:  # a CompilationError, or a keyword the kernel rejects
        raise ExecutionError("compile", short(exc)) from exc

    def launch(*call_args: Any) -> None:
        # The grid has nowhere else to come from, and is not in Triton's cache key.
        # The options are: a launch differing from the warm-up recompiles, on-lease.
        try:
            fn[grid](*call_args, **options)
        except Exception as exc:  # Triton's errors do not subclass RuntimeError
            raise ExecutionError("runtime", short(exc)) from exc

    return launch


def _grid(value: Any) -> tuple[int, ...]:
    """The launch grid as data: the server will not evaluate a client's expression."""
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise ExecutionError("compile", "compile_triton needs a 'grid' of one to three ints")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in value):
        raise ExecutionError("compile", "compile_triton grid dimensions must be positive ints")
    return tuple(value)
