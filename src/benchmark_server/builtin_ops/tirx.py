"""Compiling TIRx kernels. Needs tvm; without it ``compile_tirx`` reports
``unavailable``."""

from __future__ import annotations

from typing import Any

from ..errors import ExecutionError
from ._common import short
from ._registry import register_builtin


@register_builtin("compile_tirx")
def compile_tirx(fn: Any, bindings: Any = None) -> Any:
    """Compile a ``@T.jit`` or ``@T.prim_func`` kernel handle. ``bindings`` supplies
    the ``T.constexpr`` values a ``@T.jit`` kernel is specialized on."""
    try:
        import tvm
    except ImportError as exc:  # tvm is an optional server dependency
        raise ExecutionError(
            "unavailable",
            "server-side compilation requires tvm, which is not installed on this server",
        ) from exc

    if bindings is not None and not isinstance(bindings, dict):
        raise ExecutionError("compile", "compile_tirx bindings must be a dict of constexpr values")
    kwargs = bindings or {}
    if isinstance(fn, tvm.tirx.PrimFunc):  # a @T.prim_func kernel — already concrete
        if kwargs:
            raise ExecutionError(
                "compile",
                "bindings apply only to @T.jit kernels; this kernel is already a PrimFunc",
            )
        pf = fn
    elif hasattr(fn, "specialize"):  # a @T.jit kernel handle (TIRJit)
        try:
            pf = fn.specialize(**kwargs)  # TIRX parse happens here
        except tvm.error.DiagnosticError as exc:
            raise ExecutionError("parse", short(exc)) from exc
        except TypeError as exc:  # wrong, missing, or unhashable constexpr bindings
            raise ExecutionError("compile", short(exc)) from exc
    else:
        raise ExecutionError(
            "compile", "compile_tirx expects a @T.jit or @T.prim_func kernel handle"
        )
    try:
        mod = tvm.IRModule({"main": pf})
        return tvm.compile(mod, target=tvm.target.Target("cuda"), tir_pipeline="tirx")
    except tvm.error.InternalError as exc:  # lowering
        raise ExecutionError("compile", short(exc)) from exc
    except RuntimeError as exc:  # codegen (nvcc/nvrtc)
        raise ExecutionError("compile", short(exc)) from exc
