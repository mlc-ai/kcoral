"""Compiling CuTeDSL kernels. Needs nvidia-cutlass-dsl; without it
``compile_cutedsl`` reports ``unavailable``."""

from __future__ import annotations

import re
from typing import Any

from ..errors import ExecutionError
from ._common import short, split_cfg
from ._registry import register_builtin

# CuTeDSL colours its diagnostics, and the escapes would ship to the client.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


@register_builtin("compile_cutedsl", cpu_only=True)
def compile_cutedsl(fn: Any, *rest: Any) -> Any:
    """Compile a ``@cute.jit`` kernel handle for the tensors it will run on, whose
    dtype and layout it specializes on. Nothing is cached: what a compiled kernel
    stays valid for is CuTeDSL's business, not a key built here. cfg:
    ``options``, a CuTeDSL option string."""
    try:
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
    except ImportError as exc:  # nvidia-cutlass-dsl is an optional server dependency
        raise ExecutionError(
            "unavailable",
            "server-side CuTeDSL compilation requires nvidia-cutlass-dsl, "
            "which is not installed on this server",
        ) from exc

    args, cfg = split_cfg(rest)
    options = cfg.get("options")
    if options is not None and not isinstance(options, str):
        raise ExecutionError("compile", "compile_cutedsl options must be a string")
    try:
        operands = [_as_cute(value, from_dlpack) for value in args]
        # Compiling needs cute tensors; what comes back takes plain ones.
        return cute.compile(fn, *operands, **({"options": options} if options else {}))
    except ExecutionError:
        raise
    except Exception as exc:  # a missing @cute.jit, or anything the tracer rejects
        raise ExecutionError("compile", short(_ANSI.sub("", str(exc)).strip())) from exc


def _as_cute(value: Any, from_dlpack: Any) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        return value
    try:
        return from_dlpack(value)
    except Exception as exc:
        raise ExecutionError("compile", f"cannot hand a tensor to CuTeDSL: {short(exc)}") from exc
