"""Helpers used by more than one builtin module."""

from __future__ import annotations

from typing import Any

from ..errors import ExecutionError


def torch_dtype(name: str) -> Any:
    """Resolve a dtype name (e.g. ``"float16"``) to the torch dtype."""
    import torch

    dt = getattr(torch, str(name), None)
    if not isinstance(dt, torch.dtype):
        raise ExecutionError("runtime", f"unknown dtype: {name!r}")
    return dt


def short(text: str | Exception, limit: int = 600) -> str:
    """Truncated text keeping both ends: a tvm diagnostic leads with its message,
    a compiler failure ends with its diagnostic."""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    head = limit // 3
    return f"{text[:head]} …[truncated]… {text[head - limit :]}"


def split_cfg(args: tuple) -> tuple[tuple, dict]:
    """Split a builtin's trailing config dict from its leading tensor args."""
    if args and isinstance(args[-1], dict):
        return args[:-1], args[-1]
    return args, {}
