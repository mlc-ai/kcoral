"""The registry mapping a builtin's wire name to its callable."""

from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, Callable] = {}


def register_builtin(name: str) -> Callable:
    """Register a module-level function as the builtin ``builtin.<name>``."""

    def decorator(fn: Callable) -> Callable:
        _REGISTRY["builtin." + name] = fn
        return fn

    return decorator


def resolve(name: str) -> Callable | None:
    """The builtin registered under ``name`` (e.g. ``builtin.randn``), or None."""
    return _REGISTRY.get(name)
