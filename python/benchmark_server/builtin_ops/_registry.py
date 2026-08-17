"""The registry mapping a builtin's wire name to its callable."""

from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, Callable] = {}
_CPU_ONLY: set[str] = set()


def register_builtin(name: str, cpu_only: bool = False) -> Callable:
    """Register a module-level function as the builtin ``builtin.<name>``.

    ``cpu_only`` supplies the placement default when an instruction uses
    ``gpu="auto"``. It marks work that touches no GPU, so a worker may drop its
    lease for the duration. Only worth declaring for builtins expensive enough to
    pay for the reacquisition; the default is the safe answer.
    """

    def decorator(fn: Callable) -> Callable:
        _REGISTRY["builtin." + name] = fn
        if cpu_only:
            _CPU_ONLY.add("builtin." + name)
        return fn

    return decorator


def resolve(name: str) -> Callable | None:
    """The builtin registered under ``name`` (e.g. ``builtin.randn``), or None."""
    return _REGISTRY.get(name)


def is_cpu_only(name: str) -> bool:
    """Whether ``name`` is a builtin declared to touch no GPU."""
    return name in _CPU_ONLY
