"""The registry mapping a builtin's wire name to its callable."""

from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, Callable] = {}
_CPU_ONLY: set[str] = set()


def register_builtin(name: str, cpu_only: bool = False) -> Callable:
    """Register a module-level function as the builtin ``builtin.<name>``.

    ``cpu_only`` marks a builtin that touches no GPU, so a worker may drop its GPU
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


def cpu_only_builtins() -> frozenset[str]:
    """Every builtin declared to touch no GPU."""
    return frozenset(_CPU_ONLY)


def snapshot_registry() -> tuple[dict[str, Callable], set[str]]:
    """A copy of the registry, to hand back to :func:`restore_registry` later."""
    return dict(_REGISTRY), set(_CPU_ONLY)


def restore_registry(snapshot: tuple[dict[str, Callable], set[str]]) -> None:
    """Put the registry back as ``snapshot`` found it.

    Uploaded code runs in the worker process, where this module is importable and
    writable, so a program that rewires a builtin would otherwise serve the
    rewired one to every later request on that worker.
    """
    registry, cpu_only = snapshot
    _REGISTRY.clear()
    _REGISTRY.update(registry)
    _CPU_ONLY.clear()
    _CPU_ONLY.update(cpu_only)
