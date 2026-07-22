"""A GPU-free runtime used only by the tests.

It stands in for a real runtime so the engine, worker-pool, and HTTP-server tests
exercise the real IPC / crash / timeout / caching machinery with no GPU. Its
builtins are named for the engine behaviour they drive, not for any real
operation: ``opaque`` yields a value that returns to the client as a handle,
``structural`` yields a plain-JSON result that passes through, and ``crash`` /
``sleep`` drive the worker's crash and timeout handling.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from .errors import ExecutionError


class _Opaque:  # non-JSON object -> the engine returns it to the client as a handle
    pass


def _opaque(*_args: Any) -> _Opaque:
    return _Opaque()


def _structural(*_args: Any) -> dict:
    return {"ok": True}


def _crash(*_args: Any):
    os._exit(1)


def _sleep(seconds: Any = 0.0, *_args: Any) -> dict:
    time.sleep(float(seconds))
    return {"slept": float(seconds)}


_BUILTINS: dict[str, Callable] = {
    "builtin.opaque": _opaque,          # result is opaque -> comes back as a handle
    "builtin.structural": _structural,  # result is plain JSON -> passes through
    "builtin.crash": _crash,            # drives worker-crash handling
    "builtin.sleep": _sleep,            # drives worker-timeout handling
}


class FakeRuntime:
    def materialize(self, kind: str, data: bytes) -> Any:
        if kind != "function":
            raise ExecutionError("runtime", f"kind {kind!r} not supported")
        ns: dict = {}
        try:
            exec(compile(data.decode("utf-8"), "<uploaded>", "exec"), ns)
        except SyntaxError as exc:
            raise ExecutionError("parse", str(exc)) from exc
        fns = [v for v in ns.values() if callable(v)]
        if not fns:
            raise ExecutionError("parse", "source defines no function")
        return fns[-1]

    def builtin(self, name: str) -> Callable:
        fn = _BUILTINS.get(name)
        if fn is None:
            raise ExecutionError("runtime", f"unknown function: {name!r}")
        return fn

    def reset(self) -> None:
        pass


def fake_runtime_factory() -> FakeRuntime:
    """Module-level (picklable) factory so a spawned worker can build the fake."""
    return FakeRuntime()
