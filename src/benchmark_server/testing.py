"""GPU-free runtime used by the protocol and worker tests."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import ExecutionError
from .gpu_runtime import resolve_entry  # shared so the double cannot drift; touches no GPU


class _Opaque:
    pass


@dataclass
class _FakeTensor:
    data: bytes
    dtype: str
    shape: list[int]


def _opaque(*_args: Any) -> _Opaque:
    return _Opaque()


def _structural(*_args: Any) -> dict[str, Any]:
    return {"ok": True, "values": [None, False, 7, 1.5, "text"]}


def _binary(*_args: Any) -> bytes:
    return b"binary-result"


def _crash(*_args: Any):
    os._exit(1)


def _sleep(seconds: Any = 0.0, *_args: Any) -> dict[str, float]:
    time.sleep(float(seconds))
    return {"slept": float(seconds)}


_BUILTINS: dict[str, Callable] = {
    "builtin.opaque": _opaque,
    "builtin.structural": _structural,
    "builtin.binary": _binary,
    "builtin.crash": _crash,
    "builtin.sleep": _sleep,
}


class FakeRuntime:
    def load_module(self, source: str, entry: str | None = None) -> Any:
        namespace: dict[str, Any] = {}
        try:
            exec(compile(source, "<uploaded>", "exec"), namespace)
        except SyntaxError as exc:
            raise ExecutionError("parse", str(exc)) from exc
        except Exception as exc:
            raise ExecutionError("parse", f"{type(exc).__name__}: {exc}") from exc
        return resolve_entry(namespace, source, entry)

    def load_tensor(self, data: bytes, dtype: str, shape: list[int]) -> _FakeTensor:
        return _FakeTensor(data=data, dtype=dtype, shape=shape)

    def export_tensor(self, value: Any) -> tuple[str, list[int], bytes] | None:
        if not isinstance(value, _FakeTensor):
            return None
        return value.dtype, value.shape, value.data

    def builtin(self, name: str) -> Callable:
        fn = _BUILTINS.get(name)
        if fn is None:
            raise ExecutionError("runtime", f"unknown function: {name!r}")
        return fn

    def reset(self) -> None:
        pass


def fake_runtime_factory() -> FakeRuntime:
    return FakeRuntime()
