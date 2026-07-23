"""The instruction engine: run a Program over a runtime, produce results.

Straight-line dataflow: instructions execute in order, threading handles through
an environment. A failed instruction stops execution; the rest are recorded as
SKIPPED (never omitted). This runs inside the worker process; the runtime is the
only thing that touches the GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from .errors import ExecutionError
from .schemas import Program, Result, is_ref, to_structural


class Runtime(Protocol):
    """What the engine needs of a runtime (structural — no inheritance required)."""

    def materialize(self, kind: str, data: bytes) -> Any: ...
    def builtin(self, name: str) -> Callable: ...
    def reset(self) -> None: ...


def execute(program: Program, runtime: Runtime) -> list[Result]:
    """Run ``program`` over ``runtime``. ``program.upload_bytes`` must already hold
    each upload's resolved canonical bytes (the front-end fills it before dispatch)."""
    env: dict[str, Any] = {}
    results: list[Result] = []
    failed = False
    try:
        for ins in program.instructions:
            if failed:
                results.append(
                    Result(ins.id, ins.op, "SKIPPED", error={"reason": "predecessor_failed"})
                )
                continue
            try:
                if ins.op == "upload":
                    env[ins.id] = runtime.materialize(ins.kind, program.upload_bytes[ins.id])
                    results.append(Result(ins.id, "upload", "OK"))
                else:  # run
                    fn = _resolve_fn(ins.fn, env, runtime)
                    args = [env[a["$ref"]] if is_ref(a) else a for a in ins.args]
                    value = fn(*args)
                    env[ins.id] = value
                    results.append(Result(ins.id, "run", "OK", value=to_structural(value, ins.id)))
            except ExecutionError as exc:
                err = {"kind": exc.kind, "message": exc.message}
                results.append(Result(ins.id, ins.op, "FAILED", error=err))
                failed = True
            except Exception as exc:  # engine fault — reported, doesn't crash the worker loop
                err = {"kind": "engine", "message": f"{type(exc).__name__}: {exc}"}
                results.append(Result(ins.id, ins.op, "FAILED", error=err))
                failed = True
    finally:
        runtime.reset()
    return results


def _resolve_fn(fn: Any, env: dict, runtime: Runtime) -> Callable:
    if is_ref(fn):
        obj = env[fn["$ref"]]
        if not callable(obj):
            raise ExecutionError("runtime", f"handle {fn['$ref']!r} is not callable")
        return obj
    return runtime.builtin(fn)  # a builtin name
