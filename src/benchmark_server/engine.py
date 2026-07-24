"""The instruction engine: run a Program over a runtime, produce results.

Straight-line dataflow: instructions execute in order, threading handles through
an environment. A failed instruction stops execution; the rest are recorded as
SKIPPED (never omitted). This runs inside the worker process; the runtime is the
only thing that touches the GPU.

Each instruction's stdout/stderr is captured at the file-descriptor level (so
output from C extensions and CUDA kernel printf is included) and attached to its
result, truncated to ``options.output_limit_bytes``.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO, Any, Protocol

from .errors import ExecutionError
from .schemas import Program, Result, is_ref, to_structural

# Used when the front-end did not resolve an explicit limit into the options.
DEFAULT_OUTPUT_LIMIT_BYTES = 1024**2


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
    output_limit_bytes = _output_limit(program)
    try:
        for ins in program.instructions:
            if failed:
                results.append(
                    Result(ins.id, ins.op, "SKIPPED", error={"reason": "predecessor_failed"})
                )
                continue
            with _capture_output(output_limit_bytes) as captured:
                try:
                    if ins.op == "upload":
                        env[ins.id] = runtime.materialize(ins.kind, program.upload_bytes[ins.id])
                        result = Result(ins.id, "upload", "OK")
                    else:  # run
                        fn = _resolve_fn(ins.fn, env, runtime)
                        args = [env[a["$ref"]] if is_ref(a) else a for a in ins.args]
                        value = fn(*args)
                        env[ins.id] = value
                        result = Result(ins.id, "run", "OK", value=to_structural(value, ins.id))
                except ExecutionError as exc:
                    err = {"kind": exc.kind, "message": exc.message}
                    result = Result(ins.id, ins.op, "FAILED", error=err)
                    failed = True
                except Exception as exc:  # engine fault — reported, doesn't crash the worker
                    err = {"kind": "engine", "message": f"{type(exc).__name__}: {exc}"}
                    result = Result(ins.id, ins.op, "FAILED", error=err)
                    failed = True
            result.stdout = captured.stdout
            result.stderr = captured.stderr
            result.stdout_truncated = captured.stdout_truncated
            result.stderr_truncated = captured.stderr_truncated
            results.append(result)
    finally:
        runtime.reset()
    return results


def _output_limit(program: Program) -> int:
    limit = program.options.get("output_limit_bytes", DEFAULT_OUTPUT_LIMIT_BYTES)
    try:
        return int(limit)
    except (TypeError, ValueError):
        return DEFAULT_OUTPUT_LIMIT_BYTES


@dataclass
class CapturedOutput:
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@contextmanager
def _capture_output(limit_bytes: int) -> Iterator[CapturedOutput]:
    """Capture what the block writes to stdout/stderr, truncated to ``limit_bytes``.

    The capture swaps the process-level file descriptors (1 and 2), so output from
    C extensions and CUDA kernel printf is included, not just Python-level prints.
    A non-positive limit disables capture. This runs inside the worker process,
    which serves one program at a time, so swapping process-wide state is safe.
    """
    captured = CapturedOutput()
    if limit_bytes <= 0:
        yield captured
        return
    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        saved_sys_stdout, saved_sys_stderr = sys.stdout, sys.stderr
        os.dup2(stdout_file.fileno(), 1)
        os.dup2(stderr_file.fileno(), 2)
        # Rebind the Python-level streams too, in case the hosting process (e.g.
        # a test runner) replaced sys.stdout with an object not backed by fd 1.
        sys.stdout = _stream_over(stdout_file)
        sys.stderr = _stream_over(stderr_file)
        try:
            yield captured
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout, sys.stderr = saved_sys_stdout, saved_sys_stderr
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            captured.stdout, captured.stdout_truncated = _read_captured(stdout_file, limit_bytes)
            captured.stderr, captured.stderr_truncated = _read_captured(stderr_file, limit_bytes)


def _stream_over(file: IO[bytes]) -> IO[str]:
    return open(file.fileno(), "w", encoding="utf-8", errors="replace", closefd=False)


def _read_captured(file: IO[bytes], limit_bytes: int) -> tuple[str, bool]:
    file.seek(0)
    data = file.read(limit_bytes + 1)
    truncated = len(data) > limit_bytes
    return data[:limit_bytes].decode("utf-8", errors="replace"), truncated


def _resolve_fn(fn: Any, env: dict, runtime: Runtime) -> Callable:
    if is_ref(fn):
        obj = env[fn["$ref"]]
        if not callable(obj):
            raise ExecutionError("runtime", f"handle {fn['$ref']!r} is not callable")
        return obj
    return runtime.builtin(fn)  # a builtin name
