"""Execute a validated program inside one GPU worker."""

from __future__ import annotations

import math
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO, Any, Protocol

from .errors import ExecutionError
from .keys import compute_blob_hash
from .schemas import (
    DTYPE_ITEM_SIZES,
    Program,
    ProgramOutcome,
    Ref,
    Return,
    Run,
    Upload,
    expected_tensor_nbytes,
)

DEFAULT_OUTPUT_LIMIT_BYTES = 1024**2


class Runtime(Protocol):
    def load_module(self, source: str) -> Any: ...
    def load_tensor(self, data: bytes, dtype: str, shape: list[int]) -> Any: ...
    def export_tensor(self, value: Any) -> tuple[str, list[int], bytes] | None: ...
    def builtin(self, name: str) -> Callable: ...
    def reset(self) -> None: ...


def execute(program: Program, runtime: Runtime) -> ProgramOutcome:
    """Run a program and serialize only values selected by return instructions."""
    env: dict[str, Any] = {}
    results: dict[str, dict[str, Any]] = {}
    encoder = _ValueEncoder(runtime)
    error: dict[str, Any] | None = None
    current_index: int | None = None
    captured = CapturedOutput()
    try:
        with _capture_output(_output_limit(program)) as captured:
            try:
                for current_index, instruction in enumerate(program.instructions):
                    if isinstance(instruction, Upload):
                        if instruction.kind == "module":
                            assert instruction.source is not None
                            env[instruction.id] = runtime.load_module(instruction.source)
                        else:
                            assert (
                                instruction.blob is not None
                                and instruction.dtype is not None
                                and instruction.shape is not None
                            )
                            env[instruction.id] = runtime.load_tensor(
                                program.blob_bytes[instruction.blob],
                                instruction.dtype,
                                instruction.shape,
                            )
                    elif isinstance(instruction, Run):
                        fn = _resolve_fn(instruction.fn, env, runtime)
                        args = [
                            env[arg.id] if isinstance(arg, Ref) else arg for arg in instruction.args
                        ]
                        env[instruction.id] = fn(*args)
                    elif isinstance(instruction, Return):
                        results[instruction.key] = encoder.encode(env[instruction.value.id])
            except ExecutionError as exc:
                error = {
                    "kind": exc.kind,
                    "message": exc.message,
                    "instruction_index": current_index,
                }
            except Exception as exc:
                error = {
                    "kind": "engine",
                    "message": f"{type(exc).__name__}: {exc}",
                    "instruction_index": current_index,
                }
    finally:
        runtime.reset()

    if error is not None:
        results = {}
        encoder.binary_parts.clear()
    return ProgramOutcome(
        status="FAILED" if error is not None else "COMPLETED",
        results=results,
        error=error,
        binary_parts=encoder.binary_parts,
        stdout=captured.stdout,
        stderr=captured.stderr,
        stdout_truncated=captured.stdout_truncated,
        stderr_truncated=captured.stderr_truncated,
    )


class _ValueEncoder:
    def __init__(self, runtime: Runtime) -> None:
        self._runtime = runtime
        self.binary_parts: dict[str, bytes] = {}

    def encode(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {"type": "null"}
        if isinstance(value, bool):
            return {"type": "boolean", "value": value}
        if isinstance(value, int):
            return {"type": "integer", "value": value}
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ExecutionError("serialization", "cannot return a non-finite number")
            return {"type": "number", "value": value}
        if isinstance(value, str):
            return {"type": "string", "value": value}
        if isinstance(value, (bytes, bytearray, memoryview)):
            return self._binary_value("bytes", bytes(value))
        if isinstance(value, (list, tuple)):
            return {"type": "array", "value": [self.encode(child) for child in value]}
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ExecutionError("serialization", "returned objects must have string keys")
            return {
                "type": "object",
                "value": {key: self.encode(child) for key, child in value.items()},
            }

        try:
            tensor = self._runtime.export_tensor(value)
        except Exception as exc:
            raise ExecutionError(
                "serialization", f"failed to export tensor: {type(exc).__name__}: {exc}"
            ) from exc
        if tensor is not None:
            dtype, shape, data = tensor
            if not isinstance(dtype, str) or dtype not in DTYPE_ITEM_SIZES:
                raise ExecutionError("serialization", f"cannot return tensor dtype {dtype!r}")
            if not isinstance(shape, list) or any(
                isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
                for dimension in shape
            ):
                raise ExecutionError("serialization", "returned tensor has an invalid shape")
            if not isinstance(data, bytes):
                raise ExecutionError("serialization", "returned tensor data must be bytes")
            expected_size = expected_tensor_nbytes(dtype, shape)
            if len(data) != expected_size:
                raise ExecutionError(
                    "serialization",
                    f"returned tensor metadata expects {expected_size} bytes, got {len(data)}",
                )
            encoded = self._binary_value("tensor", data)
            encoded.update({"dtype": dtype, "shape": shape})
            return encoded
        raise ExecutionError("serialization", f"cannot return value of type {type(value).__name__}")

    def _binary_value(self, value_type: str, data: bytes) -> dict[str, Any]:
        part_name = f"return:{len(self.binary_parts)}"
        self.binary_parts[part_name] = data
        return {
            "type": value_type,
            "part": part_name,
            "sha256": compute_blob_hash(data),
        }


def _output_limit(program: Program) -> int:
    return int(program.options.get("output_limit_bytes", DEFAULT_OUTPUT_LIMIT_BYTES))


@dataclass
class CapturedOutput:
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@contextmanager
def _capture_output(limit_bytes: int) -> Iterator[CapturedOutput]:
    """Capture process-level stdout and stderr for the complete request."""
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
    return data[:limit_bytes].decode("utf-8", errors="replace"), len(data) > limit_bytes


def _resolve_fn(fn: str | Ref, env: dict[str, Any], runtime: Runtime) -> Callable:
    if isinstance(fn, Ref):
        obj = env[fn.id]
        if not callable(obj):
            raise ExecutionError("runtime", f"handle {fn.id!r} is not callable")
        return obj
    return runtime.builtin(fn)
