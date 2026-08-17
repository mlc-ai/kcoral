"""Execute a validated program inside one GPU worker."""

from __future__ import annotations

import math
import os
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO, Any, Protocol

from .deferred import DeferredGPUResult
from .errors import ExecutionError
from .keys import compute_blob_hash
from .lease import Lease
from .schemas import (
    DTYPE_ITEM_SIZES,
    Instruction,
    Program,
    ProgramOutcome,
    Ref,
    Return,
    Run,
    Upload,
    expected_tensor_nbytes,
)

DEFAULT_OUTPUT_LIMIT_BYTES = 1024**2
MAX_TRACEBACK_BYTES = 8192


class Runtime(Protocol):
    def load_module(
        self, source: str, entry: str | None = None, language: str = "python"
    ) -> Any: ...
    def load_library(self, data: bytes, entry: str) -> Any: ...
    def load_tensor(self, data: bytes, dtype: str, shape: list[int]) -> Any: ...
    def export_tensor(self, value: Any) -> tuple[str, list[int], bytes] | None: ...
    def builtin(self, name: str) -> Callable: ...
    def is_cpu_only(self, name: str) -> bool: ...
    def synchronize(self) -> None: ...
    def take_last_error(self) -> str | None: ...
    def reset(self) -> None: ...


def execute(
    program: Program,
    runtime: Runtime,
    lease: Lease,
    *,
    progress: Callable[[int], None] | None = None,
    cleanup_failed: Callable[[BaseException], None] | None = None,
) -> ProgramOutcome:
    """Run a program and serialize only values selected by return instructions.

    A run may require the GPU, release it, or leave placement to the existing
    safe defaults. Uploads and returns always acquire the lease.
    """
    env: dict[str, Any] = {}
    results: dict[str, dict[str, Any]] = {}
    encoder = _ValueEncoder(runtime)
    error: dict[str, Any] | None = None
    current_index: int | None = None
    current: Instruction | None = None
    captured = CapturedOutput()
    try:
        with _capture_output(_output_limit(program)) as captured:
            try:
                for current_index, instruction in enumerate(program.instructions):
                    current = instruction
                    if progress is not None:
                        progress(current_index)
                    _place(instruction, runtime, lease)
                    if isinstance(instruction, Upload):
                        if instruction.kind == "module":
                            assert instruction.source is not None
                            env[instruction.id] = runtime.load_module(
                                instruction.source,
                                entry=instruction.entry,
                                language=instruction.language,
                            )
                        elif instruction.kind == "bytes":
                            assert instruction.blob is not None
                            env[instruction.id] = program.blob_bytes[instruction.blob]
                        elif instruction.kind == "library":
                            assert instruction.blob is not None and instruction.entry is not None
                            env[instruction.id] = runtime.load_library(
                                program.blob_bytes[instruction.blob], instruction.entry
                            )
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
                        value = _invoke(instruction, env, runtime)
                        if isinstance(value, DeferredGPUResult):
                            if instruction.gpu == "none":
                                raise ExecutionError(
                                    "runtime",
                                    "instruction declared gpu='none' but requires GPU finalization",
                                )
                            # The builtin completed its host-only phase without the
                            # lease.  Driver/module loading must be serialized with
                            # every other use of this GPU before the instruction is
                            # considered complete.
                            lease.acquire()
                            value = value.resolve()
                        env[instruction.id] = value
                    elif isinstance(instruction, Return):
                        # A return that fails mid-encode must leave nothing behind: it
                        # adds no results entry, so a binary part it already registered
                        # for an encoded child would ship unreferenced. The client
                        # rejects that, masking the real serialization error.
                        checkpoint = encoder.checkpoint()
                        try:
                            results[instruction.key] = encoder.encode(env[instruction.value.id])
                        except BaseException:
                            encoder.rollback(checkpoint)
                            raise
            except ExecutionError as exc:
                error = _instruction_error(exc.kind, exc.message, current_index, current)
            except Exception as exc:
                error = _instruction_error(
                    "engine", f"{type(exc).__name__}: {exc}", current_index, current
                )
    finally:
        # A CUDA fault is often first reported by an instruction-level sync and
        # then reported again while draining/resetting the runtime. Preserve the
        # original instruction error, but tell the worker owner that this process
        # must not serve another request. If cleanup is where an asynchronous
        # fault first surfaces, attribute it to the active instruction as runtime.
        try:
            _drop_gpu(runtime, lease)
        except Exception as exc:
            if error is None:
                error = _instruction_error(
                    "runtime", f"{type(exc).__name__}: {exc}", current_index, current
                )
            if cleanup_failed is not None:
                cleanup_failed(exc)
        else:
            try:
                last_error = runtime.take_last_error()
            except Exception as exc:
                if error is None:
                    error = _instruction_error(
                        "runtime", f"{type(exc).__name__}: {exc}", current_index, current
                    )
                if cleanup_failed is not None:
                    cleanup_failed(exc)
            else:
                # A launch-configuration error can live in CUDA's thread-local
                # last-error slot without making synchronize fail. Consume it
                # here so it belongs to this request rather than the next CUDA
                # API call. Reading it clears the slot; the context is healthy
                # and must not be replaced.
                if last_error is not None:
                    error = _instruction_error("runtime", last_error, current_index, current)
        env.clear()
        try:
            runtime.reset()
        except Exception as exc:
            if error is None:
                error = _instruction_error(
                    "runtime", f"{type(exc).__name__}: {exc}", current_index, current
                )
            if cleanup_failed is not None:
                cleanup_failed(exc)

    # A failure stops the program but keeps the returns that already ran.
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


def _invoke(instruction: Run, env: dict[str, Any], runtime: Runtime) -> Any:
    """Call one ``Run``'s target, in its own frame so the handle and arguments die
    with the instruction. Left in ``execute``'s locals they keep an uploaded
    module's namespace alive past the ``runtime.reset()`` meant to free it."""
    fn = _resolve_fn(instruction.fn, env, runtime)
    args = [env[arg.id] if isinstance(arg, Ref) else arg for arg in instruction.args]
    return fn(*args)


def _place(instruction: Instruction, runtime: Runtime, lease: Lease) -> None:
    """Hold or drop the GPU for the instruction about to run."""
    if isinstance(instruction, Run):
        if instruction.gpu == "required":
            lease.acquire()
            return
        if instruction.gpu == "none":
            _drop_gpu(runtime, lease)
            return
        if isinstance(instruction.fn, str) and runtime.is_cpu_only(instruction.fn):
            # A compile: hand the GPU over so another worker can measure on it.
            _drop_gpu(runtime, lease)
            return
    lease.acquire()


def _drop_gpu(runtime: Runtime, lease: Lease) -> None:
    """Give the GPU up, draining it first. Guarded on ``held`` so a run of
    CPU-only instructions pays for one drain rather than one each."""
    if lease.held:
        runtime.synchronize()
        lease.release()


def _instruction_error(
    kind: str, message: str, index: int | None, instruction: Instruction | None
) -> dict[str, Any]:
    """Describe the failing instruction alongside the raised error."""
    return {
        "kind": kind,
        "message": message,
        "instruction_index": index,
        "instruction_op": instruction.op if instruction is not None else None,
        # ``Return`` carries a key rather than a handle name, so it has no id.
        "instruction_id": getattr(instruction, "id", None),
        "traceback": traceback.format_exc()[-MAX_TRACEBACK_BYTES:],
    }


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

    def checkpoint(self) -> int:
        return len(self.binary_parts)

    def rollback(self, checkpoint: int) -> None:
        """Drop parts registered since ``checkpoint``, keeping numbering contiguous."""
        for name in list(self.binary_parts)[checkpoint:]:
            del self.binary_parts[name]

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
