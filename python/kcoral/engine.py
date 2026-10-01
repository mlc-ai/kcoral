"""Execute a validated program inside one GPU worker."""

from __future__ import annotations

import math
import os
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import IO, Any, Protocol

from . import sandbox
from .artifacts import ReturnedFile, ReturnedFolder
from .errors import ExecutionError, GPUAccessViolation
from .file_transfer import collect
from .keys import compute_blob_hash
from .lease import Lease
from .schemas import (
    DTYPE_ITEM_SIZES,
    FileReturn,
    FileUpload,
    GetFunction,
    Instruction,
    Program,
    ProgramOutcome,
    Ref,
    Return,
    Run,
    Upload,
    expected_tensor_nbytes,
    normalize_file_path,
)

DEFAULT_OUTPUT_LIMIT_BYTES = 1024**2
MAX_TRACEBACK_BYTES = 8192


class Runtime(Protocol):
    def load_module(self, source: str) -> Any: ...
    def load_library(self, data: bytes) -> Any: ...
    def get_function(self, module: Any, name: str) -> Any: ...
    def load_tensor(self, data: bytes, dtype: str, shape: list[int]) -> Any: ...
    def export_tensor(self, value: Any) -> tuple[str, list[int], bytes] | None: ...
    def forbid_gpu(self) -> AbstractContextManager[None]: ...
    def synchronize(self) -> None: ...
    def prepare_to_release_gpu(self) -> None: ...
    def take_last_error(self) -> str | None: ...
    def reset(self) -> None: ...


def execute(
    program: Program,
    runtime: Runtime,
    lease: Lease,
    *,
    workspace_dir: str,
    progress: Callable[[int], None] | None = None,
    cleanup_failed: Callable[[BaseException], None] | None = None,
    capture_dir: str | None = None,
) -> ProgramOutcome:
    """Run a program and serialize only values selected by return instructions.

    The GPU is claimed on the first instruction that needs it and given up around
    each function declared ``cpu_only`` at ``get_function``. Those calls are
    watched for CUDA access.

    The caller owns creation and cleanup of ``workspace_dir``. Production
    workers use a parent-owned directory so it can be removed if the child dies.
    GPU ownership is retained after final cleanup. The caller must release it
    when reusing the worker, or after a retiring worker process has exited.
    """
    workspace_dir = os.path.abspath(workspace_dir)
    with _working_directory(workspace_dir):
        return _execute_in_workspace(
            program,
            runtime,
            lease,
            progress=progress,
            cleanup_failed=cleanup_failed,
            capture_dir=capture_dir,
            workspace_dir=workspace_dir,
        )


def _execute_in_workspace(
    program: Program,
    runtime: Runtime,
    lease: Lease,
    *,
    progress: Callable[[int], None] | None,
    cleanup_failed: Callable[[BaseException], None] | None,
    capture_dir: str | None,
    workspace_dir: str,
) -> ProgramOutcome:
    env: dict[str, Any] = {}
    results: dict[str, dict[str, Any]] = {}
    encoder = _ValueEncoder(runtime)
    off_gpu = frozenset(
        Ref(instruction.id)
        for instruction in program.instructions
        if isinstance(instruction, GetFunction) and instruction.cpu_only
    )
    error: dict[str, Any] | None = None
    current_index: int | None = None
    current: Instruction | None = None
    captured = CapturedOutput()
    try:
        with _capture_output(_output_limit(program), capture_dir) as captured:
            try:
                for current_index, instruction in enumerate(program.instructions):
                    current = instruction
                    if progress is not None:
                        progress(current_index)
                    _place(instruction, off_gpu, runtime, lease)
                    if isinstance(instruction, FileUpload):
                        _materialize_file(
                            workspace_dir,
                            instruction.path,
                            program.blob_bytes[instruction.blob],
                        )
                        env[instruction.id] = instruction.path
                    elif isinstance(instruction, Upload):
                        if instruction.kind == "module":
                            assert instruction.source is not None
                            env[instruction.id] = runtime.load_module(instruction.source)
                        elif instruction.kind == "bytes":
                            assert instruction.blob is not None
                            env[instruction.id] = program.blob_bytes[instruction.blob]
                        elif instruction.kind == "library":
                            assert instruction.blob is not None
                            env[instruction.id] = runtime.load_library(
                                program.blob_bytes[instruction.blob]
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
                    elif isinstance(instruction, GetFunction):
                        env[instruction.id] = runtime.get_function(
                            env[instruction.module.id], instruction.name
                        )
                    elif isinstance(instruction, Run):
                        env[instruction.id] = _invoke(instruction, env, runtime, off_gpu)
                    elif isinstance(instruction, FileReturn):
                        checkpoint = encoder.checkpoint()
                        try:
                            path = (
                                env[instruction.path.id]
                                if isinstance(instruction.path, Ref)
                                else instruction.path
                            )
                            snapshot = collect(
                                workspace_dir,
                                path,
                                instruction.kind,
                                max_bytes=program.max_return_bytes - encoder.binary_size,
                            )
                            results[instruction.key] = encoder.encode_artifact(snapshot)
                        except Exception as exc:
                            encoder.rollback(checkpoint)
                            raise ExecutionError(
                                "serialization",
                                f"cannot return {instruction.kind}: {type(exc).__name__}: {exc}",
                            ) from exc
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
            except GPUAccessViolation as exc:
                error = _instruction_error(exc.kind, exc.message, current_index, current)
                error["traceback"] = exc.call_stack[-MAX_TRACEBACK_BYTES:]  # at the CUDA call
                error["cuda_call"] = exc.cuda_call
                error["location"] = exc.location
                error["detected_at_ns"] = exc.detected_at_ns  # for the parent; not answered
            except ExecutionError as exc:
                error = _instruction_error(exc.kind, exc.message, current_index, current)
            except Exception as exc:
                error = _instruction_error(
                    "engine", f"{type(exc).__name__}: {exc}", current_index, current
                )
    finally:
        # Handle destruction and reset can touch CUDA, even after a CPU-only tail.
        lease.acquire()
        try:
            # A CUDA fault often surfaces twice: once at an instruction-level sync and
            # again while draining. Keep the first error, tell the owner this process is
            # done, and attribute a fault seen only here to the instruction that ran.
            try:
                runtime.synchronize()
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
                    # A launch-configuration error can sit in CUDA's last-error slot
                    # without failing synchronize. Consume it here so it belongs to this
                    # request; reading clears it, and the context is still healthy.
                    if last_error is not None and error is not None and error["kind"] == "gpu_access":
                        # The violation is the earlier fault, and the parent reads its fields.
                        error["message"] += f"; CUDA also reports {last_error}"
                    elif last_error is not None:
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
        finally:
            try:
                runtime.synchronize()
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


def _invoke(
    instruction: Run, env: dict[str, Any], runtime: Runtime, off_gpu: frozenset[Ref]
) -> Any:
    """Call one ``Run``'s target, in its own frame so the handle and arguments die
    with the instruction. Left in ``execute``'s locals they keep an uploaded
    module's namespace alive past the ``runtime.reset()`` meant to free it."""
    fn = _resolve_fn(instruction.fn, env)
    args = [env[arg.id] if isinstance(arg, Ref) else arg for arg in instruction.args]
    watched = instruction.fn in off_gpu
    with runtime.forbid_gpu() if watched else nullcontext():
        return fn(*args)


@contextmanager
def _working_directory(directory: str) -> Iterator[None]:
    previous = os.open(".", os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.chdir(directory)
        yield
    finally:
        os.fchdir(previous)
        os.close(previous)


def _materialize_file(workspace_dir: str, path: str, data: bytes) -> None:
    """Copy one blob beneath ``workspace_dir`` without following symlinks."""
    try:
        path = normalize_file_path(path)
        if sandbox.active() and path.split("/")[0] == sandbox.PRIVATE:
            raise ValueError(f"{sandbox.PRIVATE!r} is reserved for sandbox runtime files")
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory_fds = [os.open(workspace_dir, directory_flags)]
        try:
            parent_fd = directory_fds[0]
            parts = path.split("/")
            for part in parts[:-1]:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=parent_fd)
                except FileExistsError:
                    pass
                parent_fd = os.open(part, directory_flags, dir_fd=parent_fd)
                directory_fds.append(parent_fd)

            file_fd = os.open(
                parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            with os.fdopen(file_fd, "wb") as file:
                file.write(data)
        finally:
            for directory_fd in reversed(directory_fds):
                os.close(directory_fd)
    except Exception as exc:
        raise ExecutionError(
            "runtime", f"cannot materialize file {path!r}: {type(exc).__name__}: {exc}"
        ) from exc


def _place(
    instruction: Instruction, off_gpu: frozenset[Ref], runtime: Runtime, lease: Lease
) -> None:
    """Hold or drop the GPU for the instruction about to run."""
    if isinstance(instruction, (FileUpload, FileReturn)):
        _drop_gpu(runtime, lease)
        return
    if isinstance(instruction, Run) and instruction.fn in off_gpu:
        # A compile, typically: hand the GPU over so another worker can measure on it.
        _drop_gpu(runtime, lease)
        return
    if isinstance(instruction, Upload) and instruction.kind == "bytes":
        return
    lease.acquire()


def _drop_gpu(runtime: Runtime, lease: Lease) -> None:
    """Drain work and release unused memory before handing off the GPU."""
    if lease.held:
        runtime.prepare_to_release_gpu()
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
        self.binary_size = 0

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

    def encode_artifact(self, value: ReturnedFile | ReturnedFolder) -> dict[str, Any]:
        if isinstance(value, ReturnedFile):
            data = value.read_bytes()
            return {**self._binary_value("file", data), "size": len(data)}
        return {
            "type": "folder",
            "files": {path: self.encode_artifact(file) for path, file in value.files.items()},
            "directories": list(value.directories),
        }

    def checkpoint(self) -> int:
        return len(self.binary_parts)

    def rollback(self, checkpoint: int) -> None:
        """Drop parts registered since ``checkpoint``, keeping numbering contiguous."""
        for name in list(self.binary_parts)[checkpoint:]:
            self.binary_size -= len(self.binary_parts.pop(name))

    def _binary_value(self, value_type: str, data: bytes) -> dict[str, Any]:
        part_name = f"return:{len(self.binary_parts)}"
        self.binary_parts[part_name] = data
        self.binary_size += len(data)
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
def _capture_output(limit_bytes: int, capture_dir: str | None = None) -> Iterator[CapturedOutput]:
    """Capture process-level stdout and stderr for the complete request."""
    captured = CapturedOutput()
    if limit_bytes <= 0:
        yield captured
        return
    sys.stdout.flush()
    sys.stderr.flush()
    with _capture_files(capture_dir) as (stdout_file, stderr_file):
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


@contextmanager
def _capture_files(capture_dir: str | None) -> Iterator[tuple[IO[bytes], IO[bytes]]]:
    """The pair of files stdout and stderr are redirected into, removed after."""
    if capture_dir is None:
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            yield out, err
        return
    # Named by pid: one request runs in a worker at a time, so the pid the parent
    # already holds is enough to find this request's output.
    paths = [os.path.join(capture_dir, f"{os.getpid()}.{suffix}") for suffix in ("out", "err")]
    files = [open(path, "w+b") for path in paths]
    try:
        yield files[0], files[1]
    finally:
        for file, path in zip(files, paths):
            file.close()
            try:
                os.unlink(path)  # so the next request cannot inherit this one's output
            except OSError:
                pass


def read_captured_output(capture_dir: str | None, pid: int | None, limit_bytes: int) -> str:
    """The tail of what a worker had written when it was killed, or ``""``.

    Only what reached the file descriptor is here - a ``print`` still in Python's
    buffer died with the process, but a compiler writing to stderr did reach it.
    Consuming: the worker that would have removed these is the one that died.
    """
    if capture_dir is None or pid is None:
        return ""
    chunks = []
    for suffix in ("out", "err"):
        path = os.path.join(capture_dir, f"{pid}.{suffix}")
        try:
            with open(path, "rb") as file:
                file.seek(0, os.SEEK_END)
                file.seek(max(0, file.tell() - limit_bytes))
                data = file.read()
        except OSError:
            continue
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if data:
            chunks.append(f"[{suffix}] {data.decode('utf-8', errors='replace')}")
    return "\n".join(chunks)


def _stream_over(file: IO[bytes]) -> IO[str]:
    return open(file.fileno(), "w", encoding="utf-8", errors="replace", closefd=False)


def _read_captured(file: IO[bytes], limit_bytes: int) -> tuple[str, bool]:
    file.seek(0)
    data = file.read(limit_bytes + 1)
    return data[:limit_bytes].decode("utf-8", errors="replace"), len(data) > limit_bytes


def _resolve_fn(fn: Ref, env: dict[str, Any]) -> Callable:
    obj = env[fn.id]
    if not callable(obj):
        raise ExecutionError("runtime", f"handle {fn.id!r} is not callable")
    return obj
