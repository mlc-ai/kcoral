from __future__ import annotations

import asyncio
import importlib.util
import inspect
import multiprocessing
import os
import signal
import sys
import time
import traceback as traceback_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .executor import (
    InstructionExecutionError,
    InvalidProgramError,
    execute_program,
)
from .models import (
    ServerConfig,
    ValidatedProgram,
    ValidatedRequest,
    WorkerOutcome,
)
from .serialization import InvalidReturnValue, serialize_return


@dataclass
class WorkerSlot:
    index: int
    device: str
    status: str
    started_at: float


class Scheduler:
    def __init__(self, devices: tuple[str, ...]) -> None:
        now = time.monotonic()
        self.slots = [
            WorkerSlot(index, device, "idle", now)
            for index, device in enumerate(devices)
        ]
        self.available: asyncio.Queue[WorkerSlot] = asyncio.Queue()
        for slot in self.slots:
            self.available.put_nowait(slot)
        self.queue_length = 0
        self._state_lock = asyncio.Lock()

    async def acquire(self) -> WorkerSlot:
        async with self._state_lock:
            self.queue_length += 1
        try:
            slot = await self.available.get()
        finally:
            async with self._state_lock:
                self.queue_length -= 1
        slot.status = "busy"
        return slot

    async def release(self, slot: WorkerSlot, *, restarted: bool = False) -> None:
        if restarted:
            slot.status = "restarting"
            slot.started_at = time.monotonic()
            await asyncio.sleep(0)
        slot.status = "idle"
        self.available.put_nowait(slot)

    def health(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        return [
            {
                "gpu_id": slot.index,
                "status": slot.status,
                "uptime_seconds": max(0.0, now - slot.started_at),
            }
            for slot in self.slots
        ]


async def run_job(
    slot: WorkerSlot,
    job: ValidatedRequest,
    request_id: str,
    work_dir: Path,
    blob_paths: Mapping[str, Path],
    result_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    config: ServerConfig,
) -> WorkerOutcome:
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_runtime_main,
        args=(
            send,
            slot.device,
            job,
            request_id,
            work_dir,
            blob_paths,
            result_dir,
            stdout_path,
            stderr_path,
            config,
        ),
        daemon=False,
    )
    try:
        process.start()
    except Exception as exc:
        receive.close()
        send.close()
        return WorkerOutcome(
            "crash", {"message": f"could not start worker process: {exc}"}
        )
    send.close()
    try:
        return await _monitor_process(process, receive, job, config)
    except asyncio.CancelledError:
        await _stop_process(process, config.worker_termination_grace_seconds)
        receive.close()
        raise


async def _monitor_process(
    process: multiprocessing.Process,
    receive: Any,
    job: ValidatedRequest,
    config: ServerConfig,
) -> WorkerOutcome:
    started = await asyncio.to_thread(receive.poll, 60.0)
    if not started:
        await _stop_process(process, config.worker_termination_grace_seconds)
        receive.close()
        return WorkerOutcome(
            "crash", {"message": "worker process did not become ready"}
        )
    try:
        handshake = receive.recv()
    except EOFError:
        handshake = None
    if handshake != {"kind": "started"}:
        if process.is_alive():
            _terminate_process_tree(process, signal.SIGTERM)
        await asyncio.to_thread(process.join)
        receive.close()
        return WorkerOutcome(
            "crash", {"message": "worker process exited during startup"}
        )
    ready = await asyncio.to_thread(receive.poll, job.timeout_seconds)
    if not ready:
        await _stop_process(process, config.worker_termination_grace_seconds)
        receive.close()
        return WorkerOutcome(
            "timeout",
            {"message": f"execution exceeded {job.timeout_seconds:g} seconds"},
        )
    try:
        payload = receive.recv()
    except EOFError:
        payload = {
            "kind": "crash",
            "metadata": {"message": "worker process exited without a result"},
        }
    finally:
        receive.close()
        await asyncio.to_thread(process.join)
    if process.exitcode not in (0, None) and payload.get("kind") not in {
        "execution_failed",
        "invalid_program",
        "invalid_return_value",
    }:
        return WorkerOutcome(
            "crash",
            {"message": f"worker process exited with status {process.exitcode}"},
        )
    return WorkerOutcome(
        payload["kind"], payload.get("metadata", {}), payload.get("binaries", [])
    )


def _runtime_main(
    connection: Any,
    device: str,
    job: ValidatedRequest,
    request_id: str,
    work_dir: Path,
    blob_paths: Mapping[str, Path],
    result_dir: Path,
    stdout_path: Path,
    stderr_path: Path,
    config: ServerConfig,
) -> None:
    if hasattr(os, "setsid"):
        os.setsid()
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["GPU_SERVER_REQUEST_ID"] = request_id
    os.environ["GPU_SERVER_WORK_DIR"] = str(work_dir.resolve())
    os.chdir(work_dir)
    sys.path.insert(0, str(work_dir.resolve()))

    original_stdout = os.dup(1)
    original_stderr = os.dup(2)
    stdout_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(stdout_fd, 1)
    os.dup2(stderr_fd, 2)
    os.close(stdout_fd)
    os.close(stderr_fd)

    connection.send({"kind": "started"})
    started = time.perf_counter()
    payload: dict[str, Any]
    try:
        if isinstance(job, ValidatedProgram):
            value = execute_program(job, blob_paths, work_dir, request_id)
        else:
            entry_path = work_dir.joinpath(*job.entry.file.split("/"))
            module_name = f"_benchmark_server_entry_{request_id.replace('-', '_')}"
            spec = importlib.util.spec_from_file_location(module_name, entry_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot import entry file {job.entry.file}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            function = getattr(module, job.entry.function, None)
            if not callable(function):
                raise RuntimeError(
                    f"entry function {job.entry.function!r} is not callable"
                )
            signature = inspect.signature(function)
            try:
                signature.bind()
            except TypeError as exc:
                raise RuntimeError(
                    f"entry function {job.entry.function!r} must accept no arguments"
                ) from exc
            value = function()
        tree, binaries = serialize_return(value, result_dir, config)
        elapsed = (time.perf_counter() - started) * 1000
        payload = {
            "kind": "ok",
            "metadata": {"return": tree, "elapsed_ms": elapsed},
            "binaries": binaries,
        }
    except InvalidProgramError as exc:
        payload = {
            "kind": "invalid_program",
            "metadata": {
                "message": str(exc),
                "instruction_index": exc.instruction_index,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            },
        }
    except InstructionExecutionError as exc:
        payload = {
            "kind": "execution_failed",
            "metadata": {
                "message": f"instruction {exc.instruction_index} failed: {exc}",
                "instruction_index": exc.instruction_index,
                "traceback": traceback_module.format_exc(),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            },
        }
    except InvalidReturnValue as exc:
        payload = {
            "kind": "invalid_return_value",
            "metadata": {
                "message": str(exc),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            },
        }
    except BaseException as exc:
        execution_name = (
            "instruction program"
            if isinstance(job, ValidatedProgram)
            else f"{job.entry.function}()"
        )
        payload = {
            "kind": "execution_failed",
            "metadata": {
                "message": f"{execution_name} raised {type(exc).__name__}: {exc}",
                "traceback": traceback_module.format_exc(),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            },
        }
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(original_stdout, 1)
        os.dup2(original_stderr, 2)
        os.close(original_stdout)
        os.close(original_stderr)
    try:
        connection.send(payload)
    finally:
        connection.close()


def _terminate_process_tree(
    process: multiprocessing.Process, sig: signal.Signals
) -> None:
    if process.pid is None:
        return
    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, sig)
            return
        except ProcessLookupError:
            pass
    if sig == signal.SIGKILL:
        process.kill()
    else:
        process.terminate()


async def _stop_process(process: multiprocessing.Process, grace_seconds: float) -> None:
    if not process.is_alive():
        await asyncio.to_thread(process.join)
        return
    _terminate_process_tree(process, signal.SIGTERM)
    await asyncio.to_thread(process.join, grace_seconds)
    if process.is_alive():
        _terminate_process_tree(process, signal.SIGKILL)
        await asyncio.to_thread(process.join)
