"""A child worker process and its parent-side lifecycle handle.

The front-end never touches the GPU - it dispatches a Program to a worker over a
pipe and reads back one execution outcome. A worker runs one program at a time,
clears its per-request state after each (via the Runtime), and is replaced by a
fresh process once it has served ``max_requests`` of them. The worker's
`main` is the child entry point; :class:`Worker` is the parent-side handle with
crash/timeout kill + respawn.

Worker results carry the program outcome and an optional retirement reason.
The parent handles timeout and crash recovery when no answer arrives.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import nvml
from . import sandbox as sandboxing
from .engine import execute, read_captured_output
from .events import EventLogger
from .lease import GPULeases, LeaseClient, NoopLease, NoopLeases

_WORKER_PIPE_FAILURES = (EOFError, ConnectionResetError, BrokenPipeError, OSError)

# Severity of worker lifecycle events.
RETIRE_REASON_LEVEL = {
    "request_limit": "INFO",
    "poisoned_context": "WARNING",
    "timeout": "WARNING",
    "crashed": "ERROR",
    "sandbox_cleanup": "WARNING",
}


@dataclass
class WorkerResult:
    execution: object
    lease_wait_ms: float
    lease_held_ms: float
    retire_reason: str | None = None

    @property
    def finish_reason(self) -> str:
        return (
            "completed"
            if getattr(self.execution, "status", None) == "COMPLETED"
            else "program_failed"
        )


_NO_EVENTS = EventLogger(None)  # a shared no-op, for a handle given no log
_OUTPUT_TAIL_BYTES = 4096  # of a killed worker's output, kept on its log record


def worker_main(
    device: str | None,
    conn,
    runtime_factory: Callable,
    max_requests: int,
    capture_dir: str | None = None,
    isolated: bool = False,
) -> None:
    """Child entry point. Optionally pins a GPU, then serves programs.

    For a GPU worker, ``device`` is the ``CUDA_VISIBLE_DEVICES`` value the parent
    resolved, a UUID where NVML could name one. A CPU worker receives ``None``
    and leaves the environment untouched.
    """
    # Lead a new process group, so the parent can clean up anything the submitted
    # code spawned (grandchildren included) with one killpg.
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    # Select the GPU before the Runtime imports torch/tvm, so it sees one device.
    # Any value the server was launched with selects nothing here, so it goes.
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    try:
        # A factory may explicitly move dependency imports and other host-only
        # setup into ``prepare``.  Signal the parent only after that work is done;
        # it grants GPU initialization under the per-device lease below.
        prepare = getattr(runtime_factory, "prepare", None)
        prepared_factory = prepare() if prepare is not None else runtime_factory
    except Exception as exc:  # host-only preparation failed — report and exit
        conn.send({"__error__": f"runtime preparation failed: {type(exc).__name__}: {exc}"})
        return
    conn.send({"__startup__": "prepared"})
    try:
        startup = conn.recv()
    except EOFError:
        return
    if startup != {"__startup__": "initialize"}:
        return
    try:
        runtime = prepared_factory()
        described = {
            "target": runtime.target(),
            "versions": runtime.versions(),
            "device_uuid": runtime.device_uuid(),
        }
    except Exception as exc:  # runtime init failed — report and exit
        conn.send({"__error__": f"runtime init failed: {type(exc).__name__}: {exc}"})
        return
    # A GPU process describes its visible card; a CPU process reports no target.
    ready = {"__ready__": described}
    if isolated:
        ready["__pid__"] = os.getpid()
    conn.send(ready)
    lease = LeaseClient(conn) if device is not None else NoopLease()
    requests_served = 0
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:  # shutdown signal
            return
        program, workspace_dir = message
        cleanup_error: BaseException | None = None
        request_state = sandboxing.RequestState() if isolated else None

        def mark_cleanup_failed(exc: BaseException) -> None:
            nonlocal cleanup_error
            if cleanup_error is None:
                cleanup_error = exc

        outcome = execute(
            program,
            runtime,
            lease=lease,
            progress=lambda index: conn.send({"__instruction__": index}),
            cleanup_failed=mark_cleanup_failed,
            capture_dir=capture_dir,
            workspace_dir=workspace_dir,
        )
        sandbox_cleanup_error = None
        if request_state is not None:
            try:
                request_state.finish()
            except Exception as exc:
                sandbox_cleanup_error = f"{type(exc).__name__}: {exc}"
        requests_served += 1
        # None means this worker can serve another program.
        retire_reason: str | None = None
        if sandbox_cleanup_error is not None:
            retire_reason = "sandbox_cleanup"
        elif cleanup_error is not None:
            retire_reason = "poisoned_context"
        elif max_requests and requests_served >= max_requests:
            # A fresh process gives every request the same context and allocator
            # state, even where native code left no detectable sticky error.
            retire_reason = "request_limit"
        # Send the outcome before exiting; the pool schedules replacement.
        response = {"__outcome__": outcome, "__retire_reason__": retire_reason}
        if sandbox_cleanup_error is not None:
            response["__sandbox_cleanup_error__"] = sandbox_cleanup_error
        conn.send(response)
        if retire_reason is not None:
            return


class WorkerCrashed(Exception):
    # Attribution filled in by the pool when the failure happened under submit.
    worker_id: str | None = None
    gpu_id: int | None = None
    queue_ms: float | None = None
    elapsed_ms: float | None = None
    instruction_index: int | None = None
    exitcode: int | None = None
    output_tail: str = ""
    lease_wait_ms: float = 0.0
    lease_held_ms: float = 0.0


class WorkerTimeout(Exception):
    worker_id: str | None = None
    gpu_id: int | None = None
    queue_ms: float | None = None
    elapsed_ms: float | None = None
    output_tail: str = ""


class Worker:
    """Parent-side handle to one CPU or GPU worker process."""

    request_id: str | None = None  # set by the pool while a request is served

    # Identity for the log. ``worker_id`` names a seat on a GPU and outlives the
    # processes that sit in it; ``generation`` and ``pid`` say which one does now.
    worker_id: str = ""
    generation: int = 0
    pid: int | None = None
    _events: EventLogger = _NO_EVENTS
    _capture_dir: str | None = None

    def __init__(
        self,
        gpu_id: int | None,
        runtime_factory: Callable,
        spawn_timeout: float = 60.0,
        termination_grace_seconds: float = 5.0,
        max_requests: int = 1,
        index: int = 0,
        events: EventLogger | None = None,
        capture_dir: str | None = None,
        sandbox: str = "bubblewrap",
        sandbox_readonly_paths: tuple[Path, ...] = (),
    ) -> None:
        self._lifecycle_lock = threading.RLock()
        self._closing = threading.Event()
        self.gpu_id = gpu_id
        self.worker_id = f"cpu/w{index}" if gpu_id is None else f"gpu{gpu_id}/w{index}"
        if events is not None:
            self._events = events
        self._expected_uuid = nvml.device_uuid(gpu_id) if gpu_id is not None else None
        self._device = (self._expected_uuid or str(gpu_id)) if gpu_id is not None else None
        self._factory = runtime_factory
        self._spawn_timeout = spawn_timeout
        self._termination_grace_seconds = termination_grace_seconds
        self._max_requests = max_requests
        self._capture_dir = capture_dir
        self._sandbox_mode = sandbox
        self._sandbox_readonly_paths = sandbox_readonly_paths
        self._sandbox: sandboxing.Sandbox | None = None
        self._capture_pid: int | None = None
        if sandbox not in ("none", "bubblewrap"):
            raise ValueError(f"unknown worker sandbox: {sandbox!r}")
        self._ctx = mp.get_context("spawn")  # 'spawn' — 'fork' is unsafe with CUDA
        self._spawn()

    def _spawn(self) -> None:
        """Start and fully initialize a worker before it becomes available."""
        try:
            self._start_process()
            self._initialize_process()
        except Exception as exc:
            self._log_failure("startup", exc)
            raise

    def _log_failure(self, phase: str, exc: BaseException) -> None:
        self._events.emit(
            "worker_failed",
            level="ERROR",
            worker_id=self.worker_id,
            gpu_id=self.gpu_id,
            generation=self.generation,
            phase=phase,
            error=f"{type(exc).__name__}: {exc}",
        )

    def _start_process(self) -> None:
        """Spawn a worker and finish its explicitly host-only preparation."""
        with self._lifecycle_lock:
            if self._closing.is_set():
                raise WorkerCrashed("worker is shutting down")
            self.started_at = time.monotonic()
            parent, child = self._ctx.Pipe()
            self._conn = parent
            try:
                if self._sandbox_mode == "bubblewrap":
                    self._sandbox = sandboxing.Sandbox(self._sandbox_readonly_paths)
                    self._proc = sandboxing.SandboxProcess(self._sandbox, child, self.gpu_id)
                    parent.send((self._device, self._factory, self._max_requests))
                else:
                    self._proc = self._ctx.Process(
                        target=worker_main,
                        args=(
                            self._device,
                            child,
                            self._factory,
                            self._max_requests,
                            self._capture_dir,
                        ),
                        daemon=True,
                    )
                    self._proc.start()
            except BaseException:
                self._kill()
                raise
            finally:
                child.close()
            self.generation += 1
            self.pid = self._proc.pid
        msg = self._await_startup_message("prepare")
        if not (isinstance(msg, dict) and msg.get("__startup__") == "prepared"):
            self._kill()
            raise WorkerCrashed(f"worker preparation error: {msg}")

    def _initialize_process(self) -> None:
        """Create the worker runtime; GPU callers serialize this phase."""
        try:
            with self._lifecycle_lock:
                if self._closing.is_set():
                    raise WorkerCrashed("worker is shutting down")
                self._conn.send({"__startup__": "initialize"})
        except _WORKER_PIPE_FAILURES as exc:
            self._kill()
            raise WorkerCrashed(f"{self._description()} failed before initialization") from exc
        msg = self._await_startup_message("initialize")
        if not (isinstance(msg, dict) and "__ready__" in msg):
            self._kill()
            raise WorkerCrashed(f"worker init error: {msg}")
        described = msg["__ready__"]
        self._capture_pid = msg.get("__pid__", self.pid)
        self.target: dict[str, str] = described["target"]
        self.versions: dict[str, str] = described["versions"]
        self.device_uuid: str | None = described.get("device_uuid")
        self._require_expected_device()
        with self._lifecycle_lock:
            if self._closing.is_set():
                raise WorkerCrashed("worker shutdown interrupted initialization")
            self._events.emit(
                "worker_ready",
                worker_id=self.worker_id,
                gpu_id=self.gpu_id,
                generation=self.generation,
                pid=self.pid,
                device=self._device,
                arch=self.target.get("arch"),
            )

    def _require_expected_device(self) -> None:
        """Refuse a worker that came up on another card: it would measure someone
        else's GPU under the id it was asked for."""
        if self.gpu_id is None:
            return
        expected = nvml.uuid_key(self._expected_uuid)
        reported = nvml.uuid_key(self.device_uuid)
        if expected is None or reported is None or expected == reported:
            return
        self._kill()
        raise WorkerCrashed(
            f"worker for GPU {self.gpu_id} came up on device {self.device_uuid}, "
            f"not the requested {self._expected_uuid}"
        )

    def _await_startup_message(self, phase: str):
        try:
            deadline = time.monotonic() + self._spawn_timeout
            ready = False
            while time.monotonic() < deadline:
                if self._closing.is_set():
                    raise WorkerCrashed("worker shutdown interrupted startup")
                if self._conn.poll(min(0.05, max(0, deadline - time.monotonic()))):
                    ready = True
                    break
            msg = self._conn.recv() if ready else None
        except _WORKER_PIPE_FAILURES as exc:
            self._kill()
            details = (
                self._proc.output_tail()
                if isinstance(self._proc, sandboxing.SandboxProcess)
                else ""
            )
            raise WorkerCrashed(f"{self._description()} failed during {phase}: {details}") from exc
        if not ready:
            self._kill()
            raise WorkerCrashed(f"{self._description()} timed out during {phase}")
        return msg

    def run(self, program, timeout: float, leases: GPULeases | NoopLeases) -> WorkerResult:
        """Run a program, servicing its lease requests; kill+respawn on timeout or
        crash, then re-raise. A poisoned context is respawned after preserving its
        outcome. Returns the execution outcome, lease timings, and retirement reason.

        The parent owns the workspace so it is removed even when the child is
        killed before its own cleanup handlers can run.
        """
        if self._sandbox_mode == "bubblewrap":
            sandbox = self._sandbox
            if sandbox is None:
                self._abandon_and_respawn(leases, "crashed")
                sandbox = self._sandbox
            assert sandbox is not None
            try:
                sandbox.prepare()
            except OSError as exc:
                self._abandon_and_respawn(leases, "sandbox_cleanup")
                raise WorkerCrashed(f"cannot prepare sandbox workspace: {exc}") from exc
            result = self._run_in_workspace(program, timeout, leases, sandboxing.WORKSPACE)
            if result.retire_reason is not None:
                # Stop surviving tasks before the parent touches their files.
                self._kill()
            else:
                try:
                    sandbox.prepare()
                except OSError:
                    self._kill()
                    result.retire_reason = "sandbox_cleanup"
            return result
        with tempfile.TemporaryDirectory(prefix="kcoral-program-") as workspace_dir:
            return self._run_in_workspace(program, timeout, leases, workspace_dir)

    def _run_in_workspace(
        self,
        program,
        timeout: float,
        leases: GPULeases | NoopLeases,
        workspace_dir: str,
    ) -> WorkerResult:
        """Worker protocol loop for a program whose workspace already exists.

        The deadline covers only the worker's own work - time blocked on a lease
        another worker holds is not counted, or ``timeout_seconds`` would mean
        different things at different loads.
        """
        remaining = timeout
        lease_wait_ms = 0.0
        lease_held_ms = 0.0
        held_since: float | None = None
        instruction_index: int | None = None
        try:
            self._conn.send((program, workspace_dir))
            while True:
                waited_from = time.monotonic()
                if not self._conn.poll(remaining):  # no answer by the deadline -> hung
                    timed_out = WorkerTimeout(
                        f"program exceeded {timeout}s in {self._description()}"
                    )
                    timed_out.worker_id = self.worker_id
                    timed_out.output_tail = self._output_tail()  # before the kill removes it
                    self._abandon_and_respawn(leases, "timeout")
                    raise timed_out
                remaining -= time.monotonic() - waited_from
                message = self._conn.recv()
                if isinstance(message, dict) and "__instruction__" in message:
                    instruction_index = message["__instruction__"]
                    continue
                if isinstance(message, dict) and "__outcome__" in message:
                    if cleanup_error := message.get("__sandbox_cleanup_error__"):
                        self._log_failure("sandbox_cleanup", RuntimeError(cleanup_error))
                    if held_since is not None:
                        lease_held_ms += (time.monotonic() - held_since) * 1000
                    leases.release(self.gpu_id, self)  # no-op if the engine already did
                    outcome = message["__outcome__"]
                    return WorkerResult(
                        outcome, lease_wait_ms, lease_held_ms, message["__retire_reason__"]
                    )
                if not (isinstance(message, dict) and "__lease__" in message):
                    # Nothing this protocol defines. Hand it up rather than read
                    # it as lease traffic: the front-end rejects what is not a
                    # ProgramOutcome, which beats failing on the message's shape.
                    if held_since is not None:
                        lease_held_ms += (time.monotonic() - held_since) * 1000
                    leases.release(self.gpu_id, self)
                    return WorkerResult(message, lease_wait_ms, lease_held_ms)
                if message["__lease__"] == "acquire":
                    # The child is blocked until we answer, so this may wait freely.
                    lease_wait_ms += leases.acquire(self.gpu_id, self)
                    held_since = time.monotonic()
                    self._conn.send({"__lease__": "granted"})
                else:
                    # A release; the child did not wait for an answer and has moved on.
                    if held_since is not None:
                        lease_held_ms += (time.monotonic() - held_since) * 1000
                        held_since = None
                    leases.release(self.gpu_id, self)
        except _WORKER_PIPE_FAILURES as exc:
            # send, poll, recv, and lease replies can each be the first place a
            # dead worker is observed, depending on pipe timing and platform.
            if held_since is not None:
                lease_held_ms += (time.monotonic() - held_since) * 1000
            try:
                self._proc.join(timeout=0.1)
            except Exception:
                pass
            crash = WorkerCrashed(f"{self._description()} pipe failed: {exc}")
            crash.worker_id = self.worker_id
            crash.instruction_index = instruction_index
            crash.exitcode = self._proc.exitcode
            crash.lease_wait_ms = lease_wait_ms
            crash.lease_held_ms = lease_held_ms
            crash.output_tail = self._output_tail()
            self._abandon_and_respawn(leases, "crashed")
            raise crash from exc

    def _output_tail(self) -> str:
        """What the worker had printed when it stopped answering. Its own copy
        travels back in the outcome, but a worker that never answers has none."""
        if self._sandbox is not None:
            return read_captured_output(
                str(self._sandbox.workspace / sandboxing.PRIVATE / "output"),
                self._capture_pid,
                _OUTPUT_TAIL_BYTES,
            )
        return read_captured_output(self._capture_dir, self.pid, _OUTPUT_TAIL_BYTES)

    def replace(self, leases: GPULeases | NoopLeases, reason: str) -> None:
        """Swap in a fresh process, normally on the pool's replacement thread."""
        self._abandon_and_respawn(leases, reason)

    def _abandon_and_respawn(self, leases: GPULeases | NoopLeases, reason: str) -> None:
        """Replace a worker without letting its lifecycle touch an in-use GPU.

        Kill before abandoning the lease: a timed-out kernel may still be running
        until termination completes. Then prepare off-lease but initialize under it,
        or a replacement racing a waiter creates its context alongside that
        waiter's program.
        """
        self._events.emit(
            "worker_retired",
            level=RETIRE_REASON_LEVEL.get(reason, "INFO"),
            worker_id=self.worker_id,
            gpu_id=self.gpu_id,
            generation=self.generation,
            pid=self.pid,
            reason=reason,
        )
        self._kill()
        leases.abandon(self.gpu_id, self)
        if getattr(self, "_closing", None) is not None and self._closing.is_set():
            return
        try:
            self._start_process()
            leases.acquire(self.gpu_id, self)
            try:
                self._initialize_process()
            finally:
                leases.release(self.gpu_id, self)
        except Exception as exc:
            if getattr(self, "_closing", None) is not None and self._closing.is_set():
                return
            self._log_failure("respawn", exc)
            raise

    def _description(self) -> str:
        return "CPU worker" if self.gpu_id is None else f"worker on GPU {self.gpu_id}"

    def begin_shutdown(self) -> None:
        # Initialization waits must stay outside this lock.
        with self._lifecycle_lock:
            self._closing.set()

    def _kill(self) -> None:
        process = getattr(self, "_proc", None)
        sandbox = getattr(self, "_sandbox", None)
        try:
            if process is not None:
                _terminate_process_tree(process, self._termination_grace_seconds)
        except Exception:
            pass
        if sandbox is not None and process is not None and process.is_alive():
            # Never clear a workspace while a surviving process can still use
            # it, or proceed to a new generation after failed termination.
            raise WorkerCrashed("sandbox worker survived termination")
        try:
            self._conn.close()
        except Exception:
            pass
        if sandbox is not None:
            self._sandbox = None
            try:
                sandbox.close()
            except OSError as exc:
                self._log_failure("sandbox_cleanup", exc)

    def close(self) -> None:
        try:
            self._conn.send(None)
        except Exception:
            pass
        self._kill()


def _terminate_process_tree(process, grace_seconds: float) -> None:
    """Terminate the worker and everything the submitted code spawned.

    The worker leads a process group (see :func:`worker_main`), so a killpg
    covers grandchildren a plain ``Process.kill()`` would orphan. SIGTERM first,
    ``grace_seconds`` for a clean exit, then SIGKILL the survivors.
    """
    if process.pid is None:
        return
    signaled_group = _signal_process_group(process.pid, signal.SIGTERM)
    if not signaled_group:
        process.terminate()
    process.join(timeout=grace_seconds)
    if signaled_group:
        _signal_process_group(process.pid, signal.SIGKILL)
    if process.is_alive():
        process.kill()
    process.join(timeout=5)


def _signal_process_group(process_group_id: int, sig: signal.Signals) -> bool:
    """Signal a process group; False when unsupported or the group is gone."""
    if not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(process_group_id, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # the group exists but a member is not signalable


def _sandbox_main() -> None:
    """Private entry point for ``python -m kcoral.worker`` inside bubblewrap."""
    from multiprocessing.connection import Connection

    conn = Connection(os.dup(0))
    with open(os.devnull, "rb") as null:
        os.dup2(null.fileno(), 0)
    sandboxing.activate()
    # Both sides are trusted KCoral processes and use the existing protocol.
    device, factory, max_requests = conn.recv()
    worker_main(
        device,
        conn,
        factory,
        max_requests,
        capture_dir=f"{sandboxing.WORKSPACE}/{sandboxing.PRIVATE}/output",
        isolated=True,
    )


if __name__ == "__main__":
    _sandbox_main()
