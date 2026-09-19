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

import ctypes
import errno
import multiprocessing as mp
import os
import platform
import signal
import sys
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
    """Child entry point. Binds the assigned GPUs, then serves programs.

    For a GPU worker, ``device`` is the ``CUDA_VISIBLE_DEVICES`` value the parent
    resolved, using comma-separated UUIDs where available. A CPU worker receives ``None``
    and leaves the environment untouched.
    """
    # Lead a new process group, so the parent can clean up anything the submitted
    # code spawned (grandchildren included) with one killpg.
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    # Select all assigned GPUs before the runtime imports torch/tvm.
    # Any value the server was launched with selects nothing here, so it goes.
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        for name in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
            "TORCHELASTIC_RUN_ID",
        ):
            os.environ.pop(name, None)
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
        elif (max_requests and requests_served >= max_requests) or _children(os.getpid()):
            # A fresh process gives every request the same context and allocator
            # state, even where native code left no detectable sticky error.
            retire_reason = "request_limit"
        # Retiring workers retain ownership until the parent waits for context teardown.
        if retire_reason is None:
            lease.release()
        response = {"__outcome__": outcome, "__retire_reason__": retire_reason}
        if sandbox_cleanup_error is not None:
            response["__sandbox_cleanup_error__"] = sandbox_cleanup_error
        conn.send(response)
        if retire_reason is not None:
            return


class WorkerCleanupError(RuntimeError):
    """The supervisor exited without confirming that the process tree is gone."""


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
    """One worker bound to a CPU, one GPU, or a fixed GPU set."""

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
        gpu_id: int | tuple[int, ...] | None,
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
        if isinstance(gpu_id, tuple) and sys.platform != "linux":
            raise ValueError("explicit GPU sets require Linux process supervision")
        self._lifecycle_lock = threading.RLock()
        self._closing = threading.Event()
        self._gpu_ids = (
            gpu_id if isinstance(gpu_id, tuple) else (() if gpu_id is None else (gpu_id,))
        )
        self.gpu_id = self._gpu_ids[0] if self._gpu_ids else None
        self.worker_id = f"cpu/w{index}" if self.gpu_id is None else f"gpu{self.gpu_id}/w{index}"
        if events is not None:
            self._events = events
        self._expected_uuid = nvml.device_uuid(self.gpu_id) if self.gpu_id is not None else None
        self._device = (
            ",".join(nvml.device_uuid(gpu) or str(gpu) for gpu in self.gpu_ids)
            if self.gpu_ids
            else None
        )
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

    @property
    def gpu_ids(self) -> tuple[int, ...]:
        return getattr(self, "_gpu_ids", () if self.gpu_id is None else (self.gpu_id,))

    @property
    def _lease_key(self):
        return self.gpu_ids if len(self.gpu_ids) > 1 else self.gpu_id

    def _spawn(self) -> None:
        """Start and fully initialize a worker before it becomes available."""
        try:
            self._start_process()
            self._initialize_process()
        except Exception as exc:
            self._kill()
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
            self._control = None
            self._exit_status = None
            try:
                if self._sandbox_mode == "bubblewrap":
                    self._sandbox = sandboxing.Sandbox(self._sandbox_readonly_paths)
                    self._proc = sandboxing.SandboxProcess(self._sandbox, child, self._lease_key)
                    parent.send((self._device, self._factory, self._max_requests))
                else:
                    control, supervisor = self._ctx.Pipe()
                    self._conn, self._control = parent, control
                    self._proc = self._ctx.Process(
                        target=_supervise_worker,
                        args=(
                            supervisor,
                            child,
                            self._device,
                            self._factory,
                            self._max_requests,
                            self._capture_dir,
                            self._termination_grace_seconds,
                        ),
                    )
                    self._proc.start()
                    supervisor.close()
            except BaseException:
                self._kill()
                raise
            finally:
                child.close()
            self.generation += 1
            self.pid = self._proc.pid
        if self._control is not None:
            if not control.poll(self._spawn_timeout):
                self._kill()
                raise WorkerCrashed("worker supervisor timed out during startup")
            status = control.recv()
            if "pid" not in status:
                raise WorkerCleanupError(f"worker supervisor startup failed: {status}")
            self.pid = status["pid"]
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
            if result.retire_reason is None:
                try:
                    sandbox.prepare()
                except OSError:
                    wait_ms, held_ms = self._kill_with_gpu_lease(leases)
                    result.lease_wait_ms += wait_ms
                    result.lease_held_ms += held_ms
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
                    outcome = message["__outcome__"]
                    if (
                        message["__retire_reason__"] is not None
                        and getattr(self, "_control", None) is not None
                    ):
                        # An outcome precedes Python/CUDA finalizers. Keep them within
                        # the execution deadline and wait for the complete process tree.
                        status = self._wait_for_exit(max(0, remaining))
                        if status is None:
                            self._abandon_and_respawn(leases, "timeout")
                            raise WorkerTimeout("worker teardown exceeded the execution timeout")
                        if status["orphaned"] and outcome.status == "COMPLETED":
                            outcome.status = "FAILED"
                            instruction = (
                                program.instructions[instruction_index]
                                if instruction_index is not None
                                else None
                            )
                            outcome.error = {
                                "kind": "runtime",
                                "message": (
                                    "program left background processes running; "
                                    "they were terminated"
                                ),
                                "instruction_index": instruction_index,
                                "instruction_id": getattr(instruction, "id", None),
                                "instruction_op": getattr(instruction, "op", None),
                                "traceback": "",
                            }
                    if message["__retire_reason__"] is not None:
                        self._kill()
                    if held_since is not None:
                        lease_held_ms += (time.monotonic() - held_since) * 1000
                    leases.release(self._lease_key, self)
                    return WorkerResult(
                        outcome, lease_wait_ms, lease_held_ms, message["__retire_reason__"]
                    )
                if not (isinstance(message, dict) and "__lease__" in message):
                    # Nothing this protocol defines. Hand it up rather than read
                    # it as lease traffic: the front-end rejects what is not a
                    # ProgramOutcome, which beats failing on the message's shape.
                    if held_since is not None:
                        lease_held_ms += (time.monotonic() - held_since) * 1000
                    leases.release(self._lease_key, self)
                    return WorkerResult(message, lease_wait_ms, lease_held_ms)
                if message["__lease__"] == "acquire":
                    # The child is blocked until we answer, so this may wait freely.
                    lease_wait_ms += leases.acquire(self._lease_key, self)
                    held_since = time.monotonic()
                    self._conn.send({"__lease__": "granted"})
                else:
                    # A release; the child did not wait for an answer and has moved on.
                    if held_since is not None:
                        lease_held_ms += (time.monotonic() - held_since) * 1000
                        held_since = None
                    leases.release(self._lease_key, self)
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
            crash.lease_wait_ms = lease_wait_ms
            crash.lease_held_ms = lease_held_ms
            crash.output_tail = self._output_tail()
            if getattr(self, "_control", None) is not None:
                self._kill_with_gpu_lease(leases)
            crash.exitcode = (
                self._exit_status["exitcode"]
                if getattr(self, "_exit_status", None) is not None
                else self._proc.exitcode
            )
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
        self._kill_with_gpu_lease(leases)
        leases.abandon(self._lease_key, self)
        if getattr(self, "_closing", None) is not None and self._closing.is_set():
            return
        try:
            self._start_process()
            with leases.initialization(self.gpu_id):
                if getattr(self, "_closing", None) is not None and self._closing.is_set():
                    return
                leases.acquire(self._lease_key, self)
                try:
                    self._initialize_process()
                finally:
                    leases.release(self._lease_key, self)
        except Exception as exc:
            if getattr(self, "_closing", None) is not None and self._closing.is_set():
                return
            self._log_failure("respawn", exc)
            raise

    def _kill_with_gpu_lease(self, leases: GPULeases | NoopLeases) -> tuple[float, float]:
        """Hold the GPU lease through context teardown, even after CPU-stage timeouts."""
        process = getattr(self, "_proc", None)
        if self.gpu_id is None or process is None or not process.is_alive():
            self._kill()
            return 0.0, 0.0
        wait_ms = leases.acquire(self._lease_key, self)
        started = time.monotonic()
        self._kill()
        held_ms = (time.monotonic() - started) * 1000
        leases.release(self._lease_key, self)
        return wait_ms, held_ms

    def _description(self) -> str:
        return "CPU worker" if self.gpu_id is None else f"worker on GPUs {self.gpu_ids}"

    def begin_shutdown(self) -> None:
        # Initialization waits must stay outside this lock.
        with self._lifecycle_lock:
            self._closing.set()

    def _wait_for_exit(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self._exit_status is None:
            try:
                remaining = None if deadline is None else max(0, deadline - time.monotonic())
                if not self._control.poll(remaining):
                    return None
                status = self._control.recv()
            except _WORKER_PIPE_FAILURES as exc:
                raise WorkerCleanupError("worker supervisor lost before cleanup completed") from exc
            if "pid" in status:
                self.pid = status["pid"]
                continue
            if "exitcode" not in status:
                raise WorkerCleanupError(f"worker cleanup failed: {status}")
            self._exit_status = status
            self._proc.join()
        return self._exit_status

    def _kill(self) -> None:
        if getattr(self, "_control", None) is not None:
            if self._proc.pid is not None and self._exit_status is None:
                try:
                    self._control.send("terminate")
                except _WORKER_PIPE_FAILURES:
                    pass  # A completed supervisor may already have sent its acknowledgement.
                self._wait_for_exit()
            self._conn.close()
            self._control.close()
            return
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
            raise WorkerCleanupError("sandbox worker survived termination")
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


def _children(pid: int) -> set[int]:
    """Read every thread: native libraries can launch children off the main thread."""
    children: set[int] = set()
    for task in Path(f"/proc/{pid}/task").glob("*"):
        try:
            children.update(map(int, (task / "children").read_text().split()))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return children


def _descendants(pid: int) -> set[int]:
    found: set[int] = set()
    pending = [pid]
    while pending:
        for child in _children(pending.pop()) - found:
            found.add(child)
            pending.append(child)
    return found


def _pidfd_open(pid: int) -> int:
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(pid)
    # Linux x86-64 and AArch64 share these syscall numbers. This fallback also
    # works when Python or libc was built against headers predating pidfds.
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("this platform needs Python with pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)  # pidfd_open
    if fd < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open failed")
    return fd


def _pidfd_signal(fd: int, sig: int) -> None:
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        native(fd, sig)
        return
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("this platform needs Python with pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(424, fd, sig, None, 0) != 0:  # pidfd_send_signal
        raise OSError(ctypes.get_errno(), "pidfd_send_signal failed")


def _signal(pid: int, sig: int) -> None:
    try:
        fd = _pidfd_open(pid)
        try:
            # Pin the identity before checking ancestry, so PID reuse cannot
            # direct a cleanup signal at another request's process.
            if pid in _descendants(os.getpid()):
                _pidfd_signal(fd, sig)
        finally:
            os.close(fd)
    except OSError as exc:
        if exc.errno != errno.ESRCH:
            raise


def _reap() -> None:
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                return
        except ChildProcessError:
            return


def _drain_children(runner, grace: float) -> bool:
    """Terminate all descendants and reap them; never acknowledge a live tree."""
    had_children = bool(_descendants(os.getpid()) - {runner.pid})
    deadline = time.monotonic() + grace
    signaled: set[int] = set()
    while True:
        # multiprocessing owns waitpid for the immediate interpreter.
        runner.join(timeout=0)
        if runner.exitcode is not None:
            _reap()
        children = _descendants(os.getpid())
        if not children:
            runner.join(timeout=0)
            return had_children
        for pid in children:
            if time.monotonic() >= deadline:
                _signal(pid, signal.SIGKILL)
            elif pid not in signaled:
                _signal(pid, signal.SIGTERM)
                signaled.add(pid)
        # An uninterruptible process must keep its GPU reservation. Do not
        # acknowledge cleanup until the complete tree has exited.
        time.sleep(0.02)


def _wait_for_tree_exit(runner, grace: float) -> None:
    """Allow normal teardown, including adopted multiprocessing helpers."""
    deadline = time.monotonic() + grace
    while True:
        runner.join(timeout=0)
        if runner.exitcode is not None:
            _reap()
            if not _descendants(os.getpid()):
                return
        if time.monotonic() >= deadline:
            return
        time.sleep(0.02)


def _supervise_worker(control, conn, device, factory, max_requests, capture_dir, grace):
    """Own one worker and reap its descendants before acknowledging termination."""
    if hasattr(os, "setsid"):
        os.setsid()
    linux = sys.platform == "linux"
    if linux and ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        control.send({"error": "cannot supervise worker descendants"})
        return
    runner = mp.get_context("spawn").Process(
        target=worker_main, args=(device, conn, factory, max_requests, capture_dir)
    )
    runner.start()
    conn.close()
    terminating = False
    try:
        control.send({"pid": runner.pid})
        while runner.is_alive():
            if control.poll(0.02):
                terminating = True
                break  # terminate or parent disconnected
            runner.join(timeout=0)
        if linux:
            if not terminating:
                _wait_for_tree_exit(runner, grace)
            orphaned = _drain_children(runner, grace)
        else:
            _terminate_process_tree(runner, grace)
            orphaned = False
        control.send({"exitcode": runner.exitcode, "orphaned": orphaned})
    finally:
        if linux:
            _drain_children(runner, grace)
        elif runner.is_alive():
            _terminate_process_tree(runner, grace)
        control.close()


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
