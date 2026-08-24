"""The GPU worker: a child process that owns one GPU's CUDA context.

The front-end never touches the GPU - it dispatches a Program to a worker over a
pipe and reads back one execution outcome. A worker runs one program at a time,
clears its per-request state after each (via the Runtime), and is replaced by a
fresh process once it has served ``max_requests`` of them. The worker's
`main` is the child entry point; :class:`Worker` is the parent-side handle with
crash/timeout kill + respawn.

Every request ends with a *finish reason*: ``completed`` and ``program_failed``
leave the worker serving, ``request_limit`` and ``poisoned_context`` retire it,
and the parent supplies ``timeout`` or ``crashed`` when no answer arrived.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from collections.abc import Callable

from . import nvml
from .engine import execute, read_captured_output
from .events import EventLogger
from .lease import GPULeases, LeaseClient

_WORKER_PIPE_FAILURES = (EOFError, ConnectionResetError, BrokenPipeError, OSError)

# Finish reasons whose worker answered the request but must not serve another.
RETIRING_FINISH_REASONS = frozenset({"poisoned_context", "request_limit"})

# How loud each one is: a failed program is the client's kernel and stays INFO,
# so one level separates our faults from theirs.
FINISH_REASON_LEVEL = {
    "completed": "INFO",
    "program_failed": "INFO",
    "request_limit": "INFO",
    "poisoned_context": "WARNING",
    "timeout": "WARNING",
    "crashed": "ERROR",
}

_NO_EVENTS = EventLogger(None)  # a shared no-op, for a handle given no log
_OUTPUT_TAIL_BYTES = 4096  # of a killed worker's output, kept on its log record


def worker_main(
    device: str, conn, runtime_factory: Callable, max_requests: int, capture_dir: str | None = None
) -> None:
    """Child entry point. Pins the GPU, builds the Runtime, serves programs.

    ``device`` is the ``CUDA_VISIBLE_DEVICES`` value the parent resolved, a UUID
    where NVML could name one.
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
            # Read before an upload can edit it, so none can talk its worker off the lease.
            "cpu_only": sorted(runtime.cpu_only_builtins()),
            "device_uuid": runtime.device_uuid(),
        }
    except Exception as exc:  # runtime init failed — report and exit
        conn.send({"__error__": f"runtime init failed: {type(exc).__name__}: {exc}"})
        return
    # Only this process sees the GPU, so only it can describe the target.
    conn.send({"__ready__": described})
    lease = LeaseClient(conn)
    requests_served = 0
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:  # shutdown signal
            return
        program, cpu_only = message
        cleanup_error: BaseException | None = None

        def mark_cleanup_failed(exc: BaseException) -> None:
            nonlocal cleanup_error
            if cleanup_error is None:
                cleanup_error = exc

        outcome = execute(
            program,
            runtime,
            lease=lease,
            cpu_only=cpu_only,
            progress=lambda index: conn.send({"__instruction__": index}),
            cleanup_failed=mark_cleanup_failed,
            capture_dir=capture_dir,
        )
        requests_served += 1
        # The only reasons this side can name are the two that retire it; None
        # means it can serve on, and the parent reads the rest off the outcome.
        retire_reason: str | None = None
        if cleanup_error is not None:
            retire_reason = "poisoned_context"
        elif max_requests and requests_served >= max_requests:
            # A fresh process gives every request the same context and allocator
            # state, even where native code left no detectable sticky error.
            retire_reason = "request_limit"
        # Always the same shape, so the parent reads the reason rather than
        # inferring it, and the result goes first either way: the parent respawns
        # after answering, not before.
        conn.send({"__outcome__": outcome, "__finish__": retire_reason})
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
    """Parent-side handle to one GPU worker process."""

    # Replaced from the worker's ready message; empty means nothing runs off-lease.
    cpu_only: frozenset[str] = frozenset()

    # Identity for the log. ``worker_id`` names a seat on a GPU and outlives the
    # processes that sit in it; ``generation`` and ``pid`` say which one does now.
    worker_id: str = ""
    generation: int = 0
    pid: int | None = None
    _events: EventLogger = _NO_EVENTS
    _capture_dir: str | None = None

    def __init__(
        self,
        gpu_id: int,
        runtime_factory: Callable,
        spawn_timeout: float = 60.0,
        termination_grace_seconds: float = 5.0,
        max_requests: int = 1,
        index: int = 0,
        events: EventLogger | None = None,
        capture_dir: str | None = None,
    ) -> None:
        self.gpu_id = gpu_id
        self.worker_id = f"gpu{gpu_id}/w{index}"
        if events is not None:
            self._events = events
        self._expected_uuid = nvml.device_uuid(gpu_id)
        self._device = self._expected_uuid or str(gpu_id)
        self._factory = runtime_factory
        self._spawn_timeout = spawn_timeout
        self._termination_grace_seconds = termination_grace_seconds
        self._max_requests = max_requests
        self._capture_dir = capture_dir
        self._ctx = mp.get_context("spawn")  # 'spawn' — 'fork' is unsafe with CUDA
        self._spawn()

    def _spawn(self) -> None:
        """Start and fully initialize a worker when no request can use its GPU."""
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
        self.started_at = time.monotonic()
        parent, child = self._ctx.Pipe()
        self._conn = parent
        self._proc = self._ctx.Process(
            target=worker_main,
            args=(self._device, child, self._factory, self._max_requests, self._capture_dir),
            daemon=True,
        )
        self._proc.start()
        self.generation += 1
        self.pid = self._proc.pid
        child.close()  # parent keeps only its end, so it sees EOF if the child dies
        msg = self._await_startup_message("prepare")
        if not (isinstance(msg, dict) and msg.get("__startup__") == "prepared"):
            self._kill()
            raise WorkerCrashed(f"worker preparation error: {msg}")

    def _initialize_process(self) -> None:
        """Create the worker's CUDA Runtime; the caller must serialize its GPU."""
        try:
            self._conn.send({"__startup__": "initialize"})
        except _WORKER_PIPE_FAILURES as exc:
            self._kill()
            raise WorkerCrashed(
                f"worker on GPU {self.gpu_id} failed before GPU initialization"
            ) from exc
        msg = self._await_startup_message("initialize")
        if not (isinstance(msg, dict) and "__ready__" in msg):
            self._kill()
            raise WorkerCrashed(f"worker init error: {msg}")
        described = msg["__ready__"]
        self.target: dict[str, str] = described["target"]
        self.versions: dict[str, str] = described["versions"]
        self.cpu_only = frozenset(described["cpu_only"])
        self.device_uuid: str | None = described.get("device_uuid")
        self._require_expected_device()
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
            ready = self._conn.poll(self._spawn_timeout)
            msg = self._conn.recv() if ready else None
        except _WORKER_PIPE_FAILURES as exc:
            self._kill()
            raise WorkerCrashed(f"worker on GPU {self.gpu_id} failed during {phase}") from exc
        if not ready:
            self._kill()
            raise WorkerCrashed(f"worker on GPU {self.gpu_id} timed out during {phase}")
        return msg

    def run(self, program, timeout: float, leases: GPULeases) -> tuple:
        """Run a program, servicing its lease requests; kill+respawn on timeout or
        crash, then re-raise. A poisoned context is respawned after preserving its
        outcome. Returns the outcome, the lease timings, and the finish_reason.

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
            self._conn.send((program, self.cpu_only))
            while True:
                waited_from = time.monotonic()
                if not self._conn.poll(remaining):  # no answer by the deadline -> hung
                    timed_out = WorkerTimeout(f"program exceeded {timeout}s on GPU {self.gpu_id}")
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
                    if held_since is not None:
                        lease_held_ms += (time.monotonic() - held_since) * 1000
                    leases.release(self.gpu_id, self)  # no-op if the engine already did
                    outcome = message["__outcome__"]
                    # A retirement the child named, or else the program's result.
                    finish_reason = message["__finish__"] or _program_finish_reason(outcome)
                    return outcome, lease_wait_ms, lease_held_ms, finish_reason
                if not (isinstance(message, dict) and "__lease__" in message):
                    # Nothing this protocol defines. Hand it up rather than read
                    # it as lease traffic: the front-end rejects what is not a
                    # ProgramOutcome, which beats failing on the message's shape.
                    if held_since is not None:
                        lease_held_ms += (time.monotonic() - held_since) * 1000
                    leases.release(self.gpu_id, self)
                    return message, lease_wait_ms, lease_held_ms, "program_failed"
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
            crash = WorkerCrashed(f"worker on GPU {self.gpu_id} pipe failed: {exc}")
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
        return read_captured_output(self._capture_dir, self.pid, _OUTPUT_TAIL_BYTES)

    def replace(self, leases: GPULeases, reason: str) -> None:
        """Swap in a fresh process. The pool calls this after answering the request
        the old one served, so no client waits for a respawn."""
        self._abandon_and_respawn(leases, reason)

    def _abandon_and_respawn(self, leases: GPULeases, reason: str) -> None:
        """Replace a worker without letting its lifecycle touch an in-use GPU.

        Kill before abandoning the lease: a timed-out kernel may still be running
        until termination completes. Then prepare off-lease but initialize under it,
        or a replacement racing a waiter creates its context alongside that
        waiter's program.
        """
        self._events.emit(
            "worker_retired",
            level=FINISH_REASON_LEVEL.get(reason, "INFO"),
            worker_id=self.worker_id,
            gpu_id=self.gpu_id,
            generation=self.generation,
            pid=self.pid,
            reason=reason,
        )
        self._kill()
        leases.abandon(self.gpu_id, self)
        try:
            self._start_process()
            leases.acquire(self.gpu_id, self)
            try:
                self._initialize_process()
            finally:
                leases.release(self.gpu_id, self)
        except Exception as exc:
            self._log_failure("respawn", exc)
            raise

    def _kill(self) -> None:
        try:
            _terminate_process_tree(self._proc, self._termination_grace_seconds)
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._conn.send(None)
        except Exception:
            pass
        self._kill()


def _program_finish_reason(outcome) -> str:
    """Why a worker that stayed alive answered: what the program itself did."""
    return "completed" if getattr(outcome, "status", None) == "COMPLETED" else "program_failed"


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
