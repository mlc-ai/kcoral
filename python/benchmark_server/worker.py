"""The GPU worker: a child process that owns one GPU's CUDA context.

The front-end never touches the GPU - it dispatches a Program to a worker over a
pipe and reads back one execution outcome. A worker runs one program at a time
and clears its per-request state after each (via the Runtime). The worker's
`main` is the child entry point; :class:`Worker` is the parent-side handle with
crash/timeout kill + respawn.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from collections.abc import Callable

from .engine import execute
from .lease import GPULeases, LeaseClient

_WORKER_PIPE_FAILURES = (EOFError, ConnectionResetError, BrokenPipeError, OSError)


def worker_main(gpu_id: int, conn, runtime_factory: Callable, max_requests: int) -> None:
    """Child entry point. Pins the GPU, builds the Runtime, serves programs."""
    # Lead a new process group, so the parent can clean up anything the submitted
    # code spawned (grandchildren included) with one killpg.
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    # Select the GPU before the Runtime imports torch/tvm, so it sees one device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
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
        described = {"target": runtime.target(), "versions": runtime.versions()}
    except Exception as exc:  # runtime init failed — report and exit
        conn.send({"__error__": f"runtime init failed: {type(exc).__name__}: {exc}"})
        return
    # Only this process sees the GPU, so only it can describe the target.
    conn.send({"__ready__": described})
    lease = LeaseClient(conn)
    requests_served = 0
    while True:
        try:
            program = conn.recv()
        except EOFError:
            return
        if program is None:  # shutdown signal
            return
        cleanup_error: BaseException | None = None

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
        )
        requests_served += 1
        if cleanup_error is not None:
            # Send the request's result before this process is discarded. The
            # parent respawns synchronously before returning the worker to idle.
            conn.send({"__outcome__": outcome, "__replace_worker__": "poisoned_context"})
            return
        if max_requests and requests_served >= max_requests:
            # A fresh process gives every request the same CUDA context and
            # allocator starting state, including after submitted native code
            # happened to return without a detectable sticky error.
            conn.send({"__outcome__": outcome, "__replace_worker__": "request_limit"})
            return
        conn.send(outcome)


class WorkerCrashed(Exception):
    # Attribution filled in by the pool when the failure happened under submit.
    gpu_id: int | None = None
    queue_ms: float | None = None
    elapsed_ms: float | None = None
    instruction_index: int | None = None
    exitcode: int | None = None
    lease_wait_ms: float = 0.0
    lease_held_ms: float = 0.0


class WorkerTimeout(Exception):
    gpu_id: int | None = None
    queue_ms: float | None = None
    elapsed_ms: float | None = None


class Worker:
    """Parent-side handle to one GPU worker process."""

    def __init__(
        self,
        gpu_id: int,
        runtime_factory: Callable,
        spawn_timeout: float = 60.0,
        termination_grace_seconds: float = 5.0,
        max_requests: int = 1,
    ) -> None:
        self.gpu_id = gpu_id
        self.request_id: str | None = None
        self._factory = runtime_factory
        self._spawn_timeout = spawn_timeout
        self._termination_grace_seconds = termination_grace_seconds
        self._max_requests = max_requests
        self._ctx = mp.get_context("spawn")  # 'spawn' — 'fork' is unsafe with CUDA
        self._spawn()

    def _spawn(self) -> None:
        """Start and fully initialize a worker when no request can use its GPU."""
        self._start_process()
        self._initialize_process()

    def _start_process(self) -> None:
        """Spawn a worker and finish its explicitly host-only preparation."""
        self.started_at = time.monotonic()
        parent, child = self._ctx.Pipe()
        self._conn = parent
        self._proc = self._ctx.Process(
            target=worker_main,
            args=(self.gpu_id, child, self._factory, self._max_requests),
            daemon=True,
        )
        self._proc.start()
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
        outcome. Returns the outcome, lease timings, and optional restart reason.

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
            self._conn.send(program)
            while True:
                waited_from = time.monotonic()
                if not self._conn.poll(remaining):  # no answer by the deadline -> hung
                    self._abandon_and_respawn(leases)
                    raise WorkerTimeout(f"program exceeded {timeout}s on GPU {self.gpu_id}")
                remaining -= time.monotonic() - waited_from
                message = self._conn.recv()
                if isinstance(message, dict) and "__instruction__" in message:
                    instruction_index = message["__instruction__"]
                    continue
                if isinstance(message, dict) and "__replace_worker__" in message:
                    if held_since is not None:
                        lease_held_ms += (time.monotonic() - held_since) * 1000
                    outcome = message.get("__outcome__")
                    reason = message["__replace_worker__"]
                    self._abandon_and_respawn(leases)
                    return outcome, lease_wait_ms, lease_held_ms, reason
                if not (isinstance(message, dict) and "__lease__" in message):
                    # Anything that is not a lease message is the program's outcome.
                    if held_since is not None:
                        lease_held_ms += (time.monotonic() - held_since) * 1000
                    leases.release(self.gpu_id, self)  # no-op if the engine already did
                    return message, lease_wait_ms, lease_held_ms, None
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
            crash.instruction_index = instruction_index
            crash.exitcode = self._proc.exitcode
            crash.lease_wait_ms = lease_wait_ms
            crash.lease_held_ms = lease_held_ms
            self._abandon_and_respawn(leases)
            raise crash from exc

    def _abandon_and_respawn(self, leases: GPULeases) -> None:
        """Replace a failed worker without letting its lifecycle touch an in-use GPU.

        Terminate the old process before abandoning its lease: a timed-out native
        kernel may still be running until process termination completes.  Worker
        startup also creates a CUDA context and warms the runtime. Start the new
        process and finish its explicitly CPU-only preparation without the lease,
        then reacquire before granting GPU initialization. Otherwise a replacement
        racing a waiter can initialize CUDA alongside that waiter's program.
        """
        self._kill()
        leases.abandon(self.gpu_id, self)
        self._start_process()
        leases.acquire(self.gpu_id, self)
        try:
            self._initialize_process()
        finally:
            leases.release(self.gpu_id, self)

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
