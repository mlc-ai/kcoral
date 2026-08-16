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


def worker_main(gpu_id: int, conn, runtime_factory: Callable) -> None:
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
        runtime = runtime_factory()
        described = {"target": runtime.target(), "versions": runtime.versions()}
    except Exception as exc:  # runtime init failed — report and exit
        conn.send({"__error__": f"runtime init failed: {type(exc).__name__}: {exc}"})
        return
    # Only this process sees the GPU, so only it can describe the target.
    conn.send({"__ready__": described})
    lease = LeaseClient(conn)
    while True:
        try:
            program = conn.recv()
        except EOFError:
            return
        if program is None:  # shutdown signal
            return
        conn.send(
            execute(
                program,
                runtime,
                lease=lease,
                progress=lambda index: conn.send({"__instruction__": index}),
            )
        )


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
    ) -> None:
        self.gpu_id = gpu_id
        self._factory = runtime_factory
        self._spawn_timeout = spawn_timeout
        self._termination_grace_seconds = termination_grace_seconds
        self._ctx = mp.get_context("spawn")  # 'spawn' — 'fork' is unsafe with CUDA
        self._spawn()

    def _spawn(self) -> None:
        self.started_at = time.monotonic()
        parent, child = self._ctx.Pipe()
        self._conn = parent
        self._proc = self._ctx.Process(
            target=worker_main, args=(self.gpu_id, child, self._factory), daemon=True
        )
        self._proc.start()
        child.close()  # parent keeps only its end, so it sees EOF if the child dies
        if not self._conn.poll(self._spawn_timeout):
            self._kill()
            raise WorkerCrashed(f"worker on GPU {self.gpu_id} did not become ready")
        msg = self._conn.recv()
        if not (isinstance(msg, dict) and "__ready__" in msg):
            self._kill()
            raise WorkerCrashed(f"worker init error: {msg}")
        described = msg["__ready__"]
        self.target: dict[str, str] = described["target"]
        self.versions: dict[str, str] = described["versions"]

    def run(self, program, timeout: float, leases: GPULeases) -> tuple:
        """Run a program, servicing its lease requests; kill+respawn on timeout or
        crash, then re-raise. Returns ``(outcome, lease_wait_ms, lease_held_ms)``.

        The deadline covers only the worker's own work - time blocked on a lease
        another worker holds is not counted, or ``timeout_seconds`` would mean
        different things at different loads.
        """
        self._conn.send(program)
        remaining = timeout
        lease_wait_ms = 0.0
        lease_held_ms = 0.0
        held_since: float | None = None
        instruction_index: int | None = None
        while True:
            waited_from = time.monotonic()
            if not self._conn.poll(remaining):  # no answer by the deadline -> hung
                self._abandon_and_respawn(leases)
                raise WorkerTimeout(f"program exceeded {timeout}s on GPU {self.gpu_id}")
            remaining -= time.monotonic() - waited_from
            try:
                message = self._conn.recv()
            except EOFError:  # child died mid-run
                if held_since is not None:
                    lease_held_ms += (time.monotonic() - held_since) * 1000
                # Refresh multiprocessing's view of a child that has just closed
                # its pipe before replacing it.
                self._proc.join(timeout=0.1)
                crash = WorkerCrashed(f"worker on GPU {self.gpu_id} crashed")
                crash.instruction_index = instruction_index
                crash.exitcode = self._proc.exitcode
                crash.lease_wait_ms = lease_wait_ms
                crash.lease_held_ms = lease_held_ms
                self._abandon_and_respawn(leases)
                raise crash
            if isinstance(message, dict) and "__instruction__" in message:
                instruction_index = message["__instruction__"]
                continue
            if not (isinstance(message, dict) and "__lease__" in message):
                # Anything that is not a lease message is the program's outcome.
                if held_since is not None:
                    lease_held_ms += (time.monotonic() - held_since) * 1000
                leases.release(self.gpu_id, self)  # no-op if the engine already did
                return message, lease_wait_ms, lease_held_ms
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

    def _abandon_and_respawn(self, leases: GPULeases) -> None:
        """Free the GPU before respawning: the dead worker cannot do it itself,
        and a lease left held would strand every other worker on that GPU."""
        leases.abandon(self.gpu_id, self)
        self._kill_and_respawn()

    def _kill(self) -> None:
        try:
            _terminate_process_tree(self._proc, self._termination_grace_seconds)
        except Exception:
            pass

    def _kill_and_respawn(self) -> None:
        self._kill()
        self._spawn()

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
