"""Worker pool: one worker per GPU, idle-worker assignment, backpressure.

`submit` blocks (call it from a thread), acquires an idle worker, runs the
program, and returns a :class:`SubmitOutcome` - the execution outcome, the GPU
that ran the program, and the queue/execution timings. The worker returns to the
idle set respawned already if it crashed or timed out; the raised
:class:`WorkerTimeout` / :class:`WorkerCrashed` carries the same attribution
(``gpu_id``, ``queue_ms``, ``elapsed_ms``). When no worker becomes free within
``worker_wait_timeout`` it raises :class:`PoolBusy` (the front-end maps that to
HTTP 503).
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .worker import Worker, WorkerCrashed, WorkerTimeout


class MixedTargets(Exception):
    """The configured GPUs do not share one compilation target."""


class PoolBusy(Exception):
    def __init__(self, message: str, queue_ms: float = 0.0):
        super().__init__(message)
        self.queue_ms = queue_ms


@dataclass
class SubmitOutcome:
    execution: object
    gpu_id: int
    queue_ms: float
    elapsed_ms: float


class WorkerPool:
    def __init__(
        self,
        gpus: list[int],
        runtime_factory: Callable,
        termination_grace_seconds: float = 5.0,
    ) -> None:
        self._workers = [
            Worker(g, runtime_factory, termination_grace_seconds=termination_grace_seconds)
            for g in gpus
        ]
        self._require_one_target()
        self._idle: queue.Queue[Worker] = queue.Queue()
        for w in self._workers:
            self._idle.put(w)
        self._busy_gpu_ids: set[int] = set()
        self._waiting = 0
        self._state_lock = threading.Lock()

    def submit(self, program, timeout: float, worker_wait_timeout: float = 0.0) -> SubmitOutcome:
        queue_started = time.monotonic()
        with self._state_lock:
            self._waiting += 1
        try:
            if worker_wait_timeout > 0:
                worker = self._idle.get(timeout=worker_wait_timeout)
            else:
                worker = self._idle.get_nowait()
        except queue.Empty:
            queue_ms = (time.monotonic() - queue_started) * 1000
            raise PoolBusy("all workers busy", queue_ms=queue_ms) from None
        finally:
            with self._state_lock:
                self._waiting -= 1
        queue_ms = (time.monotonic() - queue_started) * 1000
        with self._state_lock:
            self._busy_gpu_ids.add(worker.gpu_id)
        run_started = time.monotonic()
        try:
            execution = worker.run(program, timeout)
            elapsed_ms = (time.monotonic() - run_started) * 1000
            return SubmitOutcome(execution, worker.gpu_id, queue_ms, elapsed_ms)
        except (WorkerTimeout, WorkerCrashed) as exc:
            exc.gpu_id = worker.gpu_id
            exc.queue_ms = queue_ms
            exc.elapsed_ms = (time.monotonic() - run_started) * 1000
            raise
        finally:
            with self._state_lock:
                self._busy_gpu_ids.discard(worker.gpu_id)
            self._idle.put(worker)  # worker was respawned in-place on crash/timeout

    def _require_one_target(self) -> None:
        """One pool serves one target, so a client builds one library that any
        worker runs. Mixed GPUs mean two servers."""
        by_target = {}
        for worker in self._workers:
            by_target.setdefault(tuple(sorted(worker.target.items())), []).append(worker.gpu_id)
        if len(by_target) > 1:
            self.shutdown()
            grouped = "; ".join(
                f"GPU(s) {gpus} are {dict(target)['arch']}" for target, gpus in by_target.items()
            )
            raise MixedTargets(f"a pool serves one target, but {grouped}")

    def health(self) -> dict:
        with self._state_lock:
            busy_gpu_ids = set(self._busy_gpu_ids)
            waiting = self._waiting
        now = time.monotonic()
        return {
            "queue_length": waiting,
            "target": self.target(),
            "versions": self.versions(),
            "workers": [
                {
                    "gpu_id": w.gpu_id,
                    "status": "busy" if w.gpu_id in busy_gpu_ids else "idle",
                    "uptime_seconds": max(0.0, now - w.started_at),
                }
                for w in self._workers
            ],
        }

    def target(self) -> dict[str, str]:
        """What a client must compile an uploaded library for; startup rejects a
        pool whose workers would disagree."""
        return self._workers[0].target

    def versions(self) -> dict[str, str]:
        return self._workers[0].versions

    def shutdown(self) -> None:
        for w in self._workers:
            w.close()
