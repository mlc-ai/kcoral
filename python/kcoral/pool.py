"""CPU or GPU worker pool with backpressure.

`submit` blocks (call it from a thread), acquires an idle worker, runs the
program, and returns a :class:`SubmitOutcome` - the execution outcome, the
worker's optional GPU id, and the queue/execution/lease timings. The worker
returns to the idle set respawned already if it crashed or timed out; the raised
:class:`WorkerTimeout` / :class:`WorkerCrashed` carries the same attribution
(``gpu_id``, ``queue_ms``, ``elapsed_ms``). When no worker becomes free within
``worker_wait_timeout`` it raises :class:`PoolBusy` (the front-end maps that to
HTTP 503).

More workers than GPUs is the point: while one compiles, another can measure on
the GPU it is not using. They take turns through a per-GPU lease, so no two ever
run on one GPU at once - see :mod:`kcoral.lease`.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from .events import EventLogger
from .lease import GPULeases, NoopLeases, Ticket
from .worker import RETIRING_FINISH_REASONS, Worker, WorkerCrashed, WorkerTimeout


class IdleWorkers:
    """The idle set, handing out a worker from the least-loaded GPU.

    A request takes a free worker from whichever GPU has fewest out; when it
    finishes the worker goes back, handed straight to the longest waiter if
    there is one. One flat FIFO queue would be GPU-blind, landing two requests
    on one GPU's workers while another GPU sits idle.
    """

    def __init__(self, workers: list[Worker]) -> None:
        self._lock = threading.Lock()
        self._idle: dict[int | None, list[Worker]] = {}  # device -> its free workers
        self._assigned: dict[int | None, int] = {}  # device -> how many are out
        for worker in workers:
            self._idle.setdefault(worker.gpu_id, []).append(worker)
            self._assigned.setdefault(worker.gpu_id, 0)
        self._waiters: deque[Ticket] = deque()  # requests with no worker yet

    def acquire(self, timeout: float) -> Worker | None:
        """A worker from the least-loaded GPU, or None if none frees up in time."""
        with self._lock:
            worker = self._claim_least_loaded()
            if worker is not None:
                return worker
            if timeout <= 0:
                return None
            ticket = Ticket()
            self._waiters.append(ticket)
        if not ticket.event.wait(timeout):
            with self._lock:
                if ticket.value is None:  # nothing arrived while we timed out
                    self._waiters.remove(ticket)
                    return None
        return ticket.value

    def release(self, worker: Worker) -> None:
        with self._lock:
            self._assigned[worker.gpu_id] -= 1
            self._idle[worker.gpu_id].append(worker)
            if self._waiters:
                ticket = self._waiters.popleft()
                ticket.value = self._claim_least_loaded()  # never None: one just returned
                ticket.event.set()

    def _claim_least_loaded(self) -> Worker | None:
        """A free worker from the GPU with fewest out, marked out. Needs ``_lock``."""
        with_a_free_worker = [gpu for gpu, workers in self._idle.items() if workers]
        if not with_a_free_worker:
            return None
        gpu_id = min(with_a_free_worker, key=lambda gpu: self._assigned[gpu])
        self._assigned[gpu_id] += 1
        return self._idle[gpu_id].pop()

    def snapshot(self) -> tuple[set[int], int]:
        """``id()`` of every idle worker, and the number of requests waiting."""
        with self._lock:
            idle = {id(w) for workers in self._idle.values() for w in workers}
            return idle, len(self._waiters)


class MixedTargets(Exception):
    """The configured GPUs do not share one compilation target."""


class PoolBusy(Exception):
    def __init__(self, message: str, queue_ms: float = 0.0):
        super().__init__(message)
        self.queue_ms = queue_ms


@dataclass
class SubmitOutcome:
    execution: object
    gpu_id: int | None
    queue_ms: float
    elapsed_ms: float
    lease_wait_ms: float = 0.0
    lease_held_ms: float = 0.0
    worker_id: str = ""
    finish_reason: str = "completed"
    interfered_request_id: str | None = None  # who held the GPU when a cpu_only call touched it


class WorkerPool:
    def __init__(
        self,
        gpus: list[int],
        runtime_factory: Callable,
        termination_grace_seconds: float = 5.0,
        workers_per_gpu: int = 1,
        max_requests_per_worker: int = 1,
        cpu_workers: int | None = None,
        events: EventLogger | None = None,
    ) -> None:
        if max_requests_per_worker < 0:
            raise ValueError("max_requests_per_worker must be non-negative")
        if workers_per_gpu < 1:
            raise ValueError(f"workers_per_gpu must be at least 1, got {workers_per_gpu}")
        if cpu_workers is not None and cpu_workers < 1:
            raise ValueError(f"cpu_workers must be at least 1, got {cpu_workers}")
        if gpus and cpu_workers is not None:
            raise ValueError("a worker pool cannot mix CPU and GPU workers")
        if not gpus and cpu_workers is None:
            raise ValueError("a GPU worker pool needs at least one GPU")
        self._events = events or EventLogger(None)
        capture_dir = self._events.subdir("output")
        # (device, index) pairs: the index names a worker within its device, so a
        # CPU pool's workers stay distinguishable in the log without a GPU id.
        devices: list[tuple[int | None, int]]
        if cpu_workers is not None:
            devices = [(None, index) for index in range(cpu_workers)]
        else:
            devices = [(gpu, index) for gpu in gpus for index in range(workers_per_gpu)]
        self._workers = [
            Worker(
                device,
                runtime_factory,
                termination_grace_seconds=termination_grace_seconds,
                max_requests=max_requests_per_worker,
                index=index,
                events=self._events,
                capture_dir=str(capture_dir) if capture_dir is not None else None,
            )
            for device, index in devices
        ]
        self._require_one_target()
        self._gpus = list(dict.fromkeys(gpus))
        self._idle = IdleWorkers(self._workers)
        self._leases = NoopLeases() if cpu_workers is not None else GPULeases(self._gpus)
        self._replacing: set[threading.Thread] = set()
        self._replacing_lock = threading.Lock()  # so shutdown can copy it mid-flight
        self._closing = threading.Event()

    def submit(
        self,
        program,
        timeout: float,
        worker_wait_timeout: float = 0.0,
        request_id: str | None = None,
    ) -> SubmitOutcome:
        queue_started = time.monotonic()
        worker = self._idle.acquire(worker_wait_timeout)
        queue_ms = (time.monotonic() - queue_started) * 1000
        if worker is None:
            raise PoolBusy("all workers busy", queue_ms=queue_ms)
        self._events.emit(
            "request_routed",
            request_id=request_id,
            worker_id=worker.worker_id,
            gpu_id=worker.gpu_id,
            generation=worker.generation,
            pid=worker.pid,
            queue_ms=queue_ms,
        )
        run_started = time.monotonic()
        finish_reason = None
        worker.request_id = request_id
        try:
            execution, lease_wait_ms, lease_held_ms, finish_reason = worker.run(
                program, timeout, self._leases
            )
            return SubmitOutcome(
                execution,
                worker.gpu_id,
                queue_ms,
                (time.monotonic() - run_started) * 1000,
                lease_wait_ms,
                lease_held_ms,
                worker.worker_id,
                finish_reason,
                self._interfered_request(worker, execution, request_id),
            )
        except (WorkerTimeout, WorkerCrashed) as exc:
            exc.worker_id = worker.worker_id
            exc.gpu_id = worker.gpu_id
            exc.queue_ms = queue_ms
            exc.elapsed_ms = (time.monotonic() - run_started) * 1000
            raise
        finally:
            self._leases.abandon(worker.gpu_id, worker)  # no-op unless it still holds
            worker.request_id = None
            self._release_or_replace(worker, finish_reason)

    def _interfered_request(self, worker: Worker, execution, request_id: str | None) -> str | None:
        """The request holding the GPU when this one's ``cpu_only`` call touched it."""
        error = getattr(execution, "error", None)
        if not isinstance(error, dict) or error.get("kind") != "gpu_access":
            return None
        holder = self._leases.request_at(worker.gpu_id, error.pop("detected_at_ns"))
        return None if holder == request_id else holder  # its own release may reach us late

    def _release_or_replace(self, worker: Worker, finish_reason: str | None) -> None:
        """Hand the worker back, replacing its process first if it retired.

        A retired worker has already exited, so it stays out of the idle set until
        its replacement is ready. Building it on its own thread keeps it out of the
        answered request's timings; a crash or timeout respawned in place already.
        """
        if finish_reason not in RETIRING_FINISH_REASONS or self._closing.is_set():
            self._idle.release(worker)
            return
        thread = threading.Thread(target=self._replace, args=(worker, finish_reason), daemon=True)
        with self._replacing_lock:
            self._replacing.add(thread)
        try:
            thread.start()
        except RuntimeError:  # no thread to be had; rebuild it here instead
            with self._replacing_lock:
                self._replacing.discard(thread)
            self._replace(worker, finish_reason)

    def _replace(self, worker: Worker, reason: str) -> None:
        try:
            worker.replace(self._leases, reason)
        except Exception as exc:
            # A respawn that failed leaves a dead worker; the next run revives
            # it, so unlogged this shows up only as the pool being slow.
            self._events.emit(
                "worker_replace_failed",
                level="ERROR",
                worker_id=worker.worker_id,
                gpu_id=worker.gpu_id,
                reason=reason,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            self._idle.release(worker)
            with self._replacing_lock:
                self._replacing.discard(threading.current_thread())

    def _require_one_target(self) -> None:
        """One pool serves one target, so a client builds one library that any
        worker runs. Mixed GPUs mean two servers."""
        by_target = {}
        for worker in self._workers:
            by_target.setdefault(tuple(sorted(worker.target.items())), []).append(worker.gpu_id)
        if len(by_target) > 1:
            self.shutdown()
            grouped = "; ".join(
                f"GPU(s) {sorted(set(gpus))} are {dict(target)['arch']}"
                for target, gpus in by_target.items()
            )
            raise MixedTargets(f"a pool serves one target, but {grouped}")

    def health(self) -> dict:
        idle_ids, waiting = self._idle.snapshot()
        now = time.monotonic()
        return {
            "queue_length": waiting,
            "target": self.target(),
            "versions": self.versions(),
            "gpus": [{"gpu_id": gpu, "lease_depth": self._leases.depth(gpu)} for gpu in self._gpus],
            "workers": [
                {
                    "worker_id": w.worker_id,
                    "gpu_id": w.gpu_id,
                    "status": "idle" if id(w) in idle_ids else "busy",
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
        self._closing.set()
        with self._replacing_lock:
            replacing = list(self._replacing)
        for thread in replacing:  # let a half-built replacement finish, not leak
            thread.join(timeout=60)
        for w in self._workers:
            w.close()
