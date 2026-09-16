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

import asyncio
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import nvml
from .events import EventLogger
from .job import JobCleanupError, run_job
from .lease import GPULeases, GPUUnavailable, NoopLeases, Ticket
from .worker import Worker, WorkerCrashed, WorkerTimeout


class IdleWorkers:
    """The idle set, handing out a worker from the least-loaded GPU.

    A request takes a free worker from whichever GPU has fewest out; when it
    finishes the worker goes back, handed straight to the longest waiter if
    there is one. One flat FIFO queue would be GPU-blind, landing two requests
    on one GPU's workers while another GPU sits idle.
    """

    def __init__(self, workers: list[Worker]) -> None:
        self._lock = threading.Lock()
        self._closed = False
        self._idle: dict[int | None, list[Worker]] = {}  # device -> its free workers
        self._assigned: dict[int | None, int] = {}  # device -> how many are out
        for worker in workers:
            self._idle.setdefault(worker.gpu_id, []).append(worker)
            self._assigned.setdefault(worker.gpu_id, 0)
        self._waiters: deque[Ticket] = deque()  # requests with no worker yet

    def acquire(self, timeout: float) -> Worker | None:
        """A worker from the least-loaded GPU, or None if none frees up in time."""
        with self._lock:
            if self._closed:
                return None
            worker = self._claim_least_loaded()
            if worker is not None:
                return worker
            if timeout <= 0:
                return None
            ticket = Ticket()
            self._waiters.append(ticket)
        if not ticket.event.wait(timeout):
            with self._lock:
                if ticket.value is None and not self._closed:  # nothing arrived while we timed out
                    self._waiters.remove(ticket)
                    return None
        return ticket.value

    def release(self, worker: Worker) -> None:
        with self._lock:
            self._assigned[worker.gpu_id] -= 1
            if self._closed:
                return
            self._idle[worker.gpu_id].append(worker)
            if self._waiters:
                ticket = self._waiters.popleft()
                ticket.value = self._claim_least_loaded()  # never None: one just returned
                ticket.event.set()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            while self._waiters:
                self._waiters.popleft().event.set()

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
    gpu_ids: tuple[int, ...] = ()


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
        sandbox: str = "bubblewrap",
        sandbox_readonly_paths: tuple[Path, ...] = (),
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
        self._factory = runtime_factory
        self._termination_grace_seconds = termination_grace_seconds
        self._events = events or EventLogger(None)
        capture_dir = self._events.subdir("output")
        # (device, index) pairs: the index names a worker within its device, so a
        # CPU pool's workers stay distinguishable in the log without a GPU id.
        devices: list[tuple[int | None, int]]
        if cpu_workers is not None:
            devices = [(None, index) for index in range(cpu_workers)]
        else:
            devices = [(gpu, index) for gpu in gpus for index in range(workers_per_gpu)]
        self._closing = threading.Event()
        self._condition = threading.Condition()
        self._active = 0
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False
        self._workers = []
        try:
            for device, index in devices:
                self._workers.append(
                    Worker(
                        device,
                        runtime_factory,
                        termination_grace_seconds=termination_grace_seconds,
                        max_requests=max_requests_per_worker,
                        index=index,
                        events=self._events,
                        capture_dir=str(capture_dir) if capture_dir is not None else None,
                        sandbox=sandbox,
                        sandbox_readonly_paths=sandbox_readonly_paths,
                    )
                )
        except BaseException:
            for worker in self._workers:
                worker.close()
            raise
        self._gpus = list(dict.fromkeys(gpus))
        self._idle = IdleWorkers(self._workers)
        self._leases = NoopLeases() if cpu_workers is not None else GPULeases(self._gpus)
        self._replacing: dict[threading.Thread, Worker] = {}
        self._replacing_lock = threading.Lock()  # so shutdown can copy it mid-flight
        self._require_one_target()

    def submit(
        self,
        program,
        timeout: float,
        worker_wait_timeout: float = 0.0,
        request_id: str | None = None,
    ) -> SubmitOutcome:
        gpu_count = program.options.get("gpu_count")
        if gpu_count is not None and not 1 <= gpu_count <= min(8, len(self._gpus)):
            raise ValueError("gpu_count exceeds this GPU server's configured capacity")
        with self._condition:
            if self._closing.is_set():
                raise PoolBusy("server is shutting down")
        queue_started = time.monotonic()
        worker = self._idle.acquire(worker_wait_timeout)
        queue_ms = (time.monotonic() - queue_started) * 1000
        if worker is None:
            raise PoolBusy("all workers busy", queue_ms=queue_ms)
        with self._condition:
            if self._closing.is_set():
                self._idle.release(worker)
                raise PoolBusy("server is shutting down", queue_ms=queue_ms)
            self._active += 1
        if gpu_count is not None:
            return self._submit_job(worker, program, gpu_count, timeout, queue_ms, request_id)
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
        retire_reason = None
        worker.request_id = request_id
        try:
            result = worker.run(program, timeout, self._leases)
            retire_reason = result.retire_reason
            return SubmitOutcome(
                result.execution,
                worker.gpu_id,
                queue_ms,
                (time.monotonic() - run_started) * 1000,
                result.lease_wait_ms,
                result.lease_held_ms,
                worker.worker_id,
                result.finish_reason,
                self._interfered_request(worker, result.execution, request_id),
            )
        except GPUUnavailable as exc:
            raise PoolBusy(str(exc), queue_ms) from exc
        except (WorkerTimeout, WorkerCrashed) as exc:
            exc.worker_id = worker.worker_id
            exc.gpu_id = worker.gpu_id
            exc.queue_ms = queue_ms
            exc.elapsed_ms = (time.monotonic() - run_started) * 1000
            raise
        finally:
            self._leases.abandon(worker.gpu_id, worker)  # no-op unless it still holds
            worker.request_id = None
            try:
                self._release_or_replace(worker, retire_reason)
            finally:
                with self._condition:
                    self._active -= 1
                    if self._closing.is_set():
                        self._events.emit("shutdown_waiting", remaining=self._active)
                    self._condition.notify_all()

    def _submit_job(self, worker, program, count, timeout, queue_ms, request_id) -> SubmitOutcome:
        # The idle worker is an admission slot only. Its CUDA interpreter remains
        # unused; the job gets a fresh process after the entire set is reserved.
        started = time.monotonic()
        gpu_ids: tuple[int, ...] = ()
        held_since = None
        wait_ms = 0.0
        worker.request_id = request_id
        try:
            allocation = self._leases.acquire_count(count, worker, cancelled=self._closing)
            if allocation is None:
                raise PoolBusy("server is shutting down", queue_ms)
            gpu_ids, wait_ms = allocation
            held_since = time.monotonic()
            if self._closing.is_set():
                raise PoolBusy("server is shutting down", queue_ms)
            devices = ",".join(nvml.device_uuid(gpu) or str(gpu) for gpu in gpu_ids)
            self._events.emit(
                "request_routed",
                request_id=request_id,
                worker_id=worker.worker_id,
                gpu_ids=list(gpu_ids),
                gpu_count=count,
                queue_ms=queue_ms,
            )
            captures = self._events.subdir("output")
            outcome = run_job(
                program,
                self._factory,
                devices,
                timeout,
                self._termination_grace_seconds,
                capture_dir=str(captures) if captures else None,
            )
            if outcome.error is not None:
                outcome.error.pop("detected_at_ns", None)
            return SubmitOutcome(
                execution=outcome,
                gpu_id=gpu_ids[0],
                queue_ms=queue_ms,
                elapsed_ms=(time.monotonic() - started) * 1000,
                lease_wait_ms=wait_ms,
                lease_held_ms=(time.monotonic() - held_since) * 1000,
                worker_id=worker.worker_id,
                finish_reason="completed" if outcome.status == "COMPLETED" else "program_failed",
                gpu_ids=gpu_ids,
            )
        except JobCleanupError:
            # Releasing an unverified process tree could overlap the next job.
            self._leases.quarantine(gpu_ids)
            self.begin_shutdown()
            self._events.emit("gpu_job_cleanup_failed", level="ERROR", gpu_ids=list(gpu_ids))
            raise
        except GPUUnavailable as exc:
            raise PoolBusy(str(exc), queue_ms) from exc
        except (WorkerTimeout, WorkerCrashed) as exc:
            exc.worker_id = worker.worker_id
            exc.gpu_id = gpu_ids[0] if gpu_ids else None
            exc.gpu_ids = gpu_ids
            exc.queue_ms = queue_ms
            exc.elapsed_ms = (time.monotonic() - started) * 1000
            exc.lease_wait_ms = wait_ms
            exc.lease_held_ms = (time.monotonic() - held_since) * 1000 if held_since else 0.0
            raise
        finally:
            if gpu_ids:
                self._leases.release_many(gpu_ids, worker)
            worker.request_id = None
            self._idle.release(worker)
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def _interfered_request(self, worker: Worker, execution, request_id: str | None) -> str | None:
        """The request holding the GPU when this one's ``cpu_only`` call touched it."""
        error = getattr(execution, "error", None)
        if not isinstance(error, dict) or error.get("kind") != "gpu_access":
            return None
        holder = self._leases.request_at(worker.gpu_id, error.pop("detected_at_ns"))
        return None if holder == request_id else holder  # its own release may reach us late

    def _release_or_replace(self, worker: Worker, retire_reason: str | None) -> None:
        """Hand the worker back, replacing its process first if it retired.

        A retired worker has already exited, so it stays out of the idle set until
        its replacement is ready. Building it on its own thread keeps it out of the
        answered request's timings; a crash or timeout respawned in place already.
        """
        with self._condition:
            if retire_reason is None or self._closing.is_set():
                self._idle.release(worker)
                return
            thread = threading.Thread(
                target=self._replace, args=(worker, retire_reason), daemon=True
            )
            with self._replacing_lock:
                self._replacing[thread] = worker
            try:
                thread.start()
                return
            except RuntimeError:
                with self._replacing_lock:
                    self._replacing.pop(thread, None)
        # This fallback remains tracked by the active submit.
        self._replace(worker, retire_reason)

    def _replace(self, worker: Worker, reason: str) -> None:
        try:
            if self._closing.is_set():
                return
            worker.replace(self._leases, reason)
        except Exception as exc:
            if self._closing.is_set():
                return
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
                self._replacing.pop(threading.current_thread(), None)

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

    def load(self) -> dict[str, int]:
        """Read capacity and assigned/waiting request counts."""
        with self._condition:
            idle_ids, waiting = self._idle.snapshot()
            with self._replacing_lock:
                replacing = {id(worker) for worker in self._replacing.values()} - idle_ids
                capacity = len(self._workers) - len(replacing)
            return {
                "request_capacity": 0 if self._closing.is_set() else capacity,
                "requests_in_progress": self._active,
                "requests_waiting": waiting,
            }

    def worker_status(self) -> dict[str, int]:
        """Read worker occupancy for the supervisor."""
        idle_ids, _ = self._idle.snapshot()
        return {
            "worker_count": len(self._workers),
            "busy_workers": len(self._workers) - len(idle_ids),
        }

    def target(self) -> dict[str, str]:
        """What a client must compile an uploaded library for; startup rejects a
        pool whose workers would disagree."""
        return self._workers[0].target

    def versions(self) -> dict[str, str]:
        return self._workers[0].versions

    @property
    def closing(self) -> bool:
        return self._closing.is_set()

    @property
    def active_requests(self) -> int:
        with self._condition:
            return self._active

    async def shutdown_async(self) -> None:
        # The default executor may be full of submit calls blocked on workers.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kcoral-shutdown")
        try:
            await asyncio.get_running_loop().run_in_executor(executor, self.shutdown)
        finally:
            executor.shutdown(wait=False)

    def begin_shutdown(self) -> None:
        with self._condition:
            if self._closing.is_set():
                return
            self._closing.set()
            self._idle.close()
            for worker in self._workers:
                worker.begin_shutdown()
            self._events.emit("shutdown_started", remaining=self._active)

    def shutdown(self) -> None:
        self.begin_shutdown()
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            with self._condition:
                while self._active:
                    self._condition.wait()
            with self._replacing_lock:
                replacing = list(self._replacing)
            for thread in replacing:
                thread.join()
            for worker in self._workers:
                worker.close()
            self._shutdown_done = True
            self._events.emit("shutdown_complete")
