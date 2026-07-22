"""Worker pool: one worker per GPU, idle-worker assignment, backpressure.

`submit` blocks (call it from a thread), acquires an idle worker, runs the
program, and returns the worker to the idle set — respawned already if it crashed
or timed out. When no worker becomes free within ``worker_wait_timeout`` it raises
:class:`PoolBusy` (the front-end maps that to HTTP 503).
"""

from __future__ import annotations

import queue
from typing import Callable

from .worker import Worker, WorkerCrashed, WorkerTimeout  # noqa: F401  (re-exported)


class PoolBusy(Exception):
    pass


class WorkerPool:
    def __init__(self, gpus: list[int], runtime_factory: Callable) -> None:
        self._workers = [Worker(g, runtime_factory) for g in gpus]
        self._idle: "queue.Queue[Worker]" = queue.Queue()
        for w in self._workers:
            self._idle.put(w)

    def submit(self, program, timeout: float, worker_wait_timeout: float = 0.0):
        try:
            if worker_wait_timeout > 0:
                worker = self._idle.get(timeout=worker_wait_timeout)
            else:
                worker = self._idle.get_nowait()
        except queue.Empty:
            raise PoolBusy("all workers busy")
        try:
            return worker.run(program, timeout)
        finally:
            self._idle.put(worker)  # worker was respawned in-place on crash/timeout

    def shutdown(self) -> None:
        for w in self._workers:
            w.close()
