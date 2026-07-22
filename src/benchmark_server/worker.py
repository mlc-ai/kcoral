"""The GPU worker: a child process that owns one GPU's CUDA context.

The front-end never touches the GPU — it dispatches a Program to a worker over a
pipe and reads back results. A worker runs one program at a time and clears its
per-request state after each (via the Runtime). The worker's `main` is the child
entry point; :class:`Worker` is the parent-side handle with crash/timeout kill +
respawn.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable

from .engine import execute


def worker_main(gpu_id: int, conn, runtime_factory: Callable) -> None:
    """Child entry point. Pins the GPU, builds the Runtime, serves programs."""
    # Select the GPU before the Runtime imports torch/tvm, so it sees one device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        runtime = runtime_factory()
    except Exception as exc:  # runtime init failed — report and exit
        conn.send({"__error__": f"runtime init failed: {type(exc).__name__}: {exc}"})
        return
    conn.send("__ready__")
    while True:
        try:
            program = conn.recv()
        except EOFError:
            return
        if program is None:  # shutdown signal
            return
        conn.send(execute(program, runtime))


class WorkerCrashed(Exception):
    pass


class WorkerTimeout(Exception):
    pass


class Worker:
    """Parent-side handle to one GPU worker process."""

    def __init__(self, gpu_id: int, runtime_factory: Callable, spawn_timeout: float = 60.0) -> None:
        self.gpu_id = gpu_id
        self._factory = runtime_factory
        self._spawn_timeout = spawn_timeout
        self._ctx = mp.get_context("spawn")  # 'spawn' — 'fork' is unsafe with CUDA
        self._spawn()

    def _spawn(self) -> None:
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
        if msg != "__ready__":
            self._kill()
            raise WorkerCrashed(f"worker init error: {msg}")

    def run(self, program, timeout: float):
        """Run a program; kill+respawn on timeout or crash, then re-raise."""
        self._conn.send(program)
        if not self._conn.poll(timeout):  # no answer by the deadline -> hung
            self._kill_and_respawn()
            raise WorkerTimeout(f"program exceeded {timeout}s on GPU {self.gpu_id}")
        try:
            return self._conn.recv()
        except EOFError:  # child died mid-run
            self._kill_and_respawn()
            raise WorkerCrashed(f"worker on GPU {self.gpu_id} crashed")

    def _kill(self) -> None:
        try:
            self._proc.kill()
            self._proc.join(timeout=5)
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
