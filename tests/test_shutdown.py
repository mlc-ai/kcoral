import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from kcoral.pool import PoolBusy, WorkerPool
from kcoral.schemas import Program, Run
from kcoral.testing import fake_runtime_factory
from kcoral.worker import WorkerTimeout


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def running(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_shutdown_drains_active_job_and_rejects_waiter():
    pool = WorkerPool([], fake_runtime_factory, cpu_workers=1)
    with ThreadPoolExecutor(2) as executor:
        try:
            active = executor.submit(
                pool.submit, Program(instructions=[Run("sleep", "builtin.sleep", [0.3])]), 10
            )
            wait_until(lambda: pool.active_requests == 1)
            queued = executor.submit(pool.submit, Program(instructions=[]), 10, 30)
            wait_until(lambda: pool._idle.snapshot()[1] == 1)
            pool.begin_shutdown()
            with pytest.raises(PoolBusy):
                queued.result(timeout=1)
            with pytest.raises(PoolBusy, match="shutting down"):
                pool.submit(Program(instructions=[]), 10)
            pool.shutdown()
            assert active.result().execution.status == "COMPLETED"
            assert pool._workers[0].generation == 1
            assert not running(pool._workers[0].pid)
        finally:
            pool.shutdown()


def test_shutdown_interrupts_replacement_initialization(monkeypatch):
    pool = WorkerPool([], fake_runtime_factory, cpu_workers=1, termination_grace_seconds=0.1)
    worker = pool._workers[0]
    entered, release = threading.Event(), threading.Event()
    initialize = worker._initialize_process

    def paused_initialize():
        entered.set()
        assert release.wait(5)
        initialize()

    monkeypatch.setattr(worker, "_initialize_process", paused_initialize)
    try:
        pool.submit(Program(instructions=[]), 10)
        assert entered.wait(5)
        pid = worker.pid
        pool.begin_shutdown()
        release.set()
        pool.shutdown()
        assert not running(pid)
        assert not pool._replacing
    finally:
        release.set()
        pool.shutdown()


def test_cancelled_future_still_drains_with_a_saturated_executor():
    pool = WorkerPool([], fake_runtime_factory, cpu_workers=1)

    async def exercise():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        future = asyncio.get_running_loop().run_in_executor(
            None, pool.submit, Program(instructions=[Run("sleep", "builtin.sleep", [0.4])]), 10
        )
        while not pool.active_requests:
            await asyncio.sleep(0.01)
        future.cancel()
        assert pool.active_requests == 1
        await asyncio.wait_for(pool.shutdown_async(), timeout=5)
        assert pool.active_requests == 0
        assert pool._workers[0].generation == 1
        assert not running(pool._workers[0].pid)

    try:
        asyncio.run(exercise())
    finally:
        pool.shutdown()


def test_shutdown_respects_request_timeouts_and_gpu_lease_waiters():
    pool = WorkerPool([0], fake_runtime_factory, workers_per_gpu=2, termination_grace_seconds=0.1)
    with ThreadPoolExecutor(2) as executor:
        try:
            jobs = [
                executor.submit(
                    pool.submit, Program(instructions=[Run("sleep", "builtin.sleep", [30])]), 0.3
                )
                for _ in range(2)
            ]
            wait_until(lambda: pool._leases.depth(0) == 2)
            pool.shutdown()
            for job in jobs:
                with pytest.raises(WorkerTimeout):
                    job.result(timeout=1)
            assert pool._leases.depth(0) == 0
            assert all(w.generation == 1 and not running(w.pid) for w in pool._workers)
        finally:
            pool.shutdown()


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_real_server_reports_progress_and_drains_through_repeated_signals(tmp_path, sig):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    script = f"""
from pathlib import Path
import uvicorn
from kcoral.app import create_app
from kcoral.config import ServerConfig
from kcoral.shutdown import ShutdownServer
from kcoral.testing import fake_runtime_factory
app = create_app(ServerConfig(device="cpu", num_workers=2, log_dir=Path({str(tmp_path)!r})),
                 runtime_factory=fake_runtime_factory)
ShutdownServer(uvicorn.Config(app, host="127.0.0.1", port={port}), app).run()
"""
    output_path = tmp_path / "console.txt"
    with output_path.open("w") as output:
        server = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pids = []

    def events():
        files = list(tmp_path.glob("runs/*/events.jsonl"))
        if not files:
            return []
        rows = []
        for line in files[0].read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return rows

    try:

        def healthy():
            try:
                return httpx.get(f"http://127.0.0.1:{port}/health").status_code == 200
            except httpx.TransportError:
                return False

        wait_until(healthy)
        pids = [row["pid"] for row in events() if row["event"] == "worker_ready"]

        def post_sleep(duration):
            instruction = {"op": "run", "id": "sleep", "fn": "builtin.sleep", "args": [duration]}
            program = json.dumps({"instructions": [instruction]})
            return httpx.post(
                f"http://127.0.0.1:{port}/execute",
                files={"program": (None, program, "application/json")},
                timeout=10,
            )

        with ThreadPoolExecutor(2) as executor:
            responses = [executor.submit(post_sleep, duration) for duration in (0.6, 1.5)]
            wait_until(lambda: sum(row["event"] == "request_routed" for row in events()) == 2)
            server.send_signal(sig)
            wait_until(lambda: any(row["event"] == "shutdown_started" for row in events()))
            for _ in range(3):
                server.send_signal(sig)
                time.sleep(0.03)
            server.wait(timeout=10)
            for response in responses:
                result = response.result()
                assert result.status_code == 200
                assert result.json()["status"] == "COMPLETED"
        logs = output_path.read_text()
        assert server.returncode == 0, logs
        assert all(not running(pid) for pid in pids)
        assert "Finishing remaining benchmarks: 2 left." in logs
        assert "Finishing remaining benchmarks: 1 left." in logs
        assert "force quit" not in logs
        assert "KeyboardInterrupt" not in logs
        assert "CancelledError" not in logs
        assert "Traceback" not in logs
        rows = events()
        assert any(row["event"] == "shutdown_complete" for row in rows)
        shutdown_index = next(i for i, row in enumerate(rows) if row["event"] == "shutdown_started")
        assert not any(row["event"] == "worker_ready" for row in rows[shutdown_index:])
    finally:
        if server.poll() is None:
            server.kill()
            server.wait()
        for pid in pids:
            if running(pid):
                os.kill(pid, signal.SIGKILL)
