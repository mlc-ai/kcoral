import threading
import time

import pytest

from benchmark_server.pool import PoolBusy, WorkerPool
from benchmark_server.schemas import Program, Return, Run
from benchmark_server.testing import fake_runtime_factory
from benchmark_server.worker import WorkerCrashed, WorkerTimeout


def prog(*instrs):
    return Program(instructions=list(instrs))


def op(id="x"):
    return Run(id, "builtin.structural", [])


def successful_program():
    return prog(op(), Return("value", {"$ref": "x"}))


@pytest.fixture
def pool():
    p = WorkerPool([0], fake_runtime_factory)
    yield p
    p.shutdown()


def test_pool_runs_a_program(pool):
    outcome = pool.submit(successful_program(), timeout=10)
    assert outcome.execution.status == "COMPLETED"
    assert outcome.execution.results["value"]["type"] == "object"
    assert outcome.gpu_id == 0
    assert outcome.queue_ms >= 0 and outcome.elapsed_ms >= 0


def test_crash_replaces_worker_and_recovers(pool):
    with pytest.raises(WorkerCrashed) as exc_info:
        pool.submit(prog(Run("boom", "builtin.crash", [])), timeout=10)
    assert exc_info.value.gpu_id == 0 and exc_info.value.elapsed_ms >= 0
    # worker was respawned; the next request succeeds on the fresh worker
    assert pool.submit(successful_program(), timeout=10).execution.status == "COMPLETED"


def test_timeout_kills_and_replaces_worker(pool):
    with pytest.raises(WorkerTimeout) as exc_info:
        pool.submit(prog(Run("s", "builtin.sleep", [5.0])), timeout=0.5)
    assert exc_info.value.gpu_id == 0 and exc_info.value.elapsed_ms >= 500
    assert pool.submit(successful_program(), timeout=10).execution.status == "COMPLETED"


def test_health_reports_idle_workers(pool):
    pool.submit(successful_program(), timeout=10)
    health = pool.health()
    assert health["queue_length"] == 0
    assert health["workers"] == [
        {
            "gpu_id": 0,
            "status": "idle",
            "uptime_seconds": health["workers"][0]["uptime_seconds"],
        }
    ]
    assert health["workers"][0]["uptime_seconds"] >= 0


def test_backpressure_when_all_workers_busy(pool):
    def slow():
        try:
            pool.submit(prog(Run("s", "builtin.sleep", [1.0])), timeout=10)
        except Exception:
            pass

    t = threading.Thread(target=slow)
    t.start()
    time.sleep(0.3)  # let the slow job occupy the only worker
    with pytest.raises(PoolBusy):
        pool.submit(prog(op()), timeout=10, worker_wait_timeout=0)
    t.join()
