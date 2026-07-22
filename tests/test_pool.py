import threading
import time

import pytest

from benchmark_server.pool import PoolBusy, WorkerPool
from benchmark_server.schemas import Program, Run
from benchmark_server.testing import fake_runtime_factory
from benchmark_server.worker import WorkerCrashed, WorkerTimeout


def prog(*instrs):
    return Program(instructions=list(instrs))


def op(id="x"):
    return Run(id, "builtin.opaque", [])


@pytest.fixture
def pool():
    p = WorkerPool([0], fake_runtime_factory)
    yield p
    p.shutdown()


def test_pool_runs_a_program(pool):
    res = pool.submit(prog(op()), timeout=10)
    assert res[0].status == "OK" and res[0].value == {"handle": "x"}


def test_crash_replaces_worker_and_recovers(pool):
    with pytest.raises(WorkerCrashed):
        pool.submit(prog(Run("boom", "builtin.crash", [])), timeout=10)
    # worker was respawned; the next request succeeds on the fresh worker
    assert pool.submit(prog(op()), timeout=10)[0].status == "OK"


def test_timeout_kills_and_replaces_worker(pool):
    with pytest.raises(WorkerTimeout):
        pool.submit(prog(Run("s", "builtin.sleep", [5.0])), timeout=0.5)
    assert pool.submit(prog(op()), timeout=10)[0].status == "OK"


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
