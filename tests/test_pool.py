import threading
import time

import pytest

from benchmark_server.pool import PoolBusy, WorkerPool
from benchmark_server.schemas import Program, Ref, Return, Run
from benchmark_server.testing import fake_runtime_factory
from benchmark_server.worker import WorkerCrashed, WorkerTimeout


def prog(*instrs):
    return Program(instructions=list(instrs))


def op(id="x"):
    return Run(id, "builtin.structural", [])


def successful_program():
    return prog(op(), Return("value", Ref("x")))


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
    assert exc_info.value.instruction_index == 0 and exc_info.value.exitcode == 1
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
    # the pool serves one target; versions are reported separately
    assert health["target"] == {"arch": "fake"} and health["versions"] == {}


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


@pytest.fixture
def shared_gpu_pool():
    """Two workers on one GPU, so they must take turns through its lease."""
    p = WorkerPool([0], fake_runtime_factory, workers_per_gpu=2)
    yield p
    p.shutdown()


def _submit_into(results, key, pool, program, **kwargs):
    def run():
        try:
            results[key] = pool.submit(program, timeout=20, **kwargs)
        except Exception as exc:  # recorded so the assertion names it
            results[key] = exc

    return threading.Thread(target=run)


def test_gpu_work_is_exclusive_across_workers_on_one_gpu(shared_gpu_pool):
    """Two workers, one GPU: the second waits out the first rather than sharing."""
    results: dict[str, object] = {}
    busy = prog(Run("s", "builtin.sleep", [1.0]))
    holder = _submit_into(results, "holder", shared_gpu_pool, busy)
    holder.start()
    time.sleep(0.3)  # let it take the lease
    waiter = _submit_into(results, "waiter", shared_gpu_pool, successful_program())
    waiter.start()
    holder.join(), waiter.join()

    assert results["holder"].execution.status == "COMPLETED"
    assert results["waiter"].execution.status == "COMPLETED"
    assert results["holder"].gpu_id == results["waiter"].gpu_id == 0
    # It got a worker at once and then waited on the GPU, not the other way round.
    assert results["waiter"].queue_ms < 300
    assert results["waiter"].lease_wait_ms > 400


def test_a_cpu_only_builtin_hands_the_gpu_over(shared_gpu_pool):
    """The point of the whole arrangement: a worker compiling holds no GPU, so a
    second one measures during it rather than queueing behind it."""
    results: dict[str, object] = {}
    compiling = prog(op("warm"), Run("c", "builtin.cpu_sleep", [1.0]), op("after"))
    first = _submit_into(results, "compiling", shared_gpu_pool, compiling)
    first.start()
    time.sleep(0.3)  # into the cpu_sleep, where the lease is dropped
    second = _submit_into(results, "measuring", shared_gpu_pool, successful_program())
    second.start()
    first.join(), second.join()

    assert results["compiling"].execution.status == "COMPLETED"
    assert results["measuring"].execution.status == "COMPLETED"
    assert results["measuring"].lease_wait_ms < 300  # did not wait out the cpu phase
    # The compile ran off the GPU, so the lease was held for far less than the run.
    assert results["compiling"].lease_held_ms < 500
    assert results["compiling"].elapsed_ms > 1000


def test_killing_a_worker_frees_the_gpu_it_held(shared_gpu_pool):
    """A worker killed mid-program cannot release its own lease, so the parent
    does it - otherwise that GPU would be stranded for good."""
    with pytest.raises(WorkerTimeout):
        shared_gpu_pool.submit(prog(Run("s", "builtin.sleep", [10.0])), timeout=0.5)
    assert shared_gpu_pool.health()["gpus"] == [{"gpu_id": 0, "lease_depth": 0}]
    # The other worker can still reach the GPU.
    assert shared_gpu_pool.submit(successful_program(), timeout=10).execution.status == "COMPLETED"


class _LeaseRecorder:
    """Wraps a pool's leases to log every grant and release the parent makes."""

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()
        self.log: list[str] = []

    def acquire(self, gpu_id, holder):
        waited = self._inner.acquire(gpu_id, holder)
        with self._lock:
            self.log.append("grant")
        return waited

    def release(self, gpu_id, holder):
        self._record_if_held(gpu_id, holder)
        self._inner.release(gpu_id, holder)

    def abandon(self, gpu_id, holder):
        self._record_if_held(gpu_id, holder)
        self._inner.abandon(gpu_id, holder)

    def _record_if_held(self, gpu_id, holder):
        with self._lock:
            if self._inner._holder[gpu_id] is holder:
                self.log.append("release")

    def depth(self, gpu_id):
        return self._inner.depth(gpu_id)


def test_lease_invariants_hold_while_workers_die_under_load():
    """Concurrency and the failure paths together: workers killed mid-program
    cannot leave a GPU held, and no two ever hold one at once."""
    pool = WorkerPool([0], fake_runtime_factory, workers_per_gpu=4)
    recorder = _LeaseRecorder(pool._leases)
    pool._leases = recorder
    finished, killed, errors = [], [], []

    def submit(i):
        def send(program, timeout):
            return pool.submit(program, timeout=timeout, worker_wait_timeout=30)

        try:
            if i % 6 == 0:  # dies holding the GPU
                send(prog(Run(f"x{i}", "builtin.crash", [])), 30)
            elif i % 11 == 0:  # hangs holding it, and is killed
                send(prog(Run(f"h{i}", "builtin.sleep", [30.0])), 0.5)
            else:
                program = prog(
                    Run(f"c{i}", "builtin.cpu_sleep", [0.02]),
                    Run(f"g{i}", "builtin.sleep", [0.01]),
                    Return("v", Ref(f"g{i}")),
                )
                finished.append(send(program, 30).execution.status)
        except (WorkerCrashed, WorkerTimeout):
            killed.append(i)
        except Exception as exc:  # recorded so a failure names itself
            errors.append(repr(exc))

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        assert not errors, errors
        assert len(finished) + len(killed) == 24 and killed  # failures really ran
        held = 0
        for event in recorder.log:
            held += 1 if event == "grant" else -1
            assert held <= 1, "two workers held one GPU at once"
        assert recorder.depth(0) == 0, "a killed worker stranded the GPU"
    finally:
        pool.shutdown()
