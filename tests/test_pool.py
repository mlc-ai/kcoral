import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from support.programs import harness_call

from kcoral.lease import GPULeases
from kcoral.pool import PoolBusy, WorkerPool
from kcoral.schemas import GetFunction, Program, Ref, Return, Run, Upload
from kcoral.testing import fake_runtime_factory
from kcoral.worker import Worker, WorkerCrashed, WorkerTimeout, worker_main


def prog(*instrs):
    return Program(
        instructions=[
            step for item in instrs for step in (item if isinstance(item, list) else [item])
        ]
    )


def op(id="x"):
    return harness_call(id, "structural", [])


def successful_program():
    return prog(op(), Return("value", Ref("x")))


@pytest.fixture
def pool():
    p = WorkerPool([0], fake_runtime_factory, sandbox="none", max_requests_per_worker=0)
    yield p
    p.shutdown()


def test_pool_runs_a_program(pool):
    outcome = pool.submit(successful_program(), timeout=10)
    assert outcome.execution.status == "COMPLETED"
    assert outcome.execution.results["value"]["type"] == "object"
    assert outcome.gpu_id == 0
    assert outcome.queue_ms >= 0 and outcome.elapsed_ms >= 0


def test_cpu_pool_runs_without_gpus_or_leases():
    pool = WorkerPool(
        [], fake_runtime_factory, sandbox="none", cpu_workers=2, max_requests_per_worker=0
    )
    try:
        outcome = pool.submit(successful_program(), timeout=10)
        health = pool.health()
    finally:
        pool.shutdown()

    assert outcome.execution.status == "COMPLETED"
    assert outcome.gpu_id is None
    assert outcome.lease_wait_ms == 0
    assert outcome.lease_held_ms == 0
    assert health["gpus"] == []
    assert len(health["workers"]) == 2
    assert all(worker["gpu_id"] is None for worker in health["workers"])


def test_cpu_timeout_respawns_without_a_gpu():
    pool = WorkerPool(
        [], fake_runtime_factory, sandbox="none", cpu_workers=1, max_requests_per_worker=0
    )
    try:
        with pytest.raises(WorkerTimeout) as exc_info:
            pool.submit(prog(*harness_call("sleep", "sleep", [2.0])), timeout=0.2)
        recovered = pool.submit(successful_program(), timeout=10)
    finally:
        pool.shutdown()

    assert exc_info.value.gpu_id is None
    assert recovered.execution.status == "COMPLETED"
    assert recovered.lease_wait_ms == 0
    assert recovered.lease_held_ms == 0


def test_timeout_removes_the_parent_owned_request_workspace(pool, tmp_path):
    marker = tmp_path / "workspace-path"
    source = (
        "import os, time\n"
        "def main():\n"
        f"    with open({str(marker)!r}, 'w') as marker:\n"
        "        marker.write(os.getcwd())\n"
        "    time.sleep(10)\n"
    )
    program = prog(
        Upload("module", "module", source=source),
        GetFunction("fn", Ref("module"), "main"),
        Run("call", Ref("fn"), []),
    )

    with pytest.raises(WorkerTimeout):
        pool.submit(program, timeout=0.5)

    workspace = marker.read_text()
    assert os.path.basename(workspace).startswith("kcoral-program-")
    assert not os.path.exists(workspace)


def test_crash_replaces_worker_and_recovers(pool):
    with pytest.raises(WorkerCrashed) as exc_info:
        pool.submit(prog(*harness_call("boom", "crash", [])), timeout=10)
    assert exc_info.value.gpu_id == 0 and exc_info.value.elapsed_ms >= 0
    assert exc_info.value.instruction_index == 2 and exc_info.value.exitcode == 1
    # worker was respawned; the next request succeeds on the fresh worker
    assert pool.submit(successful_program(), timeout=10).execution.status == "COMPLETED"


def settled(pool):
    """Wait until background replacement finishes."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with pool._replacing_lock:
            if not pool._replacing:
                return pool
        time.sleep(0.01)
    raise AssertionError("a worker replacement never finished")


def test_poisoned_context_replaces_worker_and_recovers(pool):
    original_pid = pool._workers[0]._proc.pid
    outcome = pool.submit(prog(*harness_call("bad", "poison", [])), timeout=10)
    assert outcome.execution.status == "FAILED"
    assert outcome.execution.error["kind"] == "runtime"
    assert outcome.execution.error["message"] == "simulated illegal memory access"
    assert outcome.finish_reason == "program_failed"
    assert settled(pool)._workers[0]._proc.pid != original_pid
    assert pool.submit(successful_program(), timeout=10).execution.status == "COMPLETED"


def test_last_error_fails_current_request_without_replacing_worker(pool):
    original_pid = pool._workers[0]._proc.pid
    outcome = pool.submit(prog(*harness_call("bad", "stale_cuda_error", [])), timeout=10)

    assert outcome.execution.status == "FAILED"
    assert outcome.execution.error["kind"] == "runtime"
    assert "cudaErrorInvalidValue" in outcome.execution.error["message"]
    assert outcome.finish_reason == "program_failed"
    assert pool._workers[0]._proc.pid == original_pid
    assert pool.submit(successful_program(), timeout=10).execution.status == "COMPLETED"


@pytest.mark.parametrize("fails", [False, True])
def test_default_request_limit_replaces_worker_after_preserving_outcome(fails):
    pool = WorkerPool([0], fake_runtime_factory, sandbox="none")
    try:
        original_pid = pool._workers[0]._proc.pid
        program = (
            prog(*harness_call("bad", "stale_cuda_error", [])) if fails else successful_program()
        )
        outcome = pool.submit(program, timeout=10)

        assert outcome.execution.status == ("FAILED" if fails else "COMPLETED")
        assert outcome.finish_reason == ("program_failed" if fails else "completed")
        assert not hasattr(outcome, "retire_reason")
        assert settled(pool)._workers[0]._proc.pid != original_pid
    finally:
        pool.shutdown()


class _FailedPipe:
    def __init__(self, exc):
        self._exc = exc

    def send(self, _message):
        raise self._exc


class _ExitedProcess:
    exitcode = 1

    def join(self, timeout=None):
        pass


@pytest.mark.parametrize(
    "pipe_error",
    [EOFError("eof"), ConnectionResetError("reset"), BrokenPipeError("broken"), OSError("io")],
    ids=["eof", "connection-reset", "broken-pipe", "os-error"],
)
def test_run_replaces_worker_on_pipe_failures(pipe_error):
    worker = object.__new__(Worker)
    worker._sandbox_mode = "none"
    worker._sandbox = None
    worker.gpu_id = 0
    worker._conn = _FailedPipe(pipe_error)
    worker._proc = _ExitedProcess()
    replacements = []
    worker._abandon_and_respawn = lambda leases, reason: replacements.append((leases, reason))
    leases = GPULeases([0])

    with pytest.raises(WorkerCrashed) as exc_info:
        worker.run(successful_program(), timeout=10, leases=leases)

    assert exc_info.value.__cause__ is pipe_error
    assert exc_info.value.exitcode == 1
    assert replacements == [(leases, "crashed")]


def test_worker_prepares_before_parent_grants_gpu_initialization(monkeypatch):
    events = []

    class Connection:
        incoming = iter([{"__startup__": "initialize"}, None])

        def send(self, message):
            events.append(("send", message))

        def recv(self):
            message = next(self.incoming)
            events.append(("recv", message))
            return message

    class Factory:
        def prepare(self):
            events.append(("prepare", None))

            def initialize():
                events.append(("initialize", None))
                return fake_runtime_factory()

            return initialize

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "before-test")
    monkeypatch.setattr("kcoral.worker.os.setsid", lambda: None)
    worker_main("GPU-abc123", Connection(), Factory(), max_requests=0)

    # The parent picks the card; what the server was launched with is gone.
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-abc123"

    assert events[:5] == [
        ("prepare", None),
        ("send", {"__startup__": "prepared"}),
        ("recv", {"__startup__": "initialize"}),
        ("initialize", None),
        (
            "send",
            {
                "__ready__": {
                    "target": {"arch": "fake"},
                    "versions": {},
                    "device_uuid": None,
                }
            },
        ),
    ]


def test_cpu_worker_does_not_change_visible_devices(monkeypatch):
    class Connection:
        incoming = iter([{"__startup__": "initialize"}, None])

        def send(self, message):
            pass

        def recv(self):
            return next(self.incoming)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unchanged")
    monkeypatch.setattr("kcoral.worker.os.setsid", lambda: None)
    worker_main(None, Connection(), fake_runtime_factory, max_requests=0)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "unchanged"


def test_replacement_kills_old_process_then_prepares_off_gpu_and_initializes_under_lease():
    worker = object.__new__(Worker)
    worker.gpu_id = 0
    leases = GPULeases([0])
    leases.acquire(0, worker)
    events = []

    def kill():
        assert leases._holder[0] is worker
        events.append("kill")

    def prepare():
        assert leases._holder[0] is None
        events.append("prepare")

    def initialize():
        assert leases._holder[0] is worker
        events.append("initialize")

    worker._kill = kill
    worker._start_process = prepare
    worker._initialize_process = initialize
    worker._abandon_and_respawn(leases, "request_limit")

    assert events == ["kill", "prepare", "initialize"]
    assert leases.depth(0) == 0


def test_replacement_releases_gpu_when_respawn_fails():
    worker = object.__new__(Worker)
    worker.gpu_id = 0
    leases = GPULeases([0])
    leases.acquire(0, worker)
    worker._kill = lambda: None

    worker._start_process = lambda: None

    def fail_initialize():
        assert leases._holder[0] is worker
        raise RuntimeError("initialize failed")

    worker._initialize_process = fail_initialize
    with pytest.raises(RuntimeError, match="initialize failed"):
        worker._abandon_and_respawn(leases, "request_limit")

    assert leases.depth(0) == 0


def test_replacement_does_not_claim_gpu_when_preparation_fails():
    worker = object.__new__(Worker)
    worker.gpu_id = 0
    leases = GPULeases([0])
    leases.acquire(0, worker)
    worker._kill = lambda: None

    def fail_prepare():
        raise RuntimeError("prepare failed")

    worker._start_process = fail_prepare

    with pytest.raises(RuntimeError, match="prepare failed"):
        worker._abandon_and_respawn(leases, "request_limit")

    assert leases.depth(0) == 0


def test_timeout_kills_and_replaces_worker(pool):
    with pytest.raises(WorkerTimeout) as exc_info:
        pool.submit(prog(*harness_call("s", "sleep", [5.0])), timeout=0.5)
    assert exc_info.value.gpu_id == 0 and exc_info.value.elapsed_ms >= 500
    assert pool.submit(successful_program(), timeout=10).execution.status == "COMPLETED"


def test_health_reports_idle_workers(pool):
    pool.submit(successful_program(), timeout=10)
    health = pool.health()
    assert health["queue_length"] == 0
    assert health["workers"] == [
        {
            "worker_id": "gpu0/w0",
            "gpu_id": 0,
            "status": "idle",
            "uptime_seconds": health["workers"][0]["uptime_seconds"],
        }
    ]
    assert health["workers"][0]["uptime_seconds"] >= 0
    # the pool serves one target; versions are reported separately
    assert health["target"] == {"arch": "fake"} and health["versions"] == {}


def test_load_tracks_assigned_and_waiting_requests(pool, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original_run = pool._workers[0].run

    def blocked_run(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(pool._workers[0], "run", blocked_run)
    with ThreadPoolExecutor(2) as executor:
        first = executor.submit(pool.submit, successful_program(), 10)
        try:
            assert entered.wait(5)
            second = executor.submit(pool.submit, successful_program(), 10, 10)
            deadline = time.monotonic() + 5
            while pool.load()["requests_waiting"] != 1:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert pool.load() == {
                "request_capacity": 1,
                "requests_in_progress": 1,
                "requests_waiting": 1,
            }
        finally:
            release.set()
        assert first.result(timeout=10).execution.status == "COMPLETED"
        assert second.result(timeout=10).execution.status == "COMPLETED"
    assert pool.load() == {"request_capacity": 1, "requests_in_progress": 0, "requests_waiting": 0}


def test_load_during_background_worker_replacement(monkeypatch):
    pool = WorkerPool(
        [], fake_runtime_factory, sandbox="none", cpu_workers=1, max_requests_per_worker=1
    )
    entered, release = threading.Event(), threading.Event()
    worker = pool._workers[0]
    original_replace = worker.replace

    def blocked_replace(*args):
        entered.set()
        assert release.wait(10)
        original_replace(*args)

    monkeypatch.setattr(worker, "replace", blocked_replace)
    try:
        assert pool.submit(Program(instructions=[]), 10).execution.status == "COMPLETED"
        assert entered.wait(5)
        assert pool.load() == {
            "request_capacity": 0,
            "requests_in_progress": 0,
            "requests_waiting": 0,
        }
        assert pool.worker_status() == {"worker_count": 1, "busy_workers": 1}
        release.set()
        deadline = time.monotonic() + 5
        while pool.load()["request_capacity"] != 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert pool.worker_status() == {"worker_count": 1, "busy_workers": 0}
    finally:
        release.set()
        pool.shutdown()


def test_backpressure_when_all_workers_busy(pool):
    def slow():
        try:
            pool.submit(prog(*harness_call("s", "sleep", [1.0])), timeout=10)
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
    p = WorkerPool(
        [0], fake_runtime_factory, sandbox="none", workers_per_gpu=2, max_requests_per_worker=0
    )
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
    busy = prog(*harness_call("s", "sleep", [1.0]))
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


def test_a_cpu_only_function_hands_the_gpu_over(shared_gpu_pool):
    """The point of the whole arrangement: a worker compiling holds no GPU, so a
    second one measures during it rather than queueing behind it."""
    results: dict[str, object] = {}
    compiling = prog(op("warm"), *harness_call("c", "cpu_sleep", [1.0]), op("after"))
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
        shared_gpu_pool.submit(prog(*harness_call("s", "sleep", [10.0])), timeout=0.5)
    assert shared_gpu_pool.health()["gpus"] == [{"gpu_id": 0, "lease_depth": 0}]
    # The other worker can still reach the GPU.
    assert shared_gpu_pool.submit(successful_program(), timeout=10).execution.status == "COMPLETED"


def test_poison_replaces_only_the_corresponding_worker(shared_gpu_pool):
    before = [worker._proc.pid for worker in shared_gpu_pool._workers]
    outcome = shared_gpu_pool.submit(prog(*harness_call("bad", "poison", [])), timeout=10)
    after = [worker._proc.pid for worker in settled(shared_gpu_pool)._workers]

    assert outcome.execution.error["kind"] == "runtime"
    assert outcome.finish_reason == "program_failed"
    assert sum(old != new for old, new in zip(before, after, strict=True)) == 1


class _LeaseRecorder:
    """Wraps a pool's leases to log every grant and release the parent makes."""

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()
        self.log: list[str] = []

    def initialization(self, gpu_id):
        return self._inner.initialization(gpu_id)

    def acquire(self, gpu_id, holder):
        # Reacquiring ownership does not start a new lease interval.
        with self._inner._lock:
            already_owned = self._inner._holder[gpu_id] is holder
        waited = self._inner.acquire(gpu_id, holder)
        if not already_owned:
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
    pool = WorkerPool(
        [0], fake_runtime_factory, sandbox="none", workers_per_gpu=4, max_requests_per_worker=0
    )
    recorder = _LeaseRecorder(pool._leases)
    pool._leases = recorder
    finished, killed, errors = [], [], []

    def submit(i):
        def send(program, timeout):
            return pool.submit(program, timeout=timeout, worker_wait_timeout=30)

        try:
            if i % 6 == 0:  # dies holding the GPU
                send(prog(*harness_call(f"x{i}", "crash", [])), 30)
            elif i % 11 == 0:  # hangs holding it, and is killed
                send(prog(*harness_call(f"h{i}", "sleep", [30.0])), 0.5)
            else:
                program = prog(
                    *harness_call(f"c{i}", "cpu_sleep", [0.02]),
                    *harness_call(f"g{i}", "sleep", [0.01]),
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
            assert 0 <= held <= 1, "unbalanced or overlapping GPU ownership"
        assert held == 0, "a worker failed to release GPU ownership"
        assert recorder.depth(0) == 0, "a killed worker stranded the GPU"
    finally:
        pool.shutdown()


def test_worker_refuses_a_process_that_came_up_on_another_gpu():
    """A mis-pinned worker would measure someone else's card under our id."""
    worker = object.__new__(Worker)
    worker.gpu_id = 3
    worker._kill = lambda: None
    worker._expected_uuid = "GPU-1111-2222"

    worker.device_uuid = "1111-2222"  # torch spells it without the prefix
    worker._require_expected_device()  # same card: accepted

    worker.device_uuid = "3333-4444"
    with pytest.raises(WorkerCrashed, match="not the requested"):
        worker._require_expected_device()


def test_parent_waits_for_retiring_process_exit_before_releasing_gpu(tmp_path):
    from collections import deque

    reason = "request_limit"

    class Connection:
        def __init__(self):
            self.messages = deque(
                [
                    {"__lease__": "acquire"},
                    {"__outcome__": "outcome", "__retire_reason__": reason},
                ]
            )

        def send(self, message):
            pass

        def poll(self, timeout):
            return bool(self.messages)

        def recv(self):
            return self.messages.popleft()

    class Leases:
        held = False

        def acquire(self, gpu_id, holder):
            self.held = True
            return 0.0

        def release(self, gpu_id, holder):
            self.held = False

    leases = Leases()
    observed = []
    worker = object.__new__(Worker)
    worker.gpu_id = 0
    worker._conn = Connection()
    worker._kill = lambda: observed.append(leases.held)
    result = worker._run_in_workspace(None, 10, leases, str(tmp_path))
    assert result.retire_reason == reason
    assert observed == [True]
    assert not leases.held


@pytest.mark.parametrize("gpu_ids", [(0,), (0, 2)])
def test_cpu_timeout_waits_for_gpu_before_destroying_live_context(gpu_ids):
    from types import SimpleNamespace

    leases = GPULeases(list(gpu_ids))
    peer = object()
    leases.acquire(gpu_ids[-1], peer)
    worker = object.__new__(Worker)
    worker.gpu_id = 0
    worker._gpu_ids = gpu_ids
    worker._proc = SimpleNamespace(is_alive=lambda: True)
    worker._closing = threading.Event()
    observed = []
    worker._kill = lambda: observed.append(all(leases._holder[gpu] is worker for gpu in gpu_ids))
    worker._start_process = lambda: None
    worker._initialize_process = lambda: None
    thread = threading.Thread(
        target=worker._abandon_and_respawn, args=(leases, "timeout"), daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while leases.depth(gpu_ids[-1]) < 2 and not observed:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert not observed, "Context destruction must wait for the peer GPU stage"
    finally:
        leases.release(gpu_ids[-1], peer)
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert observed == [True]
