"""Real process supervision and scheduling, using a GPU-free execution runtime."""

import json
import os
import runpy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from support.runtime import fake_runtime_factory

from kcoral import Program
from kcoral.config import ServerConfig
from kcoral.protocol import parse_program
from kcoral.runtime.lease import GPULeases, GPUUnavailable, NoopLeases
from kcoral.runtime.pool import PoolBusy, WorkerPool
from kcoral.runtime.worker import Worker, WorkerCleanupError, WorkerCrashed, WorkerTimeout
from kcoral.server.app import create_app


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def build(source, args=(), *, count=2):
    p = Program()
    module = p.upload(id="module", kind="module", source=source)
    fn = p.get_function(id="fn", module=module, name="main")
    result = p.run(id="result", fn=fn, args=list(args))
    p.return_(key="result", value=result)
    parsed = parse_program(
        {"instructions": p.instructions, "options": {} if count is None else {"gpu_count": count}}
    )
    parsed.blob_bytes = p._blobs.copy()
    return parsed


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr("kcoral.runtime.worker.nvml.device_uuid", lambda _: None)
    with_pool = WorkerPool(
        [0, 2, 4, 6],
        fake_runtime_factory,
        sandbox="none",
        max_requests_per_worker=0,
        termination_grace_seconds=0.1,
    )
    yield with_pool
    with_pool.shutdown()


def test_job_executes_once_with_all_devices_and_no_rank_environment(pool, monkeypatch):
    monkeypatch.setenv("RANK", "99")
    monkeypatch.setenv("MASTER_PORT", "12345")
    program = build(
        """
def main():
    import os
    print("one invocation")
    return [os.environ["CUDA_VISIBLE_DEVICES"], os.getenv("RANK"), os.getenv("MASTER_PORT")]
""",
        count=4,
    )
    result = pool.submit(program, timeout=10)
    assert result.execution.status == "COMPLETED", result.execution.error
    assert result.gpu_ids == (0, 2, 4, 6)
    values = result.execution.results["result"]["value"]
    assert values == [
        {"type": "string", "value": "0,2,4,6"},
        {"type": "null"},
        {"type": "null"},
    ]
    assert result.execution.stdout.count("one invocation") == 1
    assert result.lease_held_ms > 0


@pytest.mark.parametrize("name", ["multi_gpu_single_process", "multi_gpu_multiprocess"])
def test_examples_defer_execution_until_run(pool, name):
    example = Path(__file__).parents[1] / "examples" / name / "main.py"
    program = runpy.run_path(str(example))["build_program"]()
    guard = Program()
    guard.upload(
        id="execution_guard",
        kind="module",
        source="""
import subprocess
import sys
sys.modules["torch"] = None

def reject_subprocess(*args, **kwargs):
    raise RuntimeError("unexpected script subprocess")
subprocess.Popen = reject_subprocess
""",
    )
    instructions = guard.instructions + program.instructions
    run_index = next(i for i, inst in enumerate(instructions) if inst["op"] == "run")
    for execute_kernel in (False, True):
        parsed = parse_program(
            {
                "instructions": instructions if execute_kernel else instructions[:run_index],
                "options": {"gpu_count": 2},
            }
        )
        parsed.blob_bytes = program._blobs.copy()
        outcome = pool.submit(parsed, timeout=10).execution
        if execute_kernel:
            assert outcome.status == "FAILED"
            assert outcome.error["instruction_op"] == "run"
            assert "import of torch halted" in outcome.error["message"]
        else:
            assert outcome.status == "COMPLETED", outcome.error
            assert outcome.results == {}


@pytest.mark.parametrize("count", [None, 2])
@pytest.mark.parametrize("ending", ["return 1", "os._exit(17)", "time.sleep(30)"])
def test_cleanup_reaps_descendants_after_return_crash_or_timeout(pool, tmp_path, ending, count):
    marker = tmp_path / "child.pid"
    source = f"""
def main(path):
    import os, subprocess, sys, time
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"],
        start_new_session=True,
    )
    with open(path, "w") as f:
        f.write(str(child.pid))
    time.sleep(0.1)
    {ending}
"""
    program = build(source, [str(marker)], count=count)
    if ending.startswith("os._exit"):
        with pytest.raises(WorkerCrashed) as error:
            pool.submit(program, timeout=5)
        assert error.value.exitcode == 17
    elif ending.startswith("time.sleep"):
        with pytest.raises(WorkerTimeout):
            pool.submit(program, timeout=2)
    else:
        result = pool.submit(program, timeout=5)
        assert result.execution.status == "FAILED"
        assert "background processes" in result.execution.error["message"]
    assert marker.exists()
    assert not Path(f"/proc/{marker.read_text()}").exists()
    recovered = pool.submit(build("def main(): return 42"), timeout=5)
    assert recovered.execution.status == "COMPLETED"


def test_normal_descendant_teardown_finishes_before_releasing_devices(pool, tmp_path, monkeypatch):
    monkeypatch.setattr(pool, "_termination_grace_seconds", 1)
    marker = tmp_path / "clean-exit"
    child = f"import time; from pathlib import Path; time.sleep(0.5); Path({str(marker)!r}).touch()"
    launcher = f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {child!r}])"
    program = build(f"""
def main():
    import subprocess, sys
    subprocess.run([sys.executable, '-c', {launcher!r}], check=True)
    return 42
""")
    result = pool.submit(program, timeout=10)
    assert result.execution.status == "COMPLETED", result.execution.error
    assert marker.exists(), "the adopted descendant must exit naturally before completion"


@pytest.mark.parametrize("sandbox", ["none", "bubblewrap"])
@pytest.mark.parametrize("ending", ["normal", "crash", "timeout"])
@pytest.mark.parametrize("gpus", [None, (0, 2)])
def test_interpreter_finalizers_use_the_execution_deadline(
    tmp_path, sandbox, ending, gpus, monkeypatch
):
    if sandbox == "bubblewrap":
        import shutil

        if not shutil.which("bwrap"):
            pytest.skip("bubblewrap is not installed")
    monkeypatch.setattr("kcoral.runtime.worker.nvml.device_uuid", lambda _: None)
    worker = Worker(
        gpus,
        fake_runtime_factory,
        sandbox=sandbox,
        sandbox_readonly_paths=(Path(__file__).parent,),
        termination_grace_seconds=0.1,
    )
    leases = NoopLeases() if gpus is None else GPULeases(list(gpus))
    marker = tmp_path / "interpreter-exited"
    observed = []
    if worker._sandbox is not None:
        from kcoral.support import sandbox as sandboxing

        marker = Path(sandboxing.WORKSPACE) / "interpreter-exited"
        sandbox_instance = worker._sandbox
        original_close = sandbox_instance.close

        def close():
            observed.append((sandbox_instance.workspace / marker.name).exists())
            if gpus is not None:
                assert all(leases._holder[gpu] is worker for gpu in gpus)
            original_close()

        monkeypatch.setattr(sandbox_instance, "close", close)
    finalizer_ending = {
        "normal": f"Path({str(marker)!r}).touch()",
        "crash": "os._exit(17)",
        "timeout": "time.sleep(30)",
    }[ending]
    program = build(f"""
def main():
    import atexit
    def finish():
        import os, time
        from pathlib import Path
        time.sleep(0.5)
        {finalizer_ending}
    atexit.register(finish)
    return 42
""")
    # The 0.1-second termination grace must not kill a healthy interpreter
    # that is still inside its 10-second execution deadline.
    try:
        if ending == "timeout":
            with pytest.raises(WorkerTimeout):
                worker.run(program, timeout=1, leases=leases)
        else:
            result = worker.run(program, timeout=10, leases=leases)
            if ending == "crash":
                assert result.execution.status == "FAILED"
                assert "17" in result.execution.error["message"]
            else:
                assert result.execution.status == "COMPLETED", result.execution.error
                assert observed == [True] if sandbox == "bubblewrap" else marker.exists()
        if gpus is not None:
            assert all(leases._holder[gpu] is None for gpu in gpus)
    finally:
        worker.close()


@pytest.mark.parametrize("max_requests", [0, 2])
def test_reused_worker_reaps_adopted_descendants(pool, tmp_path, monkeypatch, max_requests):
    marker = tmp_path / "adopted.pid"
    child = "import time; time.sleep(60)"
    launcher = (
        "import subprocess, sys; from pathlib import Path; "
        f"p = subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True); "
        f"Path({str(marker)!r}).write_text(str(p.pid))"
    )
    # Exercise both unlimited reuse and retirement after a later request.
    for worker in pool._workers:
        worker._max_requests = max_requests
        worker.replace(pool._leases, "request_limit")
    program = build(
        f"""
def main():
    import subprocess, sys
    subprocess.run([sys.executable, '-c', {launcher!r}], check=True)
    return 42
""",
        count=None,
    )
    original_release = pool._leases.release
    live_at_release = []

    def release(gpus, holder):
        if marker.exists():
            live_at_release.append(Path(f"/proc/{marker.read_text()}").exists())
        original_release(gpus, holder)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(pool._leases, "release", release)
            result = pool.submit(program, timeout=10)
        assert result.execution.status == "FAILED"
        assert "background processes" in result.execution.error["message"]
        assert not Path(f"/proc/{marker.read_text()}").exists()
        assert live_at_release and not any(live_at_release)
    finally:
        pool.shutdown()


def test_group_wait_never_partially_claims_and_does_not_starve():
    leases = GPULeases(list(range(8)))
    first, _ = leases.acquire_count(5, "first")
    ready = threading.Event()
    granted = []

    def acquire():
        allocation, _ = leases.acquire_count(4, "next")
        granted.extend(allocation)
        ready.set()

    thread = threading.Thread(target=acquire, daemon=True)
    thread.start()
    wait_for(lambda: leases.depth(7) == 1)
    assert all(leases._holder[gpu] is None for gpu in (5, 6, 7))
    assert not ready.is_set()
    leases.release_many(first, "first")
    assert ready.wait(2)
    assert len(granted) == 4
    leases.release_many(tuple(granted), "next")
    thread.join()


def test_group_lease_blocks_pinned_single_gpu_worker():
    leases = GPULeases([0, 1])
    group, _ = leases.acquire_count(2, "job")
    granted = threading.Event()
    thread = threading.Thread(
        target=lambda: (leases.acquire(1, "single"), granted.set()), daemon=True
    )
    thread.start()
    wait_for(lambda: leases.depth(1) == 2)
    assert not granted.is_set()
    leases.release_many(group, "job")
    assert granted.wait(2)
    leases.release(1, "single")
    thread.join()


def test_shutdown_cancels_pending_gpu_set_without_leaking_admission(pool):
    held, _ = pool._leases.acquire_count(4, "external")
    with ThreadPoolExecutor(1) as executor:
        pending = executor.submit(pool.submit, build("def main(): return 1"), 5)
        wait_for(lambda: pool.active_requests == 1)
        pool.begin_shutdown()
        with pytest.raises(PoolBusy):
            pending.result(timeout=5)
    pool._leases.release_many(held, "external")
    assert pool.active_requests == 0


@pytest.mark.parametrize("count", [0, -1, 9, 2.0, True, "2"])
def test_invalid_gpu_count_is_rejected(count):
    with pytest.raises(ValueError, match="gpu_count"):
        build("def main(): return 1", count=count)


def test_server_rejects_impossible_gpu_count_and_cpu_jobs():
    readonly = (Path(__file__).parent,)
    for config in (
        ServerConfig(gpus=[0], workers_per_gpu=1, sandbox_readonly_paths=readonly),
        ServerConfig(device="cpu", sandbox_readonly_paths=readonly),
    ):
        with TestClient(create_app(config, runtime_factory=fake_runtime_factory)) as client:
            body = {
                "instructions": [{"op": "upload", "id": "m", "kind": "module", "source": "pass"}],
                "options": {"gpu_count": 2},
            }
            response = client.post(
                "/execute", files={"program": (None, json.dumps(body), "application/json")}
            )
            assert response.status_code == 400
            assert "capacity" in response.json()["error"]["message"]


def test_cpu_only_releases_and_reacquires_the_same_gpu_set(pool, tmp_path):
    from kcoral.schemas import GetFunction, Ref, Run

    marker, resume = tmp_path / "cpu-phase", tmp_path / "resume"
    program = build(
        """
def main():
    return 1

def cpu_phase(marker, resume):
    import time
    from pathlib import Path
    Path(marker).touch()
    while not Path(resume).exists():
        time.sleep(0.01)
""",
        count=4,
    )
    program.instructions.extend(
        [
            GetFunction("cpu", Ref("module"), "cpu_phase", cpu_only=True),
            Run("paused", Ref("cpu"), [str(marker), str(resume)]),
            Run("resumed", Ref("fn"), []),
        ]
    )
    with ThreadPoolExecutor(1) as executor:
        pending = executor.submit(pool.submit, program, 10)
        try:
            wait_for(marker.exists)
            assert all(pool._leases.depth(gpu) == 0 for gpu in pool._gpus)
            held, _ = pool._leases.acquire_count(4, "other")
            try:
                resume.touch()
                wait_for(lambda: pool._leases.depth(held[0]) == 2)
                assert not pending.done()
            finally:
                pool._leases.release_many(held, "other")
        finally:
            resume.touch()
        assert pending.result().execution.status == "COMPLETED"
    assert all(pool._leases.depth(gpu) == 0 for gpu in pool._gpus)


def test_unverified_cleanup_quarantines_the_allocation(pool, monkeypatch):
    from kcoral.runtime.lease import GPUUnavailable
    from kcoral.runtime.worker import WorkerCleanupError

    def lost_supervisor(*args, **kwargs):
        raise WorkerCleanupError("supervisor disappeared")

    monkeypatch.setattr("kcoral.runtime.worker.Worker.run", lost_supervisor)
    with pytest.raises(WorkerCleanupError):
        pool.submit(build("def main(): return 1"), 5)
    with pytest.raises(GPUUnavailable):
        pool._leases.acquire_count(4, "next")
    assert pool.closing
    with pytest.raises(PoolBusy):
        pool.submit(build("def main(): return 1"), 5)


def test_quarantine_during_cpu_phase_reaps_transient_worker(pool, tmp_path, monkeypatch):
    from kcoral.schemas import GetFunction, Ref, Run

    marker, resume = tmp_path / "cpu-phase", tmp_path / "resume"
    workers = []
    original_run = Worker.run

    def run(worker, *args):
        workers.append(worker)
        return original_run(worker, *args)

    monkeypatch.setattr(Worker, "run", run)
    program = build("""
def main():
    return 1

def cpu_phase(marker, resume):
    import time
    from pathlib import Path
    Path(marker).touch()
    while not Path(resume).exists():
        time.sleep(0.01)
""")
    program.instructions.extend(
        [
            GetFunction("cpu", Ref("module"), "cpu_phase", cpu_only=True),
            Run("paused", Ref("cpu"), [str(marker), str(resume)]),
            Run("resumed", Ref("fn"), []),
        ]
    )
    try:
        with ThreadPoolExecutor(1) as executor:
            pending = executor.submit(pool.submit, program, 10)
            try:
                wait_for(marker.exists)
                pool._leases.quarantine((workers[0].gpu_ids[0],))
            finally:
                resume.touch()
            with pytest.raises(PoolBusy):
                pending.result(timeout=10)
        assert not workers[0]._proc.is_alive()
        assert pool.active_requests == 0
        for gpu in workers[0].gpu_ids:
            with pytest.raises(GPUUnavailable):
                pool._leases.acquire(gpu, "next")
    finally:
        for worker in workers:
            worker.close()


@pytest.mark.parametrize("phase", ["initialization", "execution"])
def test_unconfirmed_transient_cleanup_is_retried_at_shutdown(pool, monkeypatch, phase):
    workers = []
    original_kill = Worker._kill
    original_initialize = Worker._initialize_process

    def initialize(worker):
        original_initialize(worker)
        if phase == "initialization" and len(worker.gpu_ids) > 1:
            raise WorkerCrashed("initialization failed")

    def kill(worker):
        if len(worker.gpu_ids) > 1:
            workers.append(worker)
            raise WorkerCleanupError("cleanup acknowledgement unavailable")
        original_kill(worker)

    with monkeypatch.context() as patch:
        patch.setattr(Worker, "_kill", kill)
        patch.setattr(Worker, "_initialize_process", initialize)
        with pytest.raises(WorkerCleanupError):
            pool.submit(build("def main(): return 1"), 5)
    try:
        assert workers
        assert workers[0] in pool._transient_workers
        pool.shutdown()
        assert not workers[0]._proc.is_alive()
        assert not pool._transient_workers
    finally:
        for worker in workers:
            worker.close()


@pytest.mark.parametrize("count", [None, 2])
@pytest.mark.parametrize("ending", ["crash", "supervisor_error", "inherited_pipe"])
def test_crashed_cpu_phase_waits_for_lease_before_draining_descendants(
    pool, tmp_path, monkeypatch, count, ending
):
    from kcoral.schemas import GetFunction, Ref, Run

    marker, resume, terminated = (tmp_path / name for name in ("child.pid", "resume", "terminated"))
    workers = []
    original_run = Worker.run

    def run(worker, *args):
        workers.append(worker)
        return original_run(worker, *args)

    monkeypatch.setattr(Worker, "run", run)
    child = f"""
import os, signal, time
from pathlib import Path
def terminate(*args):
    Path({str(terminated)!r}).touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, terminate)
Path({str(marker)!r}).write_text(str(os.getpid()))
while True:
    time.sleep(0.01)
"""
    launch = (
        f"if os.fork() == 0:\n        exec({child!r}, {{}})"
        if ending == "inherited_pipe"
        else f"subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True)"
    )
    ending_source = "time.sleep(30)" if ending == "supervisor_error" else "os._exit(17)"
    program = build(
        f"""
def main():
    import os, subprocess, sys, time
    from pathlib import Path
    {launch}
    while not Path({str(marker)!r}).exists():
        time.sleep(0.01)
    return 42

def cpu_phase():
    import os, time
    from pathlib import Path
    while not Path({str(resume)!r}).exists():
        time.sleep(0.01)
    {ending_source}
""",
        count=count,
    )
    program.instructions.extend(
        [
            GetFunction("cpu", Ref("module"), "cpu_phase", cpu_only=True),
            Run("crashed", Ref("cpu"), []),
        ]
    )
    with ThreadPoolExecutor(1) as executor:
        pending = executor.submit(pool.submit, program, 10)
        held = ()
        try:
            wait_for(marker.exists)
            held, _ = pool._leases.acquire_count(4, "peer")
            if ending == "supervisor_error":
                workers[0]._control.send("invalid command")
            resume.touch()
            wait_for(lambda: any(pool._leases.depth(gpu) == 2 for gpu in held))
            # The peer owns every device while the crashed worker waits. Give
            # the old supervisor enough time to expose an unauthorized drain.
            time.sleep(0.3)
            assert not terminated.exists(), "descendant teardown overlapped the peer's lease"
            assert Path(f"/proc/{marker.read_text()}").exists()
        finally:
            resume.touch()
            pool._leases.release_many(held, "peer")
        with pytest.raises(WorkerCrashed) as error:
            pending.result(timeout=10)
        assert error.value.exitcode == (-15 if ending == "supervisor_error" else 17)
    assert terminated.exists()
    assert not Path(f"/proc/{marker.read_text()}").exists()


@dataclass
class DelayedJobFactory:
    phase: str
    marker: str
    crash: bool = False

    def pause(self, phase):
        if phase == self.phase and "," in os.environ.get("CUDA_VISIBLE_DEVICES", ""):
            Path(self.marker).write_text(str(os.getpid()))
            if self.crash:
                os._exit(17)
            time.sleep(30)

    def prepare(self):
        self.pause("prepare")
        return self

    def __call__(self):
        self.pause("initialize")
        return fake_runtime_factory()


@pytest.mark.parametrize("phase", ["prepare", "initialize"])
@pytest.mark.parametrize("crash", [False, True])
def test_job_startup_timeout_and_crash_have_distinct_http_statuses(
    tmp_path, monkeypatch, phase, crash
):
    monkeypatch.setattr("kcoral.runtime.worker.nvml.device_uuid", lambda _: None)
    marker = tmp_path / "startup.pid"
    config = ServerConfig(gpus=[0, 2], workers_per_gpu=1, sandbox="none")
    app = create_app(config, runtime_factory=DelayedJobFactory(phase, str(marker), crash))
    with TestClient(app) as client:
        body = {
            "instructions": [{"op": "upload", "id": "m", "kind": "module", "source": "pass"}],
            "options": {"gpu_count": 2, "timeout_seconds": 1},
        }
        response = client.post(
            "/execute", files={"program": (None, json.dumps(body), "application/json")}
        )
        assert marker.exists()
        assert response.status_code == (500 if crash else 504), response.text
        assert response.json()["error"]["kind"] == ("engine" if crash else "timeout")
        assert not Path(f"/proc/{marker.read_text()}").exists()
        assert not app.state.pool._transient_workers
        assert all(app.state.pool._leases.depth(gpu) == 0 for gpu in (0, 2))
