"""Real process supervision and scheduling, using a GPU-free execution runtime."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kcoral import Client, Program
from kcoral.app import create_app
from kcoral.config import ServerConfig
from kcoral.lease import GPULeases
from kcoral.pool import PoolBusy, WorkerPool
from kcoral.schemas import parse_program
from kcoral.testing import fake_runtime_factory
from kcoral.worker import WorkerCrashed, WorkerTimeout


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def build(source, args=(), *, count=2, files=None):
    p = Program()
    for name, content in (files or {}).items():
        p.upload_file(path=name, blob=content)
    module = p.upload(id="module", kind="module", source=source)
    fn = p.get_function(id="fn", module=module, name="main")
    result = p.run(id="result", fn=fn, args=list(args))
    p.return_(key="result", value=result)
    parsed = parse_program({"instructions": p.instructions, "options": {"gpu_count": count}})
    parsed.blob_bytes = p._blobs.copy()
    return parsed


@pytest.fixture
def pool(monkeypatch):
    monkeypatch.setattr("kcoral.pool.nvml.device_uuid", lambda _: None)
    with_pool = WorkerPool(
        [0, 2, 4, 6],
        fake_runtime_factory,
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


def test_uploaded_script_runs_with_its_original_command_and_returns_files(pool):
    script = (
        b'import json, os\nprint("script output")\n'
        b'open("answer.json", "w").write(json.dumps(os.environ["CUDA_VISIBLE_DEVICES"]))\n'
    )
    program = build(
        """
def main():
    import subprocess, sys
    subprocess.run([sys.executable, "infer.py"], check=True)
    return "answer.json"
""",
        files={"infer.py": script},
    )
    from kcoral.schemas import FileReturn, Ref

    program.instructions.append(FileReturn("artifact", "file", Ref("result")))
    result = pool.submit(program, timeout=10)
    assert result.execution.status == "COMPLETED", result.execution.error
    assert "script output" in result.execution.stdout
    assert b'"0,2"' in result.execution.binary_parts.values()


@pytest.mark.parametrize(
    "ending", ["return 1", "raise ValueError('model failed')", "os._exit(17)", "time.sleep(30)"]
)
def test_cleanup_reaps_detached_descendants_after_all_outcomes(pool, tmp_path, ending):
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
    program = build(source, [str(marker)])
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
        expected = "background processes" if ending.startswith("return") else "model failed"
        assert expected in result.execution.error["message"]
    assert marker.exists()
    assert not Path(f"/proc/{marker.read_text()}").exists()
    recovered = pool.submit(build("def main(): return 42"), timeout=5)
    assert recovered.execution.status == "COMPLETED"


def test_multi_gpu_jobs_never_share_devices(pool, tmp_path):
    source = """
def main(path):
    import os, time
    with open(path, "w") as f:
        f.write(os.environ["CUDA_VISIBLE_DEVICES"])
    time.sleep(0.5)
"""
    with ThreadPoolExecutor(2) as executor:
        a = executor.submit(pool.submit, build(source, [str(tmp_path / "a")]), 5)
        b = executor.submit(pool.submit, build(source, [str(tmp_path / "b")]), 5)
        ra, rb = a.result(), b.result()
    assert not set(ra.gpu_ids) & set(rb.gpu_ids)
    assert ra.execution.status == rb.execution.status == "COMPLETED"


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


def test_interpreter_finalizers_use_the_execution_deadline(pool, tmp_path):
    marker = tmp_path / "interpreter-exited"
    program = build(f"""
def main():
    import atexit
    def finish():
        import time
        from pathlib import Path
        time.sleep(0.5)
        Path({str(marker)!r}).touch()
    atexit.register(finish)
    return 42
""")
    # The pool's 0.1-second termination grace must not kill a healthy interpreter
    # that is still inside its 10-second execution deadline.
    result = pool.submit(program, timeout=10)
    assert result.execution.status == "COMPLETED", result.execution.error
    assert marker.exists()


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
    for config in (ServerConfig(gpus=[0], workers_per_gpu=1), ServerConfig(device="cpu")):
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


def test_cpu_only_and_file_steps_keep_the_complete_reservation(pool, tmp_path):
    from kcoral.schemas import FileReturn, FileUpload, GetFunction, Ref, Run

    marker = tmp_path / "cpu-phase"
    program = build(
        """
def main():
    return 1

def cpu_phase(marker):
    import time
    from pathlib import Path
    Path(marker).write_text("ready")
    time.sleep(0.5)
""",
        count=4,
    )
    from kcoral.keys import compute_blob_hash

    digest = compute_blob_hash(b"data")
    program.blob_bytes[digest] = b"data"
    program.instructions.extend(
        [
            FileUpload(digest, "input.bin"),
            FileReturn("file", "file", "input.bin"),
            GetFunction("cpu", Ref("module"), "cpu_phase", cpu_only=True),
            Run("paused", Ref("cpu"), [str(marker)]),
        ]
    )
    with ThreadPoolExecutor(1) as executor:
        pending = executor.submit(pool.submit, program, 10)
        wait_for(marker.exists)
        assert all(pool._leases._holder[gpu] is not None for gpu in pool._gpus)
        assert pending.result().execution.status == "COMPLETED"
    assert all(pool._leases.depth(gpu) == 0 for gpu in pool._gpus)


def test_shutdown_drains_an_executing_job(pool, tmp_path):
    marker = tmp_path / "started"
    program = build(
        """
def main(path):
    import time
    from pathlib import Path
    Path(path).write_text("ready")
    time.sleep(0.5)
    return 42
""",
        [str(marker)],
    )
    with ThreadPoolExecutor(2) as executor:
        pending = executor.submit(pool.submit, program, 10)
        wait_for(marker.exists)
        stopping = executor.submit(pool.shutdown)
        wait_for(lambda: pool.closing)
        assert not stopping.done()
        assert pending.result().execution.status == "COMPLETED"
        stopping.result(timeout=10)


def test_unverified_cleanup_quarantines_the_allocation(pool, monkeypatch):
    from kcoral.job import JobCleanupError
    from kcoral.lease import GPUUnavailable

    def lost_supervisor(*args, **kwargs):
        raise JobCleanupError("supervisor disappeared")

    monkeypatch.setattr("kcoral.pool.run_job", lost_supervisor)
    with pytest.raises(JobCleanupError):
        pool.submit(build("def main(): return 1"), 5)
    with pytest.raises(GPUUnavailable):
        pool._leases.acquire_count(4, "next")
    assert pool.closing
    with pytest.raises(PoolBusy):
        pool.submit(build("def main(): return 1"), 5)


def test_eight_device_job_protocol_without_gpu_hardware(monkeypatch):
    monkeypatch.setattr("kcoral.pool.nvml.device_uuid", lambda _: None)
    config = ServerConfig(gpus=list(range(8)), workers_per_gpu=1, max_requests_per_worker=0)
    with TestClient(create_app(config, runtime_factory=fake_runtime_factory)) as server:
        client = Client("http://testserver")
        client._http.close()
        client._http = server
        program = Program()
        module = program.upload(
            id="m",
            kind="module",
            source="""
def main():
    import os
    return os.environ["CUDA_VISIBLE_DEVICES"]
""",
        )
        fn = program.get_function(id="fn", module=module, name="main")
        value = program.run(id="value", fn=fn)
        program.return_(key="devices", value=value)
        result = client.execute(program, gpu_count=8)
        assert result.completed, result.error
        assert result.gpu_ids == tuple(range(8))
        assert result["devices"] == "0,1,2,3,4,5,6,7"
