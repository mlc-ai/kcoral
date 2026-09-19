"""Real bubblewrap filesystem and worker-lifecycle integration tests."""

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kcoral import Program
from kcoral.client import _decode_value
from kcoral.cpu_runtime import cpu_runtime_factory
from kcoral.events import EventLogger
from kcoral.lease import NoopLeases
from kcoral.pool import WorkerPool
from kcoral.sandbox import Sandbox
from kcoral.schemas import parse_program
from kcoral.worker import Worker, WorkerCrashed, WorkerTimeout


@pytest.fixture
def require_bubblewrap():
    if not shutil.which("bwrap"):
        if os.environ.get("KCORAL_REQUIRE_SANDBOX_TESTS") == "1":
            pytest.fail("KCORAL_REQUIRE_SANDBOX_TESTS requires bubblewrap")
        pytest.skip("bubblewrap is not installed")


def parsed(source, args=(), files=None, *, cpu_only=True):
    program = Program()
    for path, data in (files or {}).items():
        program.upload_file(blob=data, path=path)
    module = program.upload(id="module", kind="module", source=source)
    fn = program.get_function(id="fn", module=module, name="main", cpu_only=cpu_only)
    value = program.run(id="value", fn=fn, args=list(args))
    program.return_(key="value", value=value)
    result = parse_program({"instructions": program.instructions})
    result.blob_bytes = program._blobs
    return result


def values(outcome):
    assert outcome.status == "COMPLETED", outcome.error
    return _decode_value({"type": "object", "value": outcome.results}, outcome.binary_parts, set())


def run(worker, program, timeout=15):
    result, _, _, reason = worker.run(program, timeout, NoopLeases())
    return values(result)["value"], reason


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


@pytest.fixture
def worker(require_bubblewrap):
    instance = Worker(
        None,
        cpu_runtime_factory,
        max_requests=0,
        termination_grace_seconds=0.1,
    )
    try:
        yield instance
    finally:
        instance.close()


def test_missing_bubblewrap_fails_closed(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="install bubblewrap"):
        Worker(None, cpu_runtime_factory)


def test_failed_sandbox_launch_does_not_fall_back(monkeypatch):
    executable = shutil.which("false")
    if executable is None:
        pytest.skip("requires the false utility")
    monkeypatch.setattr(shutil, "which", lambda name: executable)
    with pytest.raises(WorkerCrashed, match="failed during prepare"):
        Worker(None, cpu_runtime_factory, sandbox="bubblewrap", termination_grace_seconds=0.1)


def test_partial_pool_startup_closes_already_created_sandboxes(require_bubblewrap, monkeypatch):
    import kcoral.pool

    created = []
    roots = []

    def factory(*args, **kwargs):
        if created:
            raise RuntimeError("second worker failed")
        instance = Worker(*args, **kwargs)
        created.append(instance)
        roots.append(instance._sandbox.root)
        return instance

    monkeypatch.setattr(kcoral.pool, "Worker", factory)
    with pytest.raises(RuntimeError, match="second worker failed"):
        WorkerPool([], cpu_runtime_factory, cpu_workers=2, sandbox="bubblewrap")
    assert not created[0]._proc.is_alive() and not roots[0].exists()


def test_runtime_mount_cannot_expose_all_workspaces(require_bubblewrap):
    sandbox = Sandbox((Path("/"),))
    try:
        with pytest.raises(ValueError, match="contains sandbox workspaces"):
            sandbox.command(None)
    finally:
        sandbox.close()


def test_reused_process_has_fresh_files_modules_and_temporary_storage(worker):
    original_pid = worker.pid
    source = """
import os
import tempfile
from pathlib import Path
import helper

def main():
    old = Path("previous.txt").exists()
    old_temp = (Path(tempfile.gettempdir()) / "previous.txt").exists()
    Path("previous.txt").write_text("private")
    (Path(tempfile.gettempdir()) / "previous.txt").write_text("private")
    return {
        "old": old, "old_temp": old_temp, "module": helper.VALUE,
        "input": Path("input.txt").read_text(), "cwd": os.getcwd(),
        "tmp": tempfile.gettempdir(), "home": str(Path.home()),
        "namespace": os.readlink("/proc/self/ns/pid"),
    }
"""
    first, first_reason = run(
        worker,
        parsed(
            source,
            files={
                "input.txt": b"a",
                "helper.py": b"VALUE = 'a'\ndef __getattr__(name): raise AssertionError(name)\n",
            },
        ),
    )
    second, second_reason = run(
        worker,
        parsed(
            source,
            files={"input.txt": b"b", "helper.py": b"VALUE = 'b'\n"},
        ),
    )
    assert worker.pid == original_pid and worker.generation == 1
    assert first_reason == second_reason == "completed"
    assert first["namespace"] == second["namespace"]
    assert not first["old"] and not second["old"]
    assert not first["old_temp"] and not second["old_temp"]
    assert (first["input"], second["input"]) == ("a", "b")
    assert (first["module"], second["module"]) == ("a", "b")
    assert second["cwd"] == "/work"
    assert second["tmp"].startswith("/work/") and second["home"].startswith("/work/")
    assert not (worker._sandbox.workspace / "input.txt").exists()


def test_concurrent_workers_cannot_see_each_other_or_modify_outside_files(
    require_bubblewrap,
    tmp_path,
):
    readonly = tmp_path / "readonly.txt"
    readonly.write_text("unchanged")
    original_mode = readonly.stat().st_mode
    hidden = tmp_path / "hidden.txt"
    hidden.write_text("private-host-data")
    a = Worker(
        None,
        cpu_runtime_factory,
        sandbox="bubblewrap",
        max_requests=0,
        sandbox_readonly_paths=(readonly,),
        termination_grace_seconds=0.1,
    )
    b = Worker(
        None,
        cpu_runtime_factory,
        sandbox="bubblewrap",
        max_requests=0,
        termination_grace_seconds=0.1,
    )
    workspace_b = b._sandbox.workspace
    try:
        with ThreadPoolExecutor(1) as executor:
            pending = executor.submit(
                run,
                b,
                parsed("""
import time
from pathlib import Path
def main():
    Path("secret.txt").write_text("b-private")
    Path("ready").touch()
    while not Path("release").exists():
        time.sleep(0.01)
    return Path("secret.txt").read_text()
"""),
            )
            try:
                wait_for(lambda: (workspace_b / "ready").exists())
                result, reason = run(
                    a,
                    parsed(
                        """
import os
import subprocess
import sys
from pathlib import Path
def blocked(fn):
    try:
        fn()
    except OSError:
        return True
    return False
def main(other, readonly, hidden):
    Path("link").symlink_to(other + "/secret.txt")
    Path("own.txt").write_text("a-private")
    child = subprocess.run(
        [sys.executable, "-c",
         "from pathlib import Path; Path(" + repr(readonly) + ").write_text('bad')"],
        capture_output=True,
    )
    return {
        "other_read": blocked(lambda: Path(other, "secret.txt").read_text()),
        "other_list": blocked(lambda: os.listdir(other)),
        "hidden_read": blocked(lambda: Path(hidden).read_text()),
        "link_read": blocked(lambda: Path("link").read_text()),
        "outside_write": blocked(lambda: Path(readonly).write_text("bad")),
        "outside_chmod": blocked(lambda: os.chmod(readonly, 0o777)),
        "root_write": blocked(lambda: Path("/outside.txt").write_text("bad")),
        "dev_write": blocked(lambda: Path("/dev/outside.txt").write_text("bad")),
        "child_denied": child.returncode != 0,
        "readonly": Path(readonly).read_text(),
        "own": Path("own.txt").read_text(),
    }
""",
                        (str(workspace_b), str(readonly), str(hidden)),
                    ),
                )
                assert reason == "completed"
                for key, value in result.items():
                    if key not in ("readonly", "own"):
                        assert value, key
                assert result["readonly"] == "unchanged" and result["own"] == "a-private"
                assert a.pid != b.pid
            finally:
                (workspace_b / "release").touch()
            assert pending.result()[0] == "b-private"
        assert readonly.read_text() == "unchanged" and readonly.stat().st_mode == original_mode
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("max_requests", [0, 1, 2])
def test_pool_supports_both_new_and_reused_sandbox_processes(require_bubblewrap, max_requests):
    pool = WorkerPool(
        [],
        cpu_runtime_factory,
        cpu_workers=1,
        max_requests_per_worker=max_requests,
        termination_grace_seconds=0.1,
    )
    try:
        pids = []
        for value in ("a", "b", "c"):
            outcome = pool.submit(
                parsed(
                    """
from pathlib import Path
def main(value):
    assert not Path("old.txt").exists()
    Path("old.txt").write_text(value)
    return value
""",
                    (value,),
                ),
                15,
                worker_wait_timeout=15,
            )
            pids.append(pool._workers[0].pid)
            assert values(outcome.execution)["value"] == value
        if max_requests == 0:
            assert len(set(pids)) == 1
        elif max_requests == 1:
            assert len(set(pids)) == 3
        else:
            assert pids[0] == pids[1] and pids[2] != pids[1]
    finally:
        pool.shutdown()


def test_retained_file_descriptor_retires_worker_instead_of_exposing_next_request(worker):
    root = worker._sandbox.root
    value, reason = run(
        worker,
        parsed("""
import builtins
def main():
    builtins.retained_file = open("retained.txt", "w")
    return "completed"
"""),
    )
    assert value == "completed" and reason == "sandbox_cleanup"
    assert not worker._proc.is_alive()
    assert not root.exists()


def test_background_thread_retires_worker(worker):
    _, reason = run(
        worker,
        parsed("""
import threading
import time
def main():
    threading.Thread(target=lambda: time.sleep(60), daemon=True).start()
    return None
"""),
    )
    assert reason == "sandbox_cleanup"
    assert not worker._proc.is_alive()


def test_reserved_runtime_upload_path_is_rejected(worker):
    outcome, _, _, _ = worker.run(
        parsed("def main(): return None", files={".kcoral/input": b"bad"}), 15, NoopLeases()
    )
    assert outcome.status == "FAILED" and "reserved" in outcome.error["message"]


def test_timeout_recovers_and_preserves_output(worker):
    root = worker._sandbox.root
    old_pid = worker.pid
    with pytest.raises(WorkerTimeout) as caught:
        worker.run(
            parsed("""
import os
import subprocess
import sys
import time
def main():
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                     start_new_session=True)
    os.write(2, b"sandbox-timeout-output")
    time.sleep(60)
"""),
            0.5,
            NoopLeases(),
        )
    assert "sandbox-timeout-output" in caught.value.output_tail
    assert not root.exists()
    assert worker.pid != old_pid
    assert run(worker, parsed("def main(): return 'recovered'"))[0] == "recovered"


def test_detached_child_is_terminated_before_workspace_removal(worker):
    workspace = worker._sandbox.workspace
    root = worker._sandbox.root
    with ThreadPoolExecutor(1) as executor:
        pending = executor.submit(
            run,
            worker,
            parsed("""
import os
import subprocess
import sys
import time
from pathlib import Path
def main():
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                     start_new_session=True)
    Path("namespace").write_text(os.readlink("/proc/self/ns/pid"))
    time.sleep(60)
"""),
            2,
        )
        wait_for(lambda: (workspace / "namespace").exists())
        namespace = (workspace / "namespace").read_text()
        processes = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if os.readlink(entry / "ns/pid") == namespace:
                    processes.append(entry)
            except OSError:
                pass
        assert len(processes) >= 3  # reaper, interpreter, detached child
        with pytest.raises(WorkerTimeout):
            pending.result()
    for entry in processes:
        try:
            assert os.readlink(entry / "ns/pid") != namespace
        except FileNotFoundError:
            pass  # exited or zombie: neither can access the next workspace
    assert not root.exists()


def test_subprocess_compilation_and_file_returns(worker):
    if not Path("/usr/bin/cc").exists():
        pytest.skip("a system C compiler is not installed")
    program = parsed("""
import subprocess
from pathlib import Path
def main():
    subprocess.run(
        ["/usr/bin/cc", "-x", "c", "-o", "compiled", "-"],
        input=b"int main(void) { return 0; }", check=True,
    )
    subprocess.run(["./compiled"], check=True)
    Path("result.txt").write_text("compiled")
    return "result.txt"
""")
    from kcoral.schemas import FileReturn

    program.instructions.append(FileReturn(key="file", kind="file", path="result.txt"))
    outcome, _, _, reason = worker.run(program, 30, NoopLeases())
    assert reason == "completed"
    assert values(outcome)["file"].read_bytes() == b"compiled"


def test_workspace_is_removed_on_shutdown(worker):
    root = worker._sandbox.root
    worker.close()
    assert not root.exists() and not worker._proc.is_alive()


def test_http_uploads_and_logging_use_isolated_reused_worker(require_bubblewrap, tmp_path):
    from fastapi.testclient import TestClient

    from kcoral import ServerConfig
    from kcoral.app import create_app

    cache = tmp_path / "cache"
    config = ServerConfig(
        device="cpu",
        num_workers=1,
        max_requests_per_worker=0,
        disk_cache_dir=cache,
        log_dir=tmp_path / "logs",
        log_console=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        original_pid = app.state.pool._workers[0].pid
        for data in (b"first", b"second"):
            program = parsed(
                """
from pathlib import Path
def main(cache):
    assert not Path(cache).exists()
    assert not Path("previous").exists()
    Path("previous").touch()
    print("isolated-output")
    return Path("input.txt").read_text()
""",
                (str(cache),),
                {"input.txt": data},
            )
            # Build the wire instructions through the public client helper.
            wire = Program()
            wire.upload_file(blob=data, path="input.txt")
            module = wire.upload(id="module", kind="module", source=program.instructions[1].source)
            fn = wire.get_function(id="fn", module=module, name="main", cpu_only=True)
            result = wire.run(id="result", fn=fn, args=[str(cache)])
            wire.return_(key="value", value=result)
            files = [
                (
                    "program",
                    (None, json.dumps({"instructions": wire.instructions}), "application/json"),
                )
            ]
            files.extend(
                (f"blob:{key}", (None, value, "application/octet-stream"))
                for key, value in wire._blobs.items()
            )
            response = client.post("/execute", files=files)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["status"] == "COMPLETED", body.get("error")
            assert body["results"]["value"]["value"] == data.decode()
            assert "isolated-output" in body["stdout"]
            assert app.state.pool._workers[0].pid == original_pid
    log = next((tmp_path / "logs").glob("runs/*/events.jsonl")).read_text()
    assert '"sandbox":"bubblewrap"' in log
    assert "request_finished" in log


@pytest.fixture
def gpu_sandbox(require_bubblewrap, tmp_path):
    if os.environ.get("KCORAL_GPU_TEST") != "1":
        pytest.skip("requires KCORAL_GPU_TEST=1 and an externally locked idle GPU")
    from kcoral.gpu_runtime import gpu_runtime_factory

    gpu_id = int(os.environ.get("KCORAL_SANDBOX_GPU", "0"))
    readonly = tuple(
        Path(p) for p in os.environ.get("KCORAL_SANDBOX_READONLY_PATHS", "").split(os.pathsep) if p
    )
    events = EventLogger(tmp_path / "events", console=True)
    instance = Worker(
        gpu_id,
        gpu_runtime_factory,
        sandbox="bubblewrap",
        max_requests=0,
        sandbox_readonly_paths=readonly,
        termination_grace_seconds=0.1,
        events=events,
    )
    from kcoral.lease import GPULeases

    try:
        yield instance, GPULeases([gpu_id]), readonly
    finally:
        instance.close()
        events.close()


def test_gpu_computation_reuses_process_with_fresh_files(gpu_sandbox):
    worker, leases, _ = gpu_sandbox
    original_pid = worker.pid
    for value in (3, 7):
        outcome, _, _, reason = worker.run(
            parsed(
                """
import torch
from pathlib import Path
def main(value):
    assert not Path("previous").exists()
    Path("previous").touch()
    result = torch.arange(16, device="cuda", dtype=torch.float32) + value
    return result.cpu().tolist()
""",
                (value,),
                cpu_only=False,
            ),
            30,
            leases,
        )
        assert values(outcome)["value"] == [float(i + value) for i in range(16)]
        assert reason == "completed"
        assert worker.pid == original_pid and worker._sandbox is not None


@pytest.mark.parametrize("keep_mapping", [False, True])
def test_uploaded_library_state_does_not_cross_reused_gpu_requests(gpu_sandbox, keep_mapping):
    worker, leases, readonly = gpu_sandbox
    compiler = Worker(
        None,
        cpu_runtime_factory,
        sandbox="bubblewrap",
        max_requests=0,
        sandbox_readonly_paths=readonly,
        termination_grace_seconds=0.1,
    )
    try:
        library, _ = run(
            compiler,
            parsed(
                '''
from pathlib import Path
import subprocess
from tvm_ffi.libinfo import find_include_path, find_dlpack_include_path
def main(keep_mapping):
    # Export the C interface directly so no C++ thread-local destructors keep
    # this test library mapped. The second case explicitly requests retention.
    source = b"""
#include <tvm/ffi/c_api.h>
int __tvm_ffi_next_value(void* self, const TVMFFIAny* args,
                         int32_t count, TVMFFIAny* result) {
    static int counter = 0;
    result->type_index = kTVMFFIInt;
    result->v_int64 = ++counter;
    return 0;
}
"""
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-x", "c", "-", "-o", "counter.so",
         "-I" + find_include_path(), "-I" + find_dlpack_include_path()]
        + (["-Wl,-z,nodelete"] if keep_mapping else []),
        input=source, check=True,
    )
    return Path("counter.so").read_bytes()
''',
                (keep_mapping,),
            ),
            timeout=120,
        )
    finally:
        compiler.close()
    original_pid = worker.pid
    for request_index in range(2):
        program = Program()
        module = program.upload(id="module", kind="library", value=library)
        fn = program.get_function(id="fn", module=module, name="next_value")
        result = program.run(id="result", fn=fn)
        program.return_(key="value", value=result)
        request = parse_program({"instructions": program.instructions})
        request.blob_bytes = program._blobs
        outcome, _, _, reason = worker.run(request, 30, leases)
        assert values(outcome)["value"] == 1
        if keep_mapping:
            assert reason == "sandbox_cleanup" and worker._sandbox is None
            assert worker.generation == request_index + 1
        else:
            assert reason == "completed"
            assert worker.pid == original_pid and worker._sandbox is not None
