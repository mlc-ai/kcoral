import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from support.programs import harness_instructions, python_instructions

from kcoral.app import create_app
from kcoral.config import ServerConfig
from kcoral.gpu_runtime import gpu_runtime_factory
from kcoral.keys import compute_blob_hash
from kcoral.multipart import parse_multipart
from kcoral.schemas import strict_json_loads
from kcoral.testing import fake_runtime_factory

ADD_ONE = "def main(value):\n    return value + 1\n"


def make_client(config=None):
    if config is None:
        config = ServerConfig(sandbox="none", max_requests_per_worker=0)
    return TestClient(create_app(config, runtime_factory=fake_runtime_factory))


def gpu_app(**overrides):
    """An app on the visible GPU, with one worker: these tests exercise the request
    path rather than GPU sharing, and spawning the default eight costs ~40s."""
    gpu_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    gpu_id = int(gpu_raw) if gpu_raw.isdigit() else 0
    return create_app(
        ServerConfig(
            sandbox="none",
            gpus=[gpu_id],
            workers_per_gpu=overrides.pop("workers_per_gpu", 1),
            max_requests_per_worker=overrides.pop("max_requests_per_worker", 0),
            **overrides,
        ),
        runtime_factory=gpu_runtime_factory,
    )


def post_program(client, program, blobs=None, *, raw_program=None):
    program_data = raw_program if raw_program is not None else json.dumps(program)
    files = [("program", (None, program_data, "application/json"))]
    files.extend(
        (f"blob:{blob_hash}", (None, data, "application/octet-stream"))
        for blob_hash, data in (blobs or {}).items()
    )
    return client.post("/execute", files=files)


def response_parts(response):
    media_type = response.headers["content-type"].split(";", 1)[0]
    if media_type == "application/json":
        return response.json(), {}
    result = None
    binary = {}
    for part in parse_multipart(response.headers["content-type"], response.content):
        if part.name == "result":
            result = strict_json_loads(part.data)
        else:
            binary[part.name] = part.data
    assert result is not None
    return result, binary


def scalar_program():
    return {
        "instructions": [
            {"op": "upload", "id": "fn_module", "kind": "module", "source": ADD_ONE},
            {
                "op": "get_function",
                "id": "fn",
                "module": {"$ref": "fn_module"},
                "name": "main",
            },
            {"op": "run", "id": "answer", "fn": {"$ref": "fn"}, "args": [41]},
            {"op": "return", "key": "answer", "value": {"$ref": "answer"}},
        ]
    }


def test_health():
    with make_client() as client:
        data = client.get("/health").json()
    assert set(data) == {
        "status",
        "instance_id",
        "started_at",
        "gpu_count",
        "load",
        "target",
        "versions",
    }
    assert data["status"] == "ok" and data["gpu_count"] == 1
    assert data["load"] == {
        "request_capacity": ServerConfig().workers_per_gpu,
        "requests_in_progress": 0,
        "requests_waiting": 0,
    }
    assert data["instance_id"]


def test_instance_id_changes_with_server_lifecycle():
    app = create_app(
        ServerConfig(sandbox="none", max_requests_per_worker=0),
        runtime_factory=fake_runtime_factory,
    )
    with TestClient(app) as client:
        first = client.get("/health").json()["instance_id"]
    with TestClient(app) as client:
        second = client.get("/health").json()["instance_id"]
    assert first != second


def test_cpu_health_and_execution_have_no_gpu_lease():
    config = ServerConfig(sandbox="none", device="cpu", num_workers=2, max_requests_per_worker=0)
    with make_client(config) as client:
        health = client.get("/health").json()
        result = post_program(client, scalar_program()).json()

    assert health["status"] == "ok"
    assert health["gpu_count"] == 0
    assert health["load"] == {
        "request_capacity": 2,
        "requests_in_progress": 0,
        "requests_waiting": 0,
    }
    assert result["status"] == "COMPLETED"
    assert result["lease_wait_ms"] == 0
    assert result["lease_held_ms"] == 0


def test_cpu_app_selects_cpu_runtime_by_default():
    config = ServerConfig(sandbox="none", device="cpu", num_workers=1, max_requests_per_worker=0)
    with TestClient(create_app(config)) as client:
        health = client.get("/health").json()

    assert health["gpu_count"] == 0
    assert health["target"] == {}
    assert health["load"]["request_capacity"] == 1


def _cuda_toolchain_available() -> bool:
    """Whether this machine can build CUDA C, which is all the CPU server needs.
    Probed rather than opted into: unlike a GPU test, compiling contends with
    nothing, so it should run wherever it can."""
    try:
        from support.cuda import _require_cuda_toolchain

        _require_cuda_toolchain()
    except Exception:
        return False
    return True


@pytest.mark.skipif(
    not _cuda_toolchain_available(),
    reason="needs the server extra and a CUDA toolchain (nvcc, ninja, c++)",
)
def test_cpu_cuda_compilation_end_to_end():
    arch = os.environ.get("KCORAL_CPU_COMPILE_ARCH", "sm_90a")
    program = {
        "instructions": [
            {
                "op": "upload",
                "id": "source_module",
                "kind": "module",
                "language": "cuda",
                "source": "void add_one(tvm::ffi::TensorView x) {}",
            },
            {
                "op": "get_function",
                "id": "source",
                "module": {"$ref": "source_module"},
                "name": "add_one",
            },
            *harness_instructions(
                "library", "compile_cuda_binary", [{"$ref": "source"}, {"arch": arch}]
            ),
            {"op": "return", "key": "library", "value": {"$ref": "library"}},
        ]
    }
    config = ServerConfig(sandbox="none", device="cpu", num_workers=1, max_requests_per_worker=0)
    with TestClient(create_app(config)) as client:
        response = post_program(client, program)

    result, binary_parts = response_parts(response)
    assert result["status"] == "COMPLETED", result.get("error")
    assert result["lease_wait_ms"] == 0
    assert result["lease_held_ms"] == 0
    assert binary_parts["return:0"].startswith(b"\x7fELF")


def test_request_id_and_timing_are_returned():
    with make_client() as client:
        first = post_program(client, scalar_program())
        second = post_program(client, scalar_program())
    data = first.json()
    assert data["request_id"] == first.headers["x-request-id"]
    assert first.headers["x-request-id"] != second.headers["x-request-id"]
    assert data["queue_ms"] >= 0 and data["elapsed_ms"] >= 0


def test_completed_program_returns_only_selected_values():
    with make_client() as client:
        response = post_program(client, scalar_program())
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "COMPLETED"
    assert data["results"] == {"answer": {"type": "integer", "value": 42}}
    assert data["stdout"] == "" and data["stderr"] == ""


def test_failed_instruction_stops_after_a_return_that_already_ran():
    program = {
        "instructions": [
            *harness_instructions("ok", "structural"),
            {"op": "return", "key": "ok", "value": {"$ref": "ok"}},
            *harness_instructions("bad", "nope"),
            {"op": "return", "key": "never", "value": {"$ref": "ok"}},
        ]
    }
    with make_client() as client:
        response = post_program(client, program)
    data = response.json()
    assert response.status_code == 200 and data["status"] == "FAILED"
    assert set(data["results"]) == {"ok"}
    error = data["error"]
    assert error["kind"] == "runtime" and error["instruction_index"] == 6
    assert error["instruction_op"] == "run" and error["instruction_id"] == "bad"
    assert "Traceback" in error["traceback"]


def test_tensor_cache_miss_upload_and_warm_hit():
    raw = b"\x00\x00\x80?"
    digest = compute_blob_hash(raw)
    program = {
        "instructions": [
            {
                "op": "upload",
                "id": "tensor",
                "kind": "tensor",
                "blob": digest,
                "dtype": "float32",
                "shape": [1],
            },
            {"op": "return", "key": "tensor", "value": {"$ref": "tensor"}},
        ]
    }
    with make_client() as client:
        miss = post_program(client, program)
        uploaded = post_program(client, program, {digest: raw})
        warm = post_program(client, program)
    assert miss.json()["missing_blobs"] == [digest]
    for response in (uploaded, warm):
        result, binary = response_parts(response)
        assert result["status"] == "COMPLETED"
        assert result["results"]["tensor"]["part"] == "return:0"
        assert binary == {"return:0": raw}


def test_file_cache_persists_across_server_restart_without_memory_cache(tmp_path):
    data = b"file contents"
    key = compute_blob_hash(data)
    program = {
        "instructions": [{"op": "upload", "kind": "file", "blob": key, "path": "data/input"}]
    }
    config = ServerConfig(
        sandbox="none", workers_per_gpu=1, max_requests_per_worker=0, disk_cache_dir=tmp_path
    )
    with make_client(config) as client:
        assert post_program(client, program).json()["status"] == "CACHE_MISS"
        assert post_program(client, program, {key: data}).json()["status"] == "COMPLETED"
        assert client.app.state.cache.get(key) is None
        assert client.app.state.file_cache.get(key) == data
    with make_client(config) as restarted:
        assert post_program(restarted, program).json()["status"] == "COMPLETED"
        assert restarted.app.state.cache.get(key) is None


@pytest.mark.parametrize("disabled", ["directory", "capacity", "unusable"])
def test_uncached_file_upload_still_executes(tmp_path, disabled):
    directory = tmp_path / "cache"
    if disabled == "unusable":
        directory.write_bytes(b"occupied")
    config = ServerConfig(
        sandbox="none",
        workers_per_gpu=1,
        max_requests_per_worker=0,
        disk_cache_dir=None if disabled == "directory" else directory,
        disk_cache_capacity_mbytes=0 if disabled == "capacity" else 100,
    )
    data = b"uncached file"
    key = compute_blob_hash(data)
    program = {"instructions": [{"op": "upload", "kind": "file", "blob": key, "path": "input"}]}
    with make_client(config) as client:
        assert post_program(client, program, {key: data}).json()["status"] == "COMPLETED"
        assert client.app.state.cache.get(key) is None
        assert post_program(client, program).json()["status"] == "CACHE_MISS"


@pytest.mark.parametrize("size", [1024**2, 1024**2 + 1])
def test_disk_cache_capacity_mbytes_uses_binary_megabytes(tmp_path, size):
    config = ServerConfig(
        sandbox="none",
        workers_per_gpu=1,
        max_requests_per_worker=0,
        disk_cache_dir=tmp_path,
        disk_cache_capacity_mbytes=1,
    )
    data = b"x" * size
    key = compute_blob_hash(data)
    program = {"instructions": [{"op": "upload", "kind": "file", "blob": key, "path": "input"}]}
    with make_client(config) as client:
        assert post_program(client, program, {key: data}).json()["status"] == "COMPLETED"
        warm = post_program(client, program).json()
        assert warm["status"] == ("COMPLETED" if size == 1024**2 else "CACHE_MISS")


@pytest.mark.parametrize(
    "kinds", [("file",), ("bytes",), ("tensor",), ("library",), ("file", "bytes")]
)
def test_blobs_are_cached_only_in_backends_requested_by_their_upload_kinds(tmp_path, kinds):
    data = b"1234"
    key = compute_blob_hash(data)
    instructions = []
    for kind in kinds:
        item = {"op": "upload", "kind": kind, "blob": key}
        if kind == "file":
            item["path"] = "input"
        else:
            item["id"] = kind
        if kind == "tensor":
            item.update(dtype="float32", shape=[1])
        instructions.append(item)
    config = ServerConfig(
        sandbox="none", workers_per_gpu=1, max_requests_per_worker=0, disk_cache_dir=tmp_path
    )
    with make_client(config) as client:
        # The fake runtime cannot load libraries; cache routing still happens before execution.
        response = post_program(client, {"instructions": instructions}, {key: data})
        assert response.status_code == 200
        assert client.app.state.cache.get(key) == (
            data if any(kind != "file" for kind in kinds) else None
        )
        assert client.app.state.file_cache.get(key) == (data if "file" in kinds else None)


@pytest.mark.parametrize(
    "files,match",
    [
        (
            [
                ("program", (None, json.dumps(scalar_program()), "application/json")),
                ("program", (None, json.dumps(scalar_program()), "application/json")),
            ],
            "duplicate",
        ),
        (
            [("blob:bad", (None, b"data", "application/octet-stream"))],
            "malformed",
        ),
        (
            [("unknown", (None, b"data", "application/octet-stream"))],
            "unsupported",
        ),
    ],
)
def test_invalid_multipart_parts_are_rejected(files, match):
    if not any(name == "program" for name, _ in files):
        files = [("program", (None, json.dumps(scalar_program()), "application/json")), *files]
    with make_client() as client:
        response = client.post("/execute", files=files)
    assert response.status_code == 400 and match in response.json()["error"]["message"]


def test_duplicate_blob_hash_mismatch_unreferenced_and_wrong_length_rejected():
    raw = b"data"
    digest = compute_blob_hash(raw)
    tensor_program = {
        "instructions": [
            {
                "op": "upload",
                "id": "x",
                "kind": "tensor",
                "blob": digest,
                "dtype": "float32",
                "shape": [1],
            }
        ]
    }
    duplicate_files = [
        ("program", (None, json.dumps(tensor_program), "application/json")),
        (f"blob:{digest}", (None, raw, "application/octet-stream")),
        (f"blob:{digest}", (None, raw, "application/octet-stream")),
    ]
    with make_client() as client:
        duplicate = client.post("/execute", files=duplicate_files)
        mismatch = post_program(client, tensor_program, {digest: b"bad"})
        unreferenced_hash = compute_blob_hash(b"other")
        unreferenced = post_program(client, scalar_program(), {unreferenced_hash: b"other"})
        wrong_length_hash = compute_blob_hash(b"abc")
        wrong_length_program = {
            "instructions": [
                {
                    "op": "upload",
                    "id": "x",
                    "kind": "tensor",
                    "blob": wrong_length_hash,
                    "dtype": "float32",
                    "shape": [1],
                }
            ]
        }
        wrong_length = post_program(client, wrong_length_program, {wrong_length_hash: b"abc"})
    assert duplicate.status_code == 400 and "duplicate" in duplicate.json()["error"]["message"]
    assert mismatch.status_code == 400 and "mismatch" in mismatch.json()["error"]["message"]
    assert (
        unreferenced.status_code == 400
        and "unreferenced" in unreferenced.json()["error"]["message"]
    )
    assert (
        wrong_length.status_code == 400
        and "expects 4 bytes" in wrong_length.json()["error"]["message"]
    )


def test_strict_json_and_unknown_instruction_are_400():
    with make_client() as client:
        duplicate = post_program(
            client,
            {},
            raw_program='{"instructions": [], "instructions": []}',
        )
        non_finite = post_program(
            client,
            {},
            raw_program=(
                '{"instructions":[{"op":"run","id":"x","fn":{"$ref":"fn"},"args":[NaN]}]}'
            ),
        )
        unknown = post_program(client, {"instructions": [{"op": "frob"}]})
    assert duplicate.status_code == 400 and "duplicate" in duplicate.json()["error"]["message"]
    assert non_finite.status_code == 400 and "non-finite" in non_finite.json()["error"]["message"]
    assert unknown.status_code == 400 and "unknown op" in unknown.json()["error"]["message"]


def test_request_and_response_size_limits():
    with TestClient(
        create_app(
            ServerConfig(sandbox="none", gpus=[0], max_request_bytes=100),
            runtime_factory=fake_runtime_factory,
        )
    ) as client:
        too_large = post_program(client, scalar_program())
    assert too_large.status_code == 413

    source = "def main():\n    return b'x' * 1000\n"
    program = {
        "instructions": [
            {"op": "upload", "id": "fn_module", "kind": "module", "source": source},
            {
                "op": "get_function",
                "id": "fn",
                "module": {"$ref": "fn_module"},
                "name": "main",
            },
            {"op": "run", "id": "value", "fn": {"$ref": "fn"}},
            {"op": "return", "key": "value", "value": {"$ref": "value"}},
        ]
    }
    with TestClient(
        create_app(
            ServerConfig(sandbox="none", gpus=[0], max_response_bytes=300),
            runtime_factory=fake_runtime_factory,
        )
    ) as client:
        too_large = post_program(client, program)
    assert too_large.status_code == 500
    assert too_large.json()["error"]["kind"] == "response_too_large"


def test_binary_response_uses_multipart():
    program = {
        "instructions": [
            *harness_instructions("value", "binary"),
            {"op": "return", "key": "value", "value": {"$ref": "value"}},
        ]
    }
    with make_client() as client:
        response = post_program(client, program)
    result, binary = response_parts(response)
    assert response.headers["content-type"].startswith("multipart/form-data")
    assert result["results"]["value"]["sha256"] == compute_blob_hash(b"binary-result")
    assert binary == {"return:0": b"binary-result"}


def test_stdout_stderr_and_output_limit_are_request_level():
    source = (
        "import sys\n"
        "print('load output')\n"
        "def main():\n"
        "    print('run output')\n"
        "    print('error output', file=sys.stderr)\n"
        "    return 1\n"
    )
    program = {
        "instructions": [
            {"op": "upload", "id": "fn_module", "kind": "module", "source": source},
            {
                "op": "get_function",
                "id": "fn",
                "module": {"$ref": "fn_module"},
                "name": "main",
            },
            {"op": "run", "id": "value", "fn": {"$ref": "fn"}},
        ],
        "options": {"output_limit_bytes": 8},
    }
    with make_client() as client:
        data = post_program(client, program).json()
    assert data["stdout"] == "load out" and data["stdout_truncated"] is True
    assert data["stderr"] == "error ou" and data["stderr_truncated"] is True


def test_timeout_and_worker_crash_statuses():
    timeout_program = {
        "instructions": [*harness_instructions("sleep", "sleep", [3])],
        "options": {"timeout_seconds": 0.5},
    }
    crash_program = {"instructions": [*harness_instructions("crash", "crash")]}
    with make_client() as client:
        timeout = post_program(client, timeout_program)
        crash = post_program(client, crash_program)
    assert timeout.status_code == 504 and timeout.json()["error"]["kind"] == "timeout"
    assert crash.status_code == 200
    assert crash.json()["status"] == "FAILED"
    assert crash.json()["error"] == {
        "kind": "runtime",
        "message": "worker exited while executing the instruction",
        "instruction_index": 2,
        "instruction_op": "run",
        "instruction_id": "crash",
        "traceback": "",
    }


def test_poisoned_context_returns_runtime_then_next_request_recovers():
    poison_program = {"instructions": [*harness_instructions("bad", "poison")]}
    app = create_app(
        ServerConfig(sandbox="none", gpus=[0], workers_per_gpu=1),
        runtime_factory=fake_runtime_factory,
    )
    with TestClient(app) as client:
        original_pid = app.state.pool._workers[0]._proc.pid
        failed = post_program(client, poison_program)
        # Read after the recovery request: the replacement is built once the poisoned
        # worker's answer is out, so only that request proves it landed.
        recovered = post_program(client, scalar_program())
        replacement_pid = app.state.pool._workers[0]._proc.pid

    assert failed.status_code == 200
    assert failed.json()["status"] == "FAILED"
    assert failed.json()["error"]["kind"] == "runtime"
    assert replacement_pid != original_pid
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "COMPLETED"
    assert recovered.json()["results"]["answer"]["value"] == 42


SPAWN_AND_HANG = (
    "import subprocess, time\n"
    "def main(pid_file):\n"
    "    child = subprocess.Popen(['sleep', '60'])\n"
    "    open(pid_file, 'w').write(str(child.pid))\n"
    "    time.sleep(60)\n"
)


def _spawner_program(source, pid_file):
    return {
        "instructions": [
            {"op": "upload", "id": "fn_module", "kind": "module", "source": source},
            {
                "op": "get_function",
                "id": "fn",
                "module": {"$ref": "fn_module"},
                "name": "main",
            },
            {"op": "run", "id": "call", "fn": {"$ref": "fn"}, "args": [pid_file]},
        ],
        "options": {"timeout_seconds": 1},
    }


def _wait_until_pid_gone(pid, deadline_seconds=10):
    end = time.monotonic() + deadline_seconds
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_timeout_kills_spawned_process_tree(tmp_path):
    pid_file = str(tmp_path / "pid")
    config = ServerConfig(sandbox="none", gpus=[0], worker_termination_grace_seconds=1)
    with make_client(config) as client:
        response = post_program(client, _spawner_program(SPAWN_AND_HANG, pid_file))
    assert response.status_code == 504
    assert _wait_until_pid_gone(int(open(pid_file).read()))


KERNEL = """from __future__ import annotations
from tvm.script import tirx as T

@T.jit
def main(A: T.Buffer((N,), "float32"), B: T.Buffer((N,), "float32"), *, N: T.constexpr):
    T.device_entry()
    i = T.cta_id([N])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0
"""


@pytest.mark.skipif(
    os.environ.get("KCORAL_GPU_TEST") != "1",
    reason="real-kernel end-to-end test requires KCORAL_GPU_TEST=1",
)
def test_real_kernel_end_to_end():
    import numpy as np

    raw = np.arange(256, dtype=np.float32).tobytes()
    digest = compute_blob_hash(raw)
    program = {
        "instructions": [
            {"op": "upload", "id": "kernel_module", "kind": "module", "source": KERNEL},
            {
                "op": "get_function",
                "id": "kernel",
                "module": {"$ref": "kernel_module"},
                "name": "main",
            },
            {
                "op": "upload",
                "id": "input",
                "kind": "tensor",
                "blob": digest,
                "dtype": "float32",
                "shape": [256],
            },
            *python_instructions(
                "output",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            *harness_instructions("compiled", "compile_tirx", [{"$ref": "kernel"}, {"N": 256}]),
            {
                "op": "run",
                "id": "invoke",
                "fn": {"$ref": "compiled"},
                "args": [{"$ref": "input"}, {"$ref": "output"}],
            },
            {"op": "return", "key": "output", "value": {"$ref": "output"}},
        ],
        "options": {"timeout_seconds": 120},
    }
    app = gpu_app()
    with TestClient(app) as client:
        response = post_program(client, program, {digest: raw})
    result, binary = response_parts(response)
    assert result["status"] == "COMPLETED"
    returned = np.frombuffer(binary["return:0"], dtype=np.float32)
    np.testing.assert_allclose(returned, np.arange(256, dtype=np.float32) + 1)


@pytest.mark.skipif(
    os.environ.get("KCORAL_GPU_TEST") != "1",
    reason="real-kernel end-to-end test requires KCORAL_GPU_TEST=1",
)
def test_a_cpu_only_violation_names_the_request_it_may_have_disturbed(tmp_path):
    """Two workers on one GPU: one holds the lease while the other's cpu_only
    function, running off it, reaches CUDA. The response and the log name the holder."""
    violating = {
        "instructions": [
            {
                "op": "upload",
                "id": "m",
                "kind": "module",
                "source": (
                    "import time, torch\n\n"
                    "def main():\n"
                    "    time.sleep(0.5)\n"
                    "    return float(torch.zeros(1, device='cuda').sum())\n"
                ),
            },
            {
                "op": "get_function",
                "id": "fn",
                "module": {"$ref": "m"},
                "name": "main",
                "cpu_only": True,
            },
            {"op": "run", "id": "call", "fn": {"$ref": "fn"}},
        ],
        "options": {"timeout_seconds": 60},
    }
    holding = {  # a Python upload runs on the GPU lease, and this one sits on it
        "instructions": [
            {
                "op": "upload",
                "id": "m",
                "kind": "module",
                "source": "import time\ntime.sleep(1.0)\n",
            }
        ],
        "options": {"timeout_seconds": 60},
    }
    app = gpu_app(workers_per_gpu=2, log_dir=tmp_path)
    with TestClient(app) as client:
        run_dir = app.state.events.run_dir
        with ThreadPoolExecutor(max_workers=2) as executor:
            violator = executor.submit(post_program, client, violating)
            time.sleep(0.2)  # into its cpu_only call, with the GPU released
            holder = executor.submit(post_program, client, holding)
            violator, _ = response_parts(violator.result())
            holder, _ = response_parts(holder.result())

    assert holder["status"] == "COMPLETED", holder.get("error")
    assert violator["status"] == "FAILED"
    error = violator["error"]
    assert error["kind"] == "gpu_access" and error["cuda_call"].startswith("cu")
    assert error["location"].startswith("<uploaded:")
    assert error["interfered_request_id"] == holder["request_id"]

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    warning = next(event for event in events if event["event"] == "gpu_access_violation")
    assert warning["level"] == "WARNING" and warning["request_id"] == violator["request_id"]
    assert warning["interfered_request_id"] == holder["request_id"]


CUDA_KERNEL = """
__global__ void scale_kernel(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * 3.0f;
}

void scale(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  int n = static_cast<int>(x.numel());
  scale_kernel<<<(n + 255) / 256, 256>>>(static_cast<const float*>(x.data_ptr()),
                                         static_cast<float*>(y.data_ptr()), n);
}
"""

ILLEGAL_ACCESS_KERNEL = """
__global__ void illegal_access_kernel() {
  *reinterpret_cast<volatile int*>(1) = 1;
}

void illegal_access() {
  illegal_access_kernel<<<1, 1>>>();
}
"""

STALE_LAST_ERROR_KERNEL = """
__global__ void never_runs() {}

void stale_launch() {
  never_runs<<<0, 1>>>();
}
"""

CUDA_SYNC = """
def main():
    import torch
    from kcoral.errors import ExecutionError
    try:
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise ExecutionError("runtime", str(exc)) from exc
"""

TRITON_ILLEGAL_ACCESS = """
import torch
import triton
import triton.language as tl

@triton.jit
def _bad(x):
    offsets = tl.arange(0, 256)
    tl.store(x + offsets + 1_000_000_000_000, 1.0)

def run(x):
    torch.cuda.set_device(x.device)
    _bad[(1,)](x)
"""

TRITON_FILL = """
import torch
import triton
import triton.language as tl

@triton.jit
def _fill(x):
    offsets = tl.arange(0, 256)
    tl.store(x + offsets, offsets.to(tl.float32))

def run(x):
    torch.cuda.set_device(x.device)
    _fill[(1,)](x)
    return x
"""


@pytest.mark.skipif(
    os.environ.get("KCORAL_GPU_TEST") != "1",
    reason="real-kernel end-to-end test requires KCORAL_GPU_TEST=1",
)
def test_prebuilt_library_is_cached_like_a_tensor(tmp_path):
    """A client-compiled library goes through the same content-addressed cache as a
    tensor: missing on the first attempt, then served from the cache by hash alone."""
    import numpy as np
    from test_gpu_runtime import CUDA_KERNEL, build_library

    data = build_library(CUDA_KERNEL, "add_one", tmp_path)
    digest = compute_blob_hash(data)
    raw = np.arange(256, dtype=np.float32).tobytes()
    input_digest = compute_blob_hash(raw)
    program = {
        "instructions": [
            {
                "op": "upload",
                "id": "kernels",
                "kind": "library",
                "blob": digest,
            },
            {
                "op": "get_function",
                "id": "kernel",
                "module": {"$ref": "kernels"},
                "name": "add_one",
            },
            {
                "op": "upload",
                "id": "input",
                "kind": "tensor",
                "blob": input_digest,
                "dtype": "float32",
                "shape": [256],
            },
            *python_instructions(
                "output",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            {
                "op": "run",
                "id": "invoke",
                "fn": {"$ref": "kernel"},
                "args": [{"$ref": "input"}, {"$ref": "output"}],
            },
            {"op": "return", "key": "output", "value": {"$ref": "output"}},
        ],
        "options": {"timeout_seconds": 300},
    }
    app = gpu_app()
    with TestClient(app) as client:
        # the server advertises what the library had to be built for
        assert client.get("/health").json()["target"]["arch"].startswith("sm_")

        miss = post_program(client, program).json()
        assert miss["status"] == "CACHE_MISS"
        assert set(miss["missing_blobs"]) == {digest, input_digest}

        uploaded = post_program(client, program, {digest: data, input_digest: raw})
        warm = post_program(client, program)  # hashes only; both blobs are cached
    for response in (uploaded, warm):
        result, binary = response_parts(response)
        assert result["status"] == "COMPLETED", result.get("error")
        np.testing.assert_allclose(
            np.frombuffer(binary["return:0"], dtype=np.float32),
            np.arange(256, dtype=np.float32) + 1,
        )


@pytest.mark.skipif(
    os.environ.get("KCORAL_GPU_TEST") != "1",
    reason="real-kernel end-to-end test requires KCORAL_GPU_TEST=1",
)
def test_cuda_c_kernel_end_to_end():
    import numpy as np

    raw = np.arange(256, dtype=np.float32).tobytes()
    digest = compute_blob_hash(raw)
    program = {
        "instructions": [
            {
                "op": "upload",
                "id": "kernel_module",
                "kind": "module",
                "language": "cuda",
                "source": CUDA_KERNEL,
            },
            {
                "op": "get_function",
                "id": "kernel",
                "module": {"$ref": "kernel_module"},
                "name": "scale",
            },
            {
                "op": "upload",
                "id": "input",
                "kind": "tensor",
                "blob": digest,
                "dtype": "float32",
                "shape": [256],
            },
            *python_instructions(
                "output",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            *harness_instructions("compiled", "compile_cuda", [{"$ref": "kernel"}]),
            {
                "op": "run",
                "id": "invoke",
                "fn": {"$ref": "compiled"},
                "args": [{"$ref": "input"}, {"$ref": "output"}],
            },
            {"op": "return", "key": "output", "value": {"$ref": "output"}},
        ],
        "options": {"timeout_seconds": 300},
    }
    app = gpu_app()
    with TestClient(app) as client:
        response = post_program(client, program, {digest: raw})
    result, binary = response_parts(response)
    assert result["status"] == "COMPLETED", result.get("error")
    returned = np.frombuffer(binary["return:0"], dtype=np.float32)
    np.testing.assert_allclose(returned, np.arange(256, dtype=np.float32) * 3.0)


@pytest.mark.skipif(
    os.environ.get("KCORAL_GPU_TEST") != "1",
    reason="real-kernel end-to-end test requires KCORAL_GPU_TEST=1",
)
def test_illegal_access_replaces_only_worker_and_next_gpu_request_recovers():
    poison_program = {
        "instructions": [
            {
                "op": "upload",
                "id": "kernel_module",
                "kind": "module",
                "language": "cuda",
                "source": ILLEGAL_ACCESS_KERNEL,
            },
            {
                "op": "get_function",
                "id": "kernel",
                "module": {"$ref": "kernel_module"},
                "name": "illegal_access",
            },
            {"op": "upload", "id": "sync_module", "kind": "module", "source": CUDA_SYNC},
            {
                "op": "get_function",
                "id": "sync_fn",
                "module": {"$ref": "sync_module"},
                "name": "main",
            },
            *harness_instructions("compiled", "compile_cuda", [{"$ref": "kernel"}]),
            {"op": "run", "id": "invoke", "fn": {"$ref": "compiled"}},
            {"op": "run", "id": "sync", "fn": {"$ref": "sync_fn"}},
        ],
        "options": {"timeout_seconds": 300},
    }
    healthy_program = {
        "instructions": [
            *python_instructions(
                "q",
                """import torch
def main(shape):
    return torch.zeros(shape, dtype=torch.float32, device="cuda")
""",
                [[16]],
            ),
            {"op": "return", "key": "q", "value": {"$ref": "q"}},
        ],
        "options": {"timeout_seconds": 60},
    }
    app = gpu_app()
    with TestClient(app) as client:
        original_pid = app.state.pool._workers[0]._proc.pid
        failed = post_program(client, poison_program)
        # Read after the recovery request: the replacement is built once the poisoned
        # worker's answer is out, so only that request proves it landed.
        recovered = post_program(client, healthy_program)
        replacement_pid = app.state.pool._workers[0]._proc.pid

    failed_result, _ = response_parts(failed)
    recovered_result, recovered_binary = response_parts(recovered)
    assert failed.status_code == 200
    assert failed_result["status"] == "FAILED"
    assert failed_result["error"]["kind"] == "runtime"
    assert failed_result["error"]["instruction_id"] == "sync"
    assert replacement_pid != original_pid
    assert recovered.status_code == 200
    assert recovered_result["status"] == "COMPLETED", recovered_result.get("error")
    assert recovered_binary["return:0"] == bytes(16 * 4)


@pytest.mark.skipif(
    os.environ.get("KCORAL_GPU_TEST") != "1",
    reason="real-kernel end-to-end test requires KCORAL_GPU_TEST=1",
)
def test_parallel_triton_illegal_accesses_match_and_workers_recover(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))

    def triton_program(source, *, return_output=False):
        instructions = [
            {"op": "upload", "id": "kernel_module", "kind": "module", "source": source},
            {
                "op": "get_function",
                "id": "kernel",
                "module": {"$ref": "kernel_module"},
                "name": "run",
            },
            *python_instructions(
                "output",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            {
                "op": "run",
                "id": "invoke",
                "fn": {"$ref": "kernel"},
                "args": [{"$ref": "output"}],
            },
        ]
        if return_output:
            instructions.append({"op": "return", "key": "output", "value": {"$ref": "invoke"}})
        return {"instructions": instructions, "options": {"timeout_seconds": 300}}

    poison_program = triton_program(TRITON_ILLEGAL_ACCESS)
    healthy_program = triton_program(TRITON_FILL, return_output=True)
    app = gpu_app(
        workers_per_gpu=2,
        max_requests_per_worker=0,
        worker_wait_timeout_seconds=300,
    )
    with TestClient(app) as client:
        original_pids = [worker._proc.pid for worker in app.state.pool._workers]
        with ThreadPoolExecutor(max_workers=8) as executor:
            failed = list(executor.map(lambda _: post_program(client, poison_program), range(8)))
        replacement_pids = [worker._proc.pid for worker in app.state.pool._workers]
        recovered = post_program(client, healthy_program)

    failed_results = [response_parts(response)[0] for response in failed]
    error_signatures = {json.dumps(result["error"], sort_keys=True) for result in failed_results}
    assert all(response.status_code == 200 for response in failed)
    assert all(result["status"] == "FAILED" for result in failed_results)
    assert len(error_signatures) == 1
    error = failed_results[0]["error"]
    assert error["kind"] == "runtime", error
    assert error["instruction_id"] == "invoke"
    assert "illegal memory access" in error["message"].lower()
    assert all(new_pid != old_pid for old_pid, new_pid in zip(original_pids, replacement_pids))

    recovered_result, recovered_binary = response_parts(recovered)
    assert recovered.status_code == 200
    assert recovered_result["status"] == "COMPLETED", json.dumps(
        recovered_result.get("error"), sort_keys=True
    )
    np.testing.assert_array_equal(
        np.frombuffer(recovered_binary["return:0"], dtype=np.float32),
        np.arange(256, dtype=np.float32),
    )


@pytest.mark.skipif(
    os.environ.get("KCORAL_GPU_TEST") != "1",
    reason="real-kernel end-to-end test requires KCORAL_GPU_TEST=1",
)
def test_cuda_last_error_fails_current_request_without_replacing_worker():
    stale_error_program = {
        "instructions": [
            {
                "op": "upload",
                "id": "kernel_module",
                "kind": "module",
                "language": "cuda",
                "source": STALE_LAST_ERROR_KERNEL,
            },
            {
                "op": "get_function",
                "id": "kernel",
                "module": {"$ref": "kernel_module"},
                "name": "stale_launch",
            },
            *harness_instructions("compiled", "compile_cuda", [{"$ref": "kernel"}]),
            {"op": "run", "id": "invoke", "fn": {"$ref": "compiled"}},
        ],
        "options": {"timeout_seconds": 300},
    }
    healthy_program = {
        "instructions": [
            *python_instructions(
                "q",
                """import torch
def main(shape):
    return torch.zeros(shape, dtype=torch.float32, device="cuda")
""",
                [[16]],
            ),
            {"op": "return", "key": "q", "value": {"$ref": "q"}},
        ],
        "options": {"timeout_seconds": 60},
    }
    app = gpu_app()
    with TestClient(app) as client:
        original_pid = app.state.pool._workers[0]._proc.pid
        failed = post_program(client, stale_error_program)
        worker_pid_after_failure = app.state.pool._workers[0]._proc.pid
        recovered = post_program(client, healthy_program)

    failed_result, _ = response_parts(failed)
    recovered_result, recovered_binary = response_parts(recovered)
    assert failed.status_code == 200
    assert failed_result["status"] == "FAILED"
    assert failed_result["error"]["kind"] == "runtime"
    assert failed_result["error"]["instruction_id"] == "invoke"
    assert "cudaErrorInvalidValue" in failed_result["error"]["message"]
    assert worker_pid_after_failure == original_pid
    assert recovered.status_code == 200
    assert recovered_result["status"] == "COMPLETED", recovered_result.get("error")
    assert recovered_binary["return:0"] == bytes(16 * 4)


@pytest.mark.skipif(
    os.environ.get("KCORAL_GPU_TEST") != "1",
    reason="real-kernel end-to-end test requires KCORAL_GPU_TEST=1",
)
def test_uploaded_host_compilation_does_not_hold_the_gpu():
    """Explicit-architecture CUDA building is host-only; loading is a separate step."""
    app = gpu_app()
    with TestClient(app) as client:
        arch = client.get("/health").json()["target"]["arch"]
        program = {
            "instructions": [
                {
                    "op": "upload",
                    "id": "source",
                    "kind": "module",
                    "language": "cuda",
                    "source": CUDA_KERNEL + f"\n// uncached build {time.time_ns()}\n",
                },
                {
                    "op": "get_function",
                    "id": "kernel",
                    "module": {"$ref": "source"},
                    "name": "scale",
                },
                *harness_instructions(
                    "compiled", "compile_cuda_binary", [{"$ref": "kernel"}, {"arch": arch}]
                ),
                {"op": "return", "key": "library", "value": {"$ref": "compiled"}},
            ],
            "options": {"timeout_seconds": 120},
        }
        response = post_program(client, program)
    result, parts = response_parts(response)
    assert result["status"] == "COMPLETED", result.get("error")
    assert parts["return:0"].startswith(b"\x7fELF")
    assert result["elapsed_ms"] > 100
    assert result["lease_held_ms"] < 50


def test_request_ids_accept_one_canonical_uuid_and_replace_unsafe_values():
    import uuid

    valid = "c1a92a02-34e1-4c28-a84e-a9c35c7e672b"
    cases = [
        ([], False),
        ([("x-request-id", valid)], True),
        ([("x-request-id", "../unsafe-path")], False),
        ([("x-request-id", valid.upper())], False),
        ([("x-request-id", valid), ("x-request-id", valid)], False),
    ]
    config = ServerConfig(sandbox="none", device="cpu", num_workers=1, max_requests_per_worker=0)
    with make_client(config) as client:
        for headers, accepted in cases:
            # Early rejection also uses the shared, validated request ID.
            response = client.post("/execute", content=b"invalid", headers=headers)
            request_id = response.headers["x-request-id"]
            assert str(uuid.UUID(request_id)) == request_id
            assert response.json()["request_id"] == request_id
            assert (request_id == valid) is accepted


def test_server_supplies_file_collection_byte_limit():
    config = ServerConfig(
        sandbox="none",
        max_requests_per_worker=0,
        max_response_bytes=4096,
    )
    source = """
from pathlib import Path
def make():
    Path("out").mkdir()
    Path("out/large").write_bytes(b"x" * 8192)
    return 7
"""
    program = {
        "instructions": [
            {"op": "upload", "kind": "module", "id": "module", "source": source},
            {"op": "get_function", "id": "fn", "module": {"$ref": "module"}, "name": "make"},
            {"op": "run", "id": "value", "fn": {"$ref": "fn"}},
            {"op": "return", "key": "kept", "value": {"$ref": "value"}},
            {"op": "return", "key": "out", "kind": "folder", "path": "out"},
        ]
    }
    with make_client(config) as client:
        response = post_program(client, program)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAILED"
    assert body["results"] == {"kept": {"type": "integer", "value": 7}}
    assert body["error"]["kind"] == "serialization"
    assert "max_response_bytes" in body["error"]["message"]


def test_file_return_still_obeys_final_serialized_response_cap():
    source = 'from pathlib import Path\nPath("report").write_bytes(b"x" * 4000)\n'
    program = {
        "instructions": [
            {"op": "upload", "kind": "module", "id": "module", "source": source},
            {"op": "return", "key": "report", "kind": "file", "path": "report"},
        ]
    }
    with make_client(
        ServerConfig(sandbox="none", max_requests_per_worker=0, max_response_bytes=4096)
    ) as client:
        response = post_program(client, program)
    assert response.status_code == 500
    assert response.json()["error"]["kind"] == "response_too_large"
