from __future__ import annotations

import hashlib
import json
from email.parser import BytesParser
from email.policy import default

import pytest
from fastapi.testclient import TestClient

from benchmark_server import BenchmarkServerError
from benchmark_server import Client as SDKClient
from benchmark_server import ServerConfig, create_app


@pytest.fixture
def client(tmp_path):
    config = ServerConfig(
        devices=("0", "1"),
        cache_dir=tmp_path / "cache",
        cache_capacity_bytes=1024**2,
        work_dir=tmp_path / "work",
        log_dir=tmp_path / "logs",
        default_timeout_seconds=2,
        max_timeout_seconds=5,
        worker_termination_grace_seconds=0.1,
    )
    with TestClient(create_app(config)) as value:
        yield value


def _execute(client: TestClient, source: bytes, *, extra=None, **settings):
    digest = hashlib.sha256(source).hexdigest()
    job = {"files": {"main.py": {"blob": digest}}, **settings}
    files = [
        ("job", (None, json.dumps(job), "application/json")),
        (f"blob:{digest}", (digest, source, "application/octet-stream")),
    ]
    files.extend(extra or [])
    return client.post("/execute", files=files)


def _execute_instructions(client: TestClient, instructions, blobs):
    files = [
        (
            "job",
            (None, json.dumps({"instructions": instructions}), "application/json"),
        )
    ]
    files.extend(
        (f"blob:{digest}", (digest, data, "application/octet-stream"))
        for digest, data in blobs.items()
    )
    return client.post("/execute", files=files)


def test_execute_json_and_output(client):
    response = _execute(
        client,
        b"import os\ndef main():\n print('hello')\n return {'answer': 42, 'request': os.environ['GPU_SERVER_REQUEST_ID']}\n",
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["request_id"] == response.headers["x-request-id"]
    assert body["return"]["value"]["answer"] == 42
    assert body["return"]["value"]["request"] == body["request_id"]
    assert body["stdout"] == "hello\n"


def test_binary_and_tuple_return(client):
    response = _execute(client, b"def main():\n return (b'abc', {'ok': True})\n")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("multipart/form-data")
    message = BytesParser(policy=default).parsebytes(
        b"MIME-Version: 1.0\r\nContent-Type: "
        + response.headers["content-type"].encode()
        + b"\r\n\r\n"
        + response.content
    )
    parts = {
        part.get_param("name", header="content-disposition"): part.get_payload(
            decode=True
        )
        for part in message.iter_parts()
    }
    metadata = json.loads(parts["result"])
    assert metadata["return"]["type"] == "tuple"
    assert parts["return:0"] == b"abc"


def test_blob_endpoints_and_cache_only_execute(client):
    source = b"def main():\n return 7\n"
    digest = hashlib.sha256(source).hexdigest()
    assert client.post("/blobs/check", json={"blobs": [digest]}).json() == {
        "missing": [digest]
    }
    upload = client.post(
        "/blobs",
        files=[(f"blob:{digest}", (digest, source, "application/octet-stream"))],
    )
    assert upload.json() == {"status": "ok", "stored": [digest], "already_present": []}
    job = {"files": {"main.py": {"blob": digest}}}
    response = client.post(
        "/execute", files=[("job", (None, json.dumps(job), "application/json"))]
    )
    assert response.status_code == 200
    assert response.json()["return"] == {"type": "json", "value": 7}


def test_invalid_request_execution_error_and_timeout(client):
    invalid = client.post("/execute", content=b"not multipart")
    assert invalid.status_code == 400
    assert invalid.json()["error"] == "invalid_request"
    assert invalid.headers["x-request-id"] == invalid.json()["request_id"]

    failed = _execute(
        client, b"def main():\n print('before')\n raise RuntimeError('boom')\n"
    )
    assert failed.status_code == 400
    assert failed.json()["error"] == "execution_failed"
    assert failed.json()["stdout"] == "before\n"
    assert "RuntimeError: boom" in failed.json()["traceback"]

    timed_out = _execute(
        client,
        b"import time\ndef main():\n print('waiting', flush=True)\n time.sleep(2)\n",
        timeout_seconds=0.1,
    )
    assert timed_out.status_code == 408
    assert timed_out.json()["error"] == "timeout"
    assert timed_out.json()["stdout"] == "waiting\n"


def test_unused_blob_warning_and_validation(client):
    source = b"def main():\n return 1\n"
    unused = b"unused"
    unused_hash = hashlib.sha256(unused).hexdigest()
    response = _execute(
        client,
        source,
        extra=[
            (f"blob:{unused_hash}", (unused_hash, unused, "application/octet-stream"))
        ],
    )
    assert response.status_code == 200
    assert response.json()["warnings"] == [
        {"code": "unused_blob", "blobs": [unused_hash]}
    ]
    assert client.post("/blobs/check", json={"blobs": [unused_hash]}).json() == {
        "missing": [unused_hash]
    }

    duplicate = client.post("/blobs/check", json={"blobs": [unused_hash, unused_hash]})
    assert duplicate.status_code == 400
    unsafe_source = b"def main(): return 1"
    digest = hashlib.sha256(unsafe_source).hexdigest()
    unsafe_job = {"files": {"../main.py": {"blob": digest}}}
    unsafe = client.post(
        "/execute", files=[("job", (None, json.dumps(unsafe_job), "application/json"))]
    )
    assert unsafe.status_code == 400


def test_python_sdk_end_to_end(client):
    sdk = SDKClient("http://testserver")
    sdk._http.close()
    sdk._http = client

    result = sdk.execute(
        {"main.py": "def main():\n return (b'payload', {'value': 9})\n"},
        timeout_seconds=2,
    )
    assert result.value == (b"payload", {"value": 9})
    assert result.request_id
    assert sdk.health().gpu_count == 2

    prepared = sdk.prepare_files({"main.py": "def main():\n return 10\n"})
    assert sdk.execute_prepared(prepared).value == 10

    runtime = client.app.state.runtime
    digest = prepared.manifest["main.py"]
    runtime.cache._entries.pop(digest).path.unlink()
    with pytest.raises(BenchmarkServerError, match="blob_not_found") as error:
        sdk.execute_prepared(prepared)
    assert error.value.missing_blobs == (digest,)

    fallback_source = "def main():\n return 11\n"
    fallback = sdk.prepare_files({"main.py": fallback_source})
    fallback_digest = fallback.manifest["main.py"]
    original_check = runtime.cache.check

    async def evict_after_check(digests):
        missing = await original_check(digests)
        entry = runtime.cache._entries.pop(fallback_digest, None)
        if entry is not None:
            entry.path.unlink()
        return missing

    runtime.cache.check = evict_after_check
    try:
        assert sdk.execute({"main.py": fallback_source}).value == 11
    finally:
        runtime.cache.check = original_check


def test_instruction_sdk_module_call_and_request_isolation(client):
    sdk = SDKClient("http://testserver")
    sdk._http.close()
    sdk._http = client
    source = (
        "import tvm_ffi\n"
        "@tvm_ffi.register_global_func('benchmark_server_test.add')\n"
        "def add(left, right):\n"
        " return left + right\n"
    )
    digest = hashlib.sha256(source.encode()).hexdigest()
    instructions = [
        {"op": "upload_module", "blob": digest},
        {
            "op": "call",
            "dst": 0,
            "func": "benchmark_server_test.add",
            "args": [4, 5],
        },
        {"op": "return", "reg": 0, "key": "sum"},
    ]

    result = sdk.execute_instructions(instructions, {digest: source})
    assert result.value == {"sum": 9}

    with pytest.raises(BenchmarkServerError, match="invalid_program") as error:
        sdk.execute_instructions(instructions[1:], {})
    assert error.value.instruction_index == 0


def test_instruction_tensor_call_round_trip(client):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    source = (
        "import torch\n"
        "import tvm_ffi\n"
        "@tvm_ffi.register_global_func('benchmark_server_test.scale')\n"
        "def scale(tensor, factor):\n"
        " return torch.from_dlpack(tensor) * factor\n"
    ).encode()
    tensor_data = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor_bytes = tensor_data.tobytes()
    module_digest = hashlib.sha256(source).hexdigest()
    tensor_digest = hashlib.sha256(tensor_bytes).hexdigest()
    instructions = [
        {"op": "upload_module", "blob": module_digest},
        {
            "op": "upload_tensor",
            "dst": 0,
            "blob": tensor_digest,
            "shape": [2, 3],
            "dtype": "float32",
            "device": "cpu",
        },
        {
            "op": "call",
            "dst": 1,
            "func": "benchmark_server_test.scale",
            "args": [{"reg": 0}, 2.0],
        },
        {"op": "return", "reg": 1, "key": "output"},
    ]
    sdk = SDKClient("http://testserver")
    sdk._http.close()
    sdk._http = client
    result = sdk.execute_instructions(
        instructions,
        {module_digest: source, tensor_digest: tensor_bytes},
    )
    np.testing.assert_array_equal(
        np.from_dlpack(result.value["output"]), tensor_data * 2
    )


def test_instruction_validation_and_runtime_errors(client):
    undefined = _execute_instructions(
        client, [{"op": "return", "reg": 0, "key": "output"}], {}
    )
    assert undefined.status_code == 400
    assert undefined.json()["error"] == "invalid_request"

    tensor_bytes = b"\x00\x00\x00\x00"
    tensor_digest = hashlib.sha256(tensor_bytes).hexdigest()
    wrong_size = _execute_instructions(
        client,
        [
            {
                "op": "upload_tensor",
                "dst": 0,
                "blob": tensor_digest,
                "shape": [2],
                "dtype": "float32",
                "device": "cpu",
            }
        ],
        {tensor_digest: tensor_bytes},
    )
    assert wrong_size.status_code == 400
    assert wrong_size.json()["error"] == "invalid_program"
    assert wrong_size.json()["instruction_index"] == 0


def test_tensor_round_trip_through_sdk(client):
    sdk = SDKClient("http://testserver")
    sdk._http.close()
    sdk._http = client
    result = sdk.execute(
        {
            "main.py": (
                "import numpy as np\n"
                "def main():\n"
                " return np.arange(6, dtype=np.float32).reshape(2, 3)\n"
            )
        },
        timeout_seconds=5,
    )
    import numpy as np
    import tvm_ffi

    assert isinstance(result.value, tvm_ffi.Tensor)
    np.testing.assert_array_equal(
        np.from_dlpack(result.value), np.arange(6, dtype=np.float32).reshape(2, 3)
    )
