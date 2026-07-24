"""Client SDK tests against a real uvicorn server over TCP."""

import base64
import os
import threading
import time

import pytest
import uvicorn

from benchmark_server.app import create_app
from benchmark_server.client import (
    BenchmarkServerError,
    Client,
    ProtocolError,
    TransportError,
    ref,
    run,
    upload_function,
    upload_tensor,
)
from benchmark_server.config import ServerConfig
from benchmark_server.keys import compute_key
from benchmark_server.testing import fake_runtime_factory

ADD_ONE = "def main(a):\n    return a + 1\n"


def _start_server(app):
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("test server did not start")
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}"


@pytest.fixture(scope="module")
def server_url():
    app = create_app(ServerConfig(gpus=[0]), runtime_factory=fake_runtime_factory)
    server, thread, url = _start_server(app)
    yield url
    server.should_exit = True
    thread.join(timeout=10)


def test_execute_with_transparent_cache_miss_retry(server_url):
    program = [upload_function("fn", ADD_ONE), run("y", ref("fn"), [41])]
    with Client(server_url) as client:
        # Fresh server: the key-only fast path misses, the client resends inline.
        first = client.execute(program)
        # Warm cache: the fast path succeeds without any inline bytes.
        second = client.execute(program)
    for outcome in (first, second):
        assert outcome.completed
        assert outcome["y"].value == 42
        assert outcome.request_id
        assert outcome.queue_ms >= 0 and outcome.elapsed_ms >= 0


def test_failed_instruction_is_data_not_an_exception(server_url):
    with Client(server_url) as client:
        outcome = client.execute([run("bad", "builtin.nope", [])])
    assert outcome.status == "FAILED"
    assert outcome["bad"].error["kind"] == "runtime"


def test_timeout_raises_server_error(server_url):
    with Client(server_url) as client:
        with pytest.raises(BenchmarkServerError) as exc_info:
            client.execute([run("s", "builtin.sleep", [5.0])], timeout_seconds=0.5)
    assert exc_info.value.status_code == 504
    assert exc_info.value.kind == "timeout"
    assert exc_info.value.request_id


def test_missing_bytes_raise_protocol_error(server_url):
    key_only_upload = {
        "id": "fn",
        "op": "upload",
        "kind": "function",
        "key": "sha256:" + "0" * 64,
    }
    with Client(server_url) as client:
        with pytest.raises(ProtocolError):
            client.execute([key_only_upload, run("y", ref("fn"), [1])])


def test_prepare_then_key_only_execute(server_url):
    # a source unique to this test, so the module-scoped server's cache is cold
    source = "def main(a):\n    return a * 3\n"
    program = [upload_function("fn", source), run("y", ref("fn"), [2])]
    with Client(server_url) as client:
        uploaded = client.prepare(program)
        assert uploaded == [program[0]["key"]]
        # even a program carrying no inline bytes at all now runs
        stripped = [{k: v for k, v in ins.items() if k != "inline"} for ins in program]
        outcome = client.execute(stripped)
        assert outcome.completed and outcome["y"].value == 6
        assert client.prepare(program) == []  # nothing missing the second time


def test_health(server_url):
    with Client(server_url) as client:
        health = client.health()
    assert health["gpu_count"] == 1 and health["workers"][0]["status"] == "idle"


def test_connection_refused_is_transport_error():
    with Client("http://127.0.0.1:1", connect_timeout_seconds=1.0) as client:
        with pytest.raises(TransportError):
            client.health()


# --- instruction builders -----------------------------------------------------


def test_upload_function_key_matches_canonical_form():
    upload = upload_function("fn", ADD_ONE)
    assert upload["key"] == compute_key("function", {"source": ADD_ONE})


def test_upload_tensor_from_numpy():
    numpy = pytest.importorskip("numpy")
    array = numpy.arange(6, dtype=numpy.float32).reshape(2, 3)
    upload = upload_tensor("x", array)
    assert upload["kind"] == "tensor"
    assert upload["inline"]["dtype"] == "float32" and upload["inline"]["shape"] == [2, 3]
    assert base64.b64decode(upload["inline"]["data_b64"]) == array.tobytes()
    assert upload["key"] == compute_key("tensor", upload["inline"])


def test_upload_tensor_from_torch_bfloat16():
    torch = pytest.importorskip("torch")
    tensor = torch.arange(4, dtype=torch.float32).to(torch.bfloat16).reshape(2, 2)
    upload = upload_tensor("x", tensor)
    assert upload["inline"]["dtype"] == "bfloat16" and upload["inline"]["shape"] == [2, 2]
    assert len(base64.b64decode(upload["inline"]["data_b64"])) == 4 * 2


# --- real tensors over the full stack (client -> HTTP -> worker -> GPU) -------


@pytest.mark.skipif(
    os.environ.get("BENCH_GPU_TEST") != "1",
    reason="real-tensor client e2e; set BENCH_GPU_TEST=1 with a GPU to run",
)
def test_tensor_roundtrip_on_gpu():
    import numpy

    from benchmark_server.gpu_runtime import gpu_runtime_factory

    gpu_id_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    gpu_id = int(gpu_id_raw) if gpu_id_raw.isdigit() else 0
    app = create_app(ServerConfig(gpus=[gpu_id]), runtime_factory=gpu_runtime_factory)
    server, thread, url = _start_server(app)
    try:
        x = numpy.arange(64, dtype=numpy.float32)
        with Client(url) as client:
            outcome = client.execute(
                [
                    upload_tensor("x", x),
                    upload_tensor("expected", x * 2),
                    upload_function("double", "def main(a):\n    return a * 2\n"),
                    run("y", ref("double"), [ref("x")]),
                    run("chk", "builtin.check_close", [ref("y"), ref("expected")]),
                ],
                timeout_seconds=120,
            )
        assert outcome.completed
        assert outcome["y"].value == {"handle": "y"}  # GPU tensors stay server-side
        assert outcome["chk"].value["passed"] and outcome["chk"].value["max_abs_err"] == 0.0
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.mark.skipif(
    os.environ.get("BENCH_GPU_TEST") != "1",
    reason="real-tensor download e2e; set BENCH_GPU_TEST=1 with a GPU to run",
)
def test_download_tensor_from_gpu():
    import numpy

    from benchmark_server.client import decode_tensor
    from benchmark_server.gpu_runtime import gpu_runtime_factory

    gpu_id_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    gpu_id = int(gpu_id_raw) if gpu_id_raw.isdigit() else 0
    app = create_app(ServerConfig(gpus=[gpu_id]), runtime_factory=gpu_runtime_factory)
    server, thread, url = _start_server(app)
    try:
        x = numpy.linspace(-1.0, 1.0, 128, dtype=numpy.float32)
        with Client(url) as client:
            client.prepare([upload_tensor("x", x)])  # raw binary pre-upload path
            outcome = client.execute(
                [
                    upload_tensor("x", x),
                    upload_function("double", "def main(a):\n    return a * 2\n"),
                    run("y", ref("double"), [ref("x")]),
                    run("out", "builtin.download", [ref("y")]),
                ],
                timeout_seconds=120,
            )
        assert outcome.completed
        downloaded = decode_tensor(outcome["out"].value)  # torch tensor (torch importable)
        numpy.testing.assert_array_equal(downloaded.numpy(), x * 2)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
