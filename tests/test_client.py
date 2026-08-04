"""Client tests against a real uvicorn server over TCP."""

import os
import threading
import time

import httpx
import pytest
import uvicorn

from benchmark_server import Client, Program, Register
from benchmark_server.app import create_app
from benchmark_server.client import (
    BenchmarkServerError,
    ProtocolError,
    TransportError,
    _parse_program_result,
    _response_body,
)
from benchmark_server.config import ServerConfig
from benchmark_server.keys import compute_blob_hash
from benchmark_server.multipart import MultipartPart, encode_multipart
from benchmark_server.testing import fake_runtime_factory


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


def add_one_program():
    program = Program()
    fn = program.upload(id="fn", kind="module", source="def main(x):\n    return x + 1\n")
    answer = program.run(id="answer", fn=fn, args=[41])
    program.return_(key="answer", value=answer)
    return program


def test_execute_module_and_return_value(server_url):
    with Client(server_url) as client:
        outcome = client.execute(add_one_program())
    assert outcome.completed and outcome.results == {"answer": 42}
    assert outcome.request_id and outcome.queue_ms >= 0 and outcome.elapsed_ms >= 0


def test_tensor_cache_retry_and_tvm_ffi_result(server_url):
    np = pytest.importorskip("numpy")
    tvm_ffi = pytest.importorskip("tvm_ffi")
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    program = Program()
    tensor = program.upload(id="tensor", kind="tensor", value=value)
    program.return_(key="tensor", value=tensor)
    with Client(server_url) as client:
        first = client.execute(program)
        second = client.execute(program)
    for outcome in (first, second):
        assert isinstance(outcome.results["tensor"], tvm_ffi.Tensor)
        np.testing.assert_array_equal(np.from_dlpack(outcome.results["tensor"]), value)


def test_nested_binary_results_are_decoded(server_url):
    program = Program()
    fn = program.upload(
        id="fn",
        kind="module",
        source="def main():\n    return [b'a', {'nested': b'b'}]\n",
    )
    value = program.run(id="value", fn=fn)
    program.return_(key="value", value=value)
    with Client(server_url) as client:
        outcome = client.execute(program)
    assert outcome.results == {"value": [b"a", {"nested": b"b"}]}


def test_failed_instruction_is_data(server_url):
    program = Program()
    program.run(id="bad", fn="builtin.nope")
    with Client(server_url) as client:
        outcome = client.execute(program)
    assert outcome.status == "FAILED" and outcome.results == {}
    assert outcome.error["kind"] == "runtime"


def test_interleaved_return_survives_a_later_failure(server_url):
    program = Program()
    fn = program.upload(id="fn", kind="module", source="def main():\n    return b'checkpoint'\n")
    early = program.run(id="early", fn=fn)
    program.return_(key="early", value=early)  # checkpointed before the failure
    program.run(id="bad", fn="builtin.nope")
    with Client(server_url) as client:
        outcome = client.execute(program)
    assert outcome.status == "FAILED"
    assert outcome.results == {"early": b"checkpoint"}
    assert outcome.error["instruction_op"] == "run" and outcome.error["instruction_id"] == "bad"
    assert "Traceback" in outcome.error["traceback"]


def test_timeout_raises_server_error(server_url):
    program = Program()
    program.run(id="sleep", fn="builtin.sleep", args=[5])
    with Client(server_url) as client:
        with pytest.raises(BenchmarkServerError) as exc_info:
            client.execute(program, timeout_seconds=0.5)
    assert exc_info.value.status_code == 504 and exc_info.value.kind == "timeout"
    assert exc_info.value.request_id


def test_missing_local_blob_is_protocol_error(server_url):
    program = Program()
    program.upload(id="tensor", kind="tensor", value=b"\x00\x00\x80?", dtype="float32", shape=[1])
    program._blobs.clear()
    with Client(server_url) as client:
        with pytest.raises(ProtocolError, match="no local bytes"):
            client.execute(program)


def test_health_and_transport_errors(server_url):
    with Client(server_url) as client:
        assert client.health()["gpu_count"] == 1
    with Client("http://127.0.0.1:1", connect_timeout_seconds=1) as client:
        with pytest.raises(TransportError):
            client.health()


def test_program_builder_allows_interleaved_returns():
    program = Program()
    register = program.upload(id="module", kind="module", source="def main(): pass\n")
    program.return_(key="module", value=register)
    program.run(id="later", fn="builtin.structural")
    program.return_(key="later", value=Register("later"))
    assert len(program.instructions) == 4

    with pytest.raises(ValueError, match="unknown handle"):
        Program().return_(key="missing", value=Register("nope"))


def test_program_builder_validates_ids_and_tensor_metadata():
    with pytest.raises(ValueError, match="expects 4 bytes"):
        Program().upload(id="bad", kind="tensor", value=b"abc", dtype="float32", shape=[1])

    reusable = Program()
    with pytest.raises(ValueError, match="expects 4 bytes"):
        reusable.upload(id="tensor", kind="tensor", value=b"abc", dtype="float32", shape=[1])
    reusable.upload(id="tensor", kind="tensor", value=b"\x00" * 4, dtype="float32", shape=[1])


def test_numpy_tensor_builder_uses_raw_byte_hash():
    np = pytest.importorskip("numpy")
    value = np.arange(4, dtype=np.float32)
    program = Program()
    program.upload(id="tensor", kind="tensor", value=value)
    instruction = program.instructions[0]
    assert instruction["dtype"] == "float32" and instruction["shape"] == [4]
    assert instruction["blob"] == compute_blob_hash(value.tobytes())


def test_torch_bfloat16_tensor_builder():
    torch = pytest.importorskip("torch")
    value = torch.arange(4, dtype=torch.float32).to(torch.bfloat16).reshape(2, 2)
    program = Program()
    program.upload(id="tensor", kind="tensor", value=value)
    instruction = program.instructions[0]
    assert instruction["dtype"] == "bfloat16" and instruction["shape"] == [2, 2]
    assert len(program._blobs[instruction["blob"]]) == 8


FAILED_ERROR = {
    "kind": "runtime",
    "message": "bad",
    "instruction_index": 2,
    "instruction_op": "run",
    "instruction_id": "boom",
    "traceback": "Traceback (most recent call last):\n  ...",
}


def test_client_rejects_malformed_execution_responses():
    common = {
        "request_id": "request",
        "queue_ms": 0,
        "elapsed_ms": 1,
        "stdout": "",
        "stderr": "",
    }
    with pytest.raises(ProtocolError, match="unexpected fields"):
        _parse_program_result(
            {
                **common,
                "status": "FAILED",
                "results": {},
                "error": {"kind": "runtime", "message": "bad", "instruction_index": 0},
            },
            {},
        )

    with pytest.raises(ProtocolError, match="instruction_op is invalid"):
        _parse_program_result(
            {
                **common,
                "status": "FAILED",
                "results": {},
                "error": {**FAILED_ERROR, "instruction_op": "nope"},
            },
            {},
        )

    data = b"binary"
    with pytest.raises(ProtocolError, match="invalid part index"):
        _parse_program_result(
            {
                **common,
                "status": "COMPLETED",
                "results": {
                    "value": {
                        "type": "bytes",
                        "part": "return:00",
                        "sha256": compute_blob_hash(data),
                    }
                },
            },
            {"return:00": data},
        )

    with pytest.raises(ProtocolError, match="depth-first order"):
        _parse_program_result(
            {
                **common,
                "status": "COMPLETED",
                "results": {
                    "value": {
                        "type": "bytes",
                        "part": "return:1",
                        "sha256": compute_blob_hash(data),
                    }
                },
            },
            {"return:1": data},
        )


def test_client_rejects_invalid_json_in_multipart_result():
    body, content_type = encode_multipart(
        [MultipartPart("result", "application/json", b'{"status":1,"status":2}')]
    )
    response = httpx.Response(200, headers={"Content-Type": content_type}, content=body)
    with pytest.raises(ProtocolError, match="not valid JSON"):
        _response_body(response)


@pytest.mark.skipif(
    os.environ.get("BENCH_GPU_TEST") != "1",
    reason="real tensor client test requires BENCH_GPU_TEST=1",
)
def test_tensor_round_trip_on_gpu():
    import numpy as np

    from benchmark_server.gpu_runtime import gpu_runtime_factory

    gpu_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    gpu_id = int(gpu_raw) if gpu_raw.isdigit() else 0
    app = create_app(ServerConfig(gpus=[gpu_id]), runtime_factory=gpu_runtime_factory)
    server, thread, url = _start_server(app)
    try:
        value = np.arange(64, dtype=np.float32)
        program = Program()
        tensor = program.upload(id="tensor", kind="tensor", value=value)
        program.return_(key="tensor", value=tensor)
        with Client(url) as client:
            outcome = client.execute(program)
        np.testing.assert_array_equal(np.from_dlpack(outcome.results["tensor"]), value)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
