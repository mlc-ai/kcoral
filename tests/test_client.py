"""Client tests against a real uvicorn server over TCP."""

import os
import threading
import time

import httpx
import ml_dtypes
import numpy as np
import pytest
import uvicorn

from benchmark_server import Client, Program, Register
from benchmark_server.app import create_app
from benchmark_server.client import (
    _NUMPY_DTYPES,
    BenchmarkServerError,
    ProtocolError,
    TransportError,
    _decode_tensor,
    _parse_program_result,
    _response_body,
)
from benchmark_server.config import ServerConfig
from benchmark_server.keys import compute_blob_hash
from benchmark_server.multipart import MultipartPart, encode_multipart
from benchmark_server.schemas import DTYPE_ITEM_SIZES, expected_tensor_nbytes
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


# The three protocol dtypes numpy has no native scalar type for; ml_dtypes
# supplies them, and they are the reason it is a dependency.
ML_DTYPE_NAMES = ("bfloat16", "float8_e4m3fn", "float8_e5m2")


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


def test_tensor_cache_retry_and_numpy_result(server_url):
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    program = Program()
    tensor = program.upload(id="tensor", kind="tensor", value=value)
    program.return_(key="tensor", value=tensor)
    with Client(server_url) as client:
        first = client.execute(program)
        second = client.execute(program)
    for outcome in (first, second):
        result = outcome.results["tensor"]
        assert isinstance(result, np.ndarray)
        assert result.flags.writeable and result.flags.owndata
        np.testing.assert_array_equal(result, value)


def test_bytes_cache_retry_and_result(server_url):
    value = b"safetensors contents\x00\xff"
    program = Program()
    inspect_file = program.upload(
        id="inspect_file",
        kind="module",
        source=(
            "def main(data):\n    return {'size': len(data), 'format': data[:11].decode('ascii')}\n"
        ),
    )
    file_data = program.upload(id="file", kind="bytes", value=value)
    result = program.run(id="result", fn=inspect_file, args=[file_data])
    program.return_(key="file", value=result)
    with Client(server_url) as client:
        first = client.execute(program)
        second = client.execute(program)
    expected = {"file": {"size": len(value), "format": "safetensors"}}
    assert first.results == expected
    assert second.results == expected


def test_cache_churn_falls_back_to_all_blobs():
    app = create_app(
        ServerConfig(gpus=[0], workers_per_gpu=1, cache_capacity_bytes=16),
        runtime_factory=fake_runtime_factory,
    )
    server, thread, url = _start_server(app)
    values = [np.array([index], dtype=np.float32) for index in range(5)]
    try:
        with Client(url) as client:
            for index, value in enumerate(values[:4]):
                program = Program()
                tensor = program.upload(id=f"tensor_{index}", kind="tensor", value=value)
                program.return_(key="tensor", value=tensor)
                assert client.execute(program).completed

            program = Program()
            tensors = [
                program.upload(id=f"tensor_{index}", kind="tensor", value=value)
                for index, value in enumerate(values)
            ]
            program.return_(key="tensor", value=tensors[-1])
            outcome = client.execute(program)
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert outcome.completed
    np.testing.assert_array_equal(outcome.results["tensor"], values[-1])


@pytest.mark.parametrize("name", ML_DTYPE_NAMES)
def test_ml_dtype_tensor_round_trips_through_the_server(server_url, name):
    """An ml_dtypes array survives upload, return, and decode with its dtype intact."""
    value = np.array([[1.0, -0.5], [2.0, 4.0]]).astype(getattr(ml_dtypes, name))
    program = Program()
    tensor = program.upload(id="tensor", kind="tensor", value=value)
    program.return_(key="tensor", value=tensor)
    with Client(server_url) as client:
        outcome = client.execute(program)
    result = outcome.results["tensor"]
    assert isinstance(result, np.ndarray) and result.dtype == value.dtype
    np.testing.assert_array_equal(result, value)


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


def test_target_reports_what_a_library_must_be_built_for(server_url):
    with Client(server_url) as client:
        assert client.target() == {"arch": "fake"}
        # a server that reports no target is a protocol violation, not a None result
        client.health = lambda: {"status": "ok", "gpu_count": 1}
        with pytest.raises(ProtocolError, match="no compilation target"):
            client.target()


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


def test_cuda_module_builder_emits_language_and_entry():
    program = Program()
    program.upload(id="kernel", kind="module", source="void add() {}", entry="add", language="cuda")
    assert program.instructions[0]["language"] == "cuda"
    assert program.instructions[0]["entry"] == "add"

    # python is the default and stays off the wire
    program.upload(id="py", kind="module", source="def main():\n    pass\n")
    assert "language" not in program.instructions[1]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"kind": "module", "source": "x", "language": "cuda"}, "must name its 'entry'"),
        ({"kind": "module", "source": "x", "language": "cuda", "entry": "main"}, "reserves 'main'"),
        ({"kind": "module", "source": "x", "language": "rust"}, "'python' or 'cuda'"),
    ],
)
def test_cuda_module_builder_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Program().upload(id="kernel", **kwargs)


def test_library_builder_hashes_bytes_and_carries_entry():
    program = Program()
    program.upload(id="k", kind="library", value=b"\x7fELF...", entry="add_one")
    instruction = program.instructions[0]
    assert instruction["kind"] == "library" and instruction["entry"] == "add_one"
    assert instruction["blob"] == compute_blob_hash(b"\x7fELF...")

    with pytest.raises(ValueError, match="requires an identifier 'entry'"):
        Program().upload(id="k", kind="library", value=b"x")


def test_bytes_builder_hashes_bytes_without_tensor_metadata():
    value = bytearray(b"file contents")
    program = Program()
    program.upload(id="file", kind="bytes", value=value)
    instruction = program.instructions[0]
    assert instruction == {
        "op": "upload",
        "id": "file",
        "kind": "bytes",
        "blob": compute_blob_hash(bytes(value)),
    }
    with pytest.raises(TypeError, match="bytes-like"):
        Program().upload(id="file", kind="bytes", value="text")


def test_numpy_tensor_builder_uses_raw_byte_hash():
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


def test_decoder_covers_every_protocol_dtype():
    """The decoder's dtype table must match the protocol's exactly, both ways."""
    assert set(_NUMPY_DTYPES) == set(DTYPE_ITEM_SIZES)
    for name, numpy_dtype in _NUMPY_DTYPES.items():
        assert numpy_dtype.itemsize == DTYPE_ITEM_SIZES[name]


@pytest.mark.parametrize("dtype", sorted(DTYPE_ITEM_SIZES))
def test_every_protocol_dtype_decodes_to_numpy(dtype):
    shape = [2, 3]
    data = bytes(expected_tensor_nbytes(dtype, shape))
    array = _decode_tensor(dtype, shape, data)
    assert isinstance(array, np.ndarray)
    assert array.dtype.name == dtype and list(array.shape) == shape


@pytest.mark.parametrize("name", ML_DTYPE_NAMES)
def test_ml_dtype_round_trip_is_bit_exact(name):
    """Values ml_dtypes can represent survive a tobytes/decode round trip."""
    original = np.array([1.0, 2.0, -0.5, 4.0]).astype(getattr(ml_dtypes, name))
    decoded = _decode_tensor(name, [4], original.tobytes())
    assert decoded.dtype == original.dtype
    np.testing.assert_array_equal(decoded, original)


@pytest.mark.parametrize("name", ML_DTYPE_NAMES)
def test_ml_dtype_arrays_upload_with_protocol_dtype_names(name):
    """ml_dtypes' dtype names are exactly the protocol's, so uploads need no mapping."""
    value = np.arange(4).astype(getattr(ml_dtypes, name)).reshape(2, 2)
    program = Program()
    program.upload(id="tensor", kind="tensor", value=value)
    instruction = program.instructions[0]
    assert instruction["dtype"] == name and instruction["shape"] == [2, 2]
    assert instruction["blob"] == compute_blob_hash(value.tobytes())


def test_decode_rejects_an_unknown_dtype():
    with pytest.raises(ValueError, match="cannot decode tensor dtype"):
        _decode_tensor("float128", [1], bytes(16))


def test_decoded_tensor_owns_writable_storage():
    """frombuffer would alias the read-only response bytes; the decoder copies."""
    array = _decode_tensor("float32", [2], bytes(8))
    assert array.flags.writeable and array.flags.owndata
    array[0] = 1.5  # must not raise


@pytest.mark.parametrize("shape", [[0], [], [0, 3]])
def test_empty_and_scalar_tensors_decode(shape):
    array = _decode_tensor("float32", shape, bytes(expected_tensor_nbytes("float32", shape)))
    assert list(array.shape) == shape


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
        "lease_wait_ms": 0,
        "lease_held_ms": 1,
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
    from benchmark_server.gpu_runtime import gpu_runtime_factory

    gpu_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    gpu_id = int(gpu_raw) if gpu_raw.isdigit() else 0
    # One worker: spawning the default eight outlasts _start_server's deadline.
    app = create_app(
        ServerConfig(gpus=[gpu_id], workers_per_gpu=1), runtime_factory=gpu_runtime_factory
    )
    server, thread, url = _start_server(app)
    try:
        value = np.arange(64, dtype=np.float32)
        program = Program()
        tensor = program.upload(id="tensor", kind="tensor", value=value)
        program.return_(key="tensor", value=tensor)
        with Client(url) as client:
            outcome = client.execute(program)
        np.testing.assert_array_equal(outcome.results["tensor"], value)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
