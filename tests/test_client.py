"""Client tests against a real uvicorn server over TCP."""

import json
import os
import threading
import time

import httpx
import ml_dtypes
import numpy as np
import pytest
import uvicorn
from support.programs import harness_function

from kcoral import Client, Program, Register
from kcoral.app import create_app
from kcoral.client import (
    _NUMPY_DTYPES,
    KCoralError,
    ProtocolError,
    TransportError,
    _decode_tensor,
    _parse_program_result,
    _response_body,
)
from kcoral.config import ServerConfig
from kcoral.keys import compute_blob_hash
from kcoral.multipart import MultipartPart, encode_multipart, parse_multipart
from kcoral.schemas import DTYPE_ITEM_SIZES, expected_tensor_nbytes
from kcoral.testing import fake_runtime_factory


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
def server_url(tmp_path_factory):
    app = create_app(
        ServerConfig(
            sandbox="none",
            gpus=[0],
            max_requests_per_worker=0,
            disk_cache_dir=tmp_path_factory.mktemp("file-cache"),
        ),
        runtime_factory=fake_runtime_factory,
    )
    server, thread, url = _start_server(app)
    yield url
    server.should_exit = True
    thread.join(timeout=10)


# The three protocol dtypes numpy has no native scalar type for; ml_dtypes
# supplies them, and they are the reason it is a dependency.
ML_DTYPE_NAMES = ("bfloat16", "float8_e4m3fn", "float8_e5m2")


def add_one_program():
    program = Program()
    module = program.upload(id="module", kind="module", source="def main(x):\n    return x + 1\n")
    fn = program.get_function(id="fn", module=module, name="main")
    answer = program.run(id="answer", fn=fn, args=[41])
    program.return_(key="answer", value=answer)
    return program


def test_execute_module_and_return_value(server_url):
    with Client(server_url) as client:
        outcome = client.execute(add_one_program())
    assert outcome.completed and outcome.results == {"answer": 42}
    assert outcome.request_id and outcome.queue_ms >= 0 and outcome.elapsed_ms >= 0


@pytest.mark.parametrize("id_kwargs", [{}, {"id": None}], ids=["omitted", "none"])
def test_execute_with_generated_ids(server_url, id_kwargs):
    program = Program()
    module = program.upload(kind="module", source="def main(x): return x + b'!'", **id_kwargs)
    fn = program.get_function(module=module, name="main", **id_kwargs)
    data = program.upload(kind="bytes", value=b"data", **id_kwargs)
    answer = program.run(fn=fn, args=[data], **id_kwargs)
    program.return_(key="answer", value=answer)

    assert [module.id, fn.id, data.id, answer.id] == [
        "upload_0",
        "get_function_1",
        "upload_2",
        "run_3",
    ]
    with Client(server_url) as client:
        for _ in range(2):
            outcome = client.execute(program)
            assert outcome.completed
            assert outcome["answer"] == b"data!"


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
    inspect_module = program.upload(
        id="inspect_module",
        kind="module",
        source=(
            "def main(data):\n    return {'size': len(data), 'format': data[:11].decode('ascii')}\n"
        ),
    )
    inspect_file = program.get_function(id="inspect_file", module=inspect_module, name="main")
    file_data = program.upload(id="file", kind="bytes", value=value)
    result = program.run(id="result", fn=inspect_file, args=[file_data])
    program.return_(key="file", value=result)
    with Client(server_url) as client:
        first = client.execute(program)
        second = client.execute(program)
    expected = {"file": {"size": len(value), "format": "safetensors"}}
    assert first.results == expected
    assert second.results == expected


def test_file_cache_retry_is_read_from_a_request_local_workspace(server_url):
    value = b"safetensors contents\x00\xff"
    program = Program()
    assert program.upload_file(blob=value, path="./assets//tensor") is None
    inspect_module = program.upload(
        id="inspect_module",
        kind="module",
        source=(
            "import os\n"
            "def main():\n"
            "    data = open('assets/tensor', 'rb').read()\n"
            "    return {'data': data, 'cwd': os.getcwd()}\n"
        ),
    )
    inspect_file = program.get_function(id="inspect_file", module=inspect_module, name="main")
    result = program.run(id="result", fn=inspect_file)
    program.return_(key="file", value=result)

    with Client(server_url) as client:
        first = client.execute(program)
        second = client.execute(program)

    for outcome in (first, second):
        assert outcome.results["file"]["data"] == value
        assert not os.path.exists(outcome.results["file"]["cwd"])
    assert first.results["file"]["cwd"] != second.results["file"]["cwd"]


@pytest.mark.parametrize(
    "cached,evict_after_miss,expected_attempts",
    [
        pytest.param((), False, [((), "CACHE_MISS"), (("a", "b"), "COMPLETED")], id="cold"),
        pytest.param(("a", "b"), False, [((), "COMPLETED")], id="warm"),
        pytest.param(("a",), False, [((), "CACHE_MISS"), (("b",), "COMPLETED")], id="partial"),
        pytest.param(
            ("a",),
            True,
            [((), "CACHE_MISS"), (("b",), "CACHE_MISS"), (("a", "b"), "COMPLETED")],
            id="evicted",
        ),
    ],
)
def test_folder_upload_reuses_identical_wire_program_for_every_attempt(
    tmp_path, cached, evict_after_miss, expected_attempts
):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    a, b = b"first", b"second"
    (source / "a").write_bytes(a)
    (source / "nested" / "b").write_bytes(b)
    (source / "duplicate").write_bytes(a)
    counter = tmp_path / "executions"
    program = Program()
    program.upload_folder(source, path="data")
    # Changing a local file after construction must not change any retry.
    (source / "a").write_bytes(b"changed")
    module = program.upload(
        id="reader_module",
        kind="module",
        source=(
            "from pathlib import Path\n"
            f"with open({str(counter)!r}, 'a') as counter:\n    counter.write('x')\n"
            "def main():\n"
            "    names = ('data/a', 'data/nested/b', 'data/duplicate')\n"
            "    return [Path(name).read_bytes() for name in names]\n"
        ),
    )
    fn = program.get_function(id="reader", module=module, name="main")
    value = program.run(id="value", fn=fn)
    program.return_(key="value", value=value)
    app = create_app(
        ServerConfig(
            sandbox="none",
            gpus=[0],
            workers_per_gpu=1,
            max_requests_per_worker=0,
            disk_cache_dir=tmp_path / "cache",
        ),
        runtime_factory=fake_runtime_factory,
    )
    server, thread, url = _start_server(app)
    blobs = {"a": a, "b": b}
    keys = {name: compute_blob_hash(data) for name, data in blobs.items()}
    requests = []
    statuses = []

    def record_request(request):
        parts = parse_multipart(request.headers["content-type"], request.read())
        requests.append(
            (
                next(part.data for part in parts if part.name == "program"),
                {
                    part.name.removeprefix("blob:")
                    for part in parts
                    if part.name.startswith("blob:")
                },
            )
        )

    def record_response(response):
        response.read()
        # Completed responses contain binary results; misses are plain JSON.
        if response.headers["content-type"].startswith("application/json"):
            statuses.append(response.json()["status"])
        else:
            statuses.append("COMPLETED")
        if evict_after_miss and len(statuses) == 1:
            assert statuses == ["CACHE_MISS"]
            assert response.json()["missing_blobs"] == [keys["b"]]
            # The server asked only for b. Evict cached a before that retry so
            # sending b alone misses again and forces the client's full resend.
            key = keys["a"]
            (app.state.file_cache.directory / key[:2] / key).unlink()
            assert app.state.file_cache.get(key) is None

    try:
        for name in cached:
            app.state.file_cache.put(keys[name], blobs[name])
        with Client(url) as client:
            client._http.event_hooks = {"request": [record_request], "response": [record_response]}
            result = client.execute(program)
        assert result.completed and result.results == {"value": [a, b, a]}
        assert counter.read_text() == "x"
        assert [(parts, status) for (_, parts), status in zip(requests, statuses, strict=True)] == [
            ({keys[name] for name in names}, status) for names, status in expected_attempts
        ]
        assert all(payload == requests[0][0] for payload, _ in requests)
        assert json.loads(requests[0][0])["instructions"] == program.instructions
        assert all(app.state.cache.get(key) is None for key in keys.values())
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_cache_miss_retry_returns_the_router_affinity_header():
    program = Program()
    program.upload(id="file", kind="bytes", value=b"contents")
    blob_hash = program.instructions[0]["blob"]
    seen_routes = []

    def handler(request):
        seen_routes.append(request.headers.get("x-kcoral-node"))
        if len(seen_routes) == 1:
            return httpx.Response(
                200,
                json={"status": "CACHE_MISS", "missing_blobs": [blob_hash]},
                headers={"X-KCoral-Node": "node-affinity"},
            )
        return httpx.Response(
            200,
            json={
                "status": "COMPLETED",
                "request_id": "request",
                "queue_ms": 0,
                "elapsed_ms": 1,
                "lease_wait_ms": 0,
                "lease_held_ms": 1,
                "stdout": "",
                "stderr": "",
                "results": {},
            },
        )

    with Client("http://router") as client:
        client._http.close()
        client._http = httpx.Client(
            base_url="http://router", transport=httpx.MockTransport(handler)
        )
        outcome = client.execute(program)

    assert outcome.completed
    assert seen_routes == [None, "node-affinity"]


def test_cache_churn_falls_back_to_all_blobs():
    app = create_app(
        ServerConfig(sandbox="none", gpus=[0], workers_per_gpu=1, cache_capacity_bytes=16),
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
    module = program.upload(
        id="module",
        kind="module",
        source="def main():\n    return [b'a', {'nested': b'b'}]\n",
    )
    fn = program.get_function(id="fn", module=module, name="main")
    value = program.run(id="value", fn=fn)
    program.return_(key="value", value=value)
    with Client(server_url) as client:
        outcome = client.execute(program)
    assert outcome.results == {"value": [b"a", {"nested": b"b"}]}


def test_failed_instruction_is_data(server_url):
    program = Program()
    program.run(id="bad", fn=harness_function(program, "nope", "bad"))
    with Client(server_url) as client:
        outcome = client.execute(program)
    assert outcome.status == "FAILED" and outcome.results == {}
    assert outcome.error["kind"] == "runtime"


def test_a_cpu_only_function_touching_the_gpu_fails_with_the_call_named(server_url):
    program = Program()
    module = program.upload(
        id="module",
        kind="module",
        source="from kcoral.testing import simulate_cuda_call\n\n"
        "def main():\n"
        "    simulate_cuda_call('cudaMalloc')\n",
    )
    fn = program.get_function(id="fn", module=module, name="main", cpu_only=True)
    program.run(id="call", fn=fn)
    with Client(server_url) as client:
        outcome = client.execute(program)
    assert outcome.status == "FAILED"
    assert outcome.error["kind"] == "gpu_access" and outcome.error["instruction_id"] == "call"
    assert outcome.error["cuda_call"] == "cudaMalloc"
    assert outcome.error["location"] == "<uploaded>:1 in main"
    assert outcome.error["interfered_request_id"] is None  # nobody else held the GPU


def test_interleaved_return_survives_a_later_failure(server_url):
    program = Program()
    module = program.upload(
        id="module", kind="module", source="def main():\n    return b'checkpoint'\n"
    )
    fn = program.get_function(id="fn", module=module, name="main")
    early = program.run(id="early", fn=fn)
    program.return_(key="early", value=early)  # checkpointed before the failure
    program.run(id="bad", fn=harness_function(program, "nope", "bad"))
    with Client(server_url) as client:
        outcome = client.execute(program)
    assert outcome.status == "FAILED"
    assert outcome.results == {"early": b"checkpoint"}
    assert outcome.error["instruction_op"] == "run" and outcome.error["instruction_id"] == "bad"
    assert "Traceback" in outcome.error["traceback"]


def test_timeout_raises_server_error(server_url):
    program = Program()
    program.run(id="sleep", fn=harness_function(program, "sleep", "sleep"), args=[5])
    with Client(server_url) as client:
        with pytest.raises(KCoralError) as exc_info:
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
    program.run(id="later", fn=harness_function(program, "structural", "later"))
    program.return_(key="later", value=Register("later"))
    assert len(program.instructions) == 6

    with pytest.raises(ValueError, match="unknown handle"):
        Program().return_(key="missing", value=Register("nope"))


def test_program_builder_validates_ids_and_tensor_metadata():
    with pytest.raises(ValueError, match="expects 4 bytes"):
        Program().upload(id="bad", kind="tensor", value=b"abc", dtype="float32", shape=[1])

    reusable = Program()
    with pytest.raises(ValueError, match="expects 4 bytes"):
        reusable.upload(id="tensor", kind="tensor", value=b"abc", dtype="float32", shape=[1])
    reusable.upload(id="tensor", kind="tensor", value=b"\x00" * 4, dtype="float32", shape=[1])


@pytest.mark.parametrize("id_kwargs", [{}, {"id": None}], ids=["omitted", "none"])
@pytest.mark.parametrize(
    "upload_kwargs",
    [
        {"kind": "module", "source": ""},
        {"kind": "module", "source": "", "language": "cuda"},
        {"kind": "tensor", "value": b"\x00" * 4, "dtype": "float32", "shape": [1]},
        {"kind": "bytes", "value": b"data"},
        {"kind": "library", "value": b"library"},
    ],
    ids=["python", "cuda", "tensor", "bytes", "library"],
)
def test_upload_generates_ids_for_each_kind(id_kwargs, upload_kwargs):
    program = Program()
    register = program.upload(**upload_kwargs, **id_kwargs)
    assert register == Register("upload_0")
    assert program.instructions[0]["id"] == register.id


def test_generated_ids_skip_explicit_ids_and_ignore_instructions_without_ids():
    program = Program()
    module = program.upload(id="module", kind="module", source="def main(): return 42")
    for reserved_id in ("upload_0", "get_function_2", "run_4"):
        program.upload(id=reserved_id, kind="bytes", value=b"reserved")
    assert program.upload(kind="bytes", value=b"data").id == "upload_1"
    program.upload_file(blob=b"data", path="input.txt")
    fn = program.get_function(module=module, name="main")
    assert fn.id == "get_function_3"
    answer = program.run(fn=fn)
    assert answer.id == "run_5"
    program.return_(key="answer", value=answer)
    program.return_file(key="input", path="input.txt")
    assert program.run(fn=fn).id == "run_6"
    assert Program().upload(kind="bytes", value=b"fresh").id == "upload_0"


@pytest.mark.parametrize("op", ["upload", "get_function", "run"])
@pytest.mark.parametrize("id_kwargs", [{"id": "module"}, {}], ids=["explicit", "generated"])
def test_explicit_ids_cannot_duplicate_existing_ids(op, id_kwargs):
    program = Program()
    module = program.upload(kind="module", source="def main(): pass", **id_kwargs)
    fn = program.get_function(id="fn", module=module, name="main")
    kwargs = {
        "upload": {"kind": "module", "source": ""},
        "get_function": {"module": module, "name": "main"},
        "run": {"fn": fn},
    }[op]
    before = program.instructions
    with pytest.raises(ValueError, match="duplicate instruction id"):
        getattr(program, op)(id=module.id, **kwargs)
    assert program.instructions == before


@pytest.mark.parametrize("op", ["upload", "get_function", "run"])
@pytest.mark.parametrize("invalid_id", ["", 0, False])
def test_explicit_ids_must_be_nonempty_strings(op, invalid_id):
    program = Program()
    module = program.upload(kind="module", source="def main(): pass")
    fn = program.get_function(module=module, name="main")
    kwargs = {
        "upload": {"kind": "module", "source": ""},
        "get_function": {"module": module, "name": "main"},
        "run": {"fn": fn},
    }[op]
    with pytest.raises(ValueError, match="instruction id must be a non-empty string"):
        getattr(program, op)(id=invalid_id, **kwargs)


def test_failed_builder_validation_does_not_consume_generated_ids():
    program = Program()
    with pytest.raises(TypeError, match="requires string"):
        program.upload(kind="module")
    module = program.upload(kind="module", source="def main(): pass")
    assert module.id == "upload_0"
    with pytest.raises(ValueError, match="non-empty string"):
        program.get_function(module=module, name="")
    fn = program.get_function(module=module, name="main")
    assert fn.id == "get_function_1"
    with pytest.raises(ValueError, match="unknown handle"):
        program.run(fn=Register("missing"))
    assert program.run(fn=fn).id == "run_2"


def test_cuda_module_builder_emits_language_and_get_function():
    program = Program()
    module = program.upload(id="kernel", kind="module", source="void add() {}", language="cuda")
    program.get_function(id="add", module=module, name="add")
    assert program.instructions[0]["language"] == "cuda"
    assert "entry" not in program.instructions[0]
    assert program.instructions[1] == {
        "op": "get_function",
        "id": "add",
        "module": {"$ref": "kernel"},
        "name": "add",
    }

    # python is the default and stays off the wire
    program.upload(id="py", kind="module", source="def main():\n    pass\n")
    assert "language" not in program.instructions[2]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"kind": "module", "source": "x", "language": "rust"}, "'python' or 'cuda'"),
    ],
)
def test_cuda_module_builder_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Program().upload(id="kernel", **kwargs)


def test_library_builder_hashes_bytes_and_gets_functions():
    program = Program()
    program.upload(id="k", kind="library", value=b"\x7fELF...")
    instruction = program.instructions[0]
    assert instruction["kind"] == "library" and "entry" not in instruction
    assert instruction["blob"] == compute_blob_hash(b"\x7fELF...")

    modules = Program()
    module = modules.upload(id="module", kind="library", value=b"x")
    function = modules.get_function(id="step", module=module, name="namespace.step")
    assert function == Register("step")
    assert modules.instructions == [
        {
            "op": "upload",
            "id": "module",
            "kind": "library",
            "blob": compute_blob_hash(b"x"),
        },
        {
            "op": "get_function",
            "id": "step",
            "module": {"$ref": "module"},
            "name": "namespace.step",
        },
    ]


def test_upload_has_no_entry_argument():
    with pytest.raises(TypeError, match="unexpected keyword argument 'entry'"):
        Program().upload(id="module", kind="module", source="", entry="main")


def test_get_function_builder_validates_its_module_and_name():
    program = Program()
    module = program.upload(id="module", kind="library", value=b"library")
    with pytest.raises(ValueError, match="unknown handle"):
        program.get_function(id="bad", module=Register("missing"), name="step")
    with pytest.raises(ValueError, match="non-empty string"):
        program.get_function(id="bad", module=module, name="")
    with pytest.raises(TypeError, match="module must be"):
        program.get_function(id="bad", module={"not": "a ref"}, name="step")
    with pytest.raises(TypeError, match="cpu_only must be a bool"):
        program.get_function(id="bad", module=module, name="step", cpu_only="yes")


def test_get_function_builder_emits_cpu_only_when_declared():
    program = Program()
    module = program.upload(id="module", kind="module", source="def main():\n    pass\n")
    program.get_function(id="device", module=module, name="main")
    program.get_function(id="host", module=module, name="main", cpu_only=True)
    assert "cpu_only" not in program.instructions[1]  # the default stays off the wire
    assert program.instructions[2] == {
        "op": "get_function",
        "id": "host",
        "module": {"$ref": "module"},
        "name": "main",
        "cpu_only": True,
    }


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


def test_file_builder_snapshots_blob_without_creating_a_register():
    value = bytearray(b"file contents")
    program = Program()
    assert program.upload_file(blob=value, path="./data//tensor") is None
    assert program.instructions == [
        {
            "op": "upload",
            "kind": "file",
            "blob": compute_blob_hash(bytes(value)),
            "path": "data/tensor",
        }
    ]
    value[:] = b"changed"
    assert program._blobs == {compute_blob_hash(b"file contents"): b"file contents"}
    assert program._ids == set()

    with pytest.raises(ValueError, match="must be relative"):
        Program().upload_file(blob=b"x", path="/tmp/tensor")
    with pytest.raises(ValueError, match=r"'\.\.' component"):
        Program().upload_file(blob=b"x", path="data/../tensor")
    with pytest.raises(TypeError, match="unexpected keyword argument 'id'"):
        Program().upload_file(id="file", blob=b"x", path="tensor")
    with pytest.raises(TypeError, match="bytes-like"):
        Program().upload_file(blob="text", path="tensor")


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
    os.environ.get("KCORAL_GPU_TEST") != "1",
    reason="real tensor client test requires KCORAL_GPU_TEST=1",
)
def test_tensor_round_trip_on_gpu():
    from kcoral.gpu_runtime import gpu_runtime_factory

    gpu_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    gpu_id = int(gpu_raw) if gpu_raw.isdigit() else 0
    # One worker: spawning the default eight outlasts _start_server's deadline.
    app = create_app(
        ServerConfig(sandbox="none", gpus=[gpu_id], workers_per_gpu=1),
        runtime_factory=gpu_runtime_factory,
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


@pytest.mark.parametrize("fail_after_return", [None, "return", "run"])
def test_file_folder_returns_survive_workspace_cleanup(server_url, tmp_path, fail_after_return):
    from pathlib import Path

    p = Program()
    p.upload_file(blob=b"input", path="input.txt")
    module = p.upload(
        id="module",
        kind="module",
        source="""
from pathlib import Path
import os
def make():
    Path("out/nested/empty").mkdir(parents=True)
    Path("out/.hidden").write_bytes(b"")
    Path("out/nested/report").write_bytes(Path("input.txt").read_bytes() + b" report")
    return os.getcwd()
def report_path():
    return "out/nested/report"
""",
    )
    make = p.get_function(id="make", module=module, name="make", cpu_only=True)
    get_path = p.get_function(id="get_path", module=module, name="report_path", cpu_only=True)
    workspace = p.run(id="workspace", fn=make)
    path = p.run(id="path", fn=get_path)
    p.return_(key="workspace", value=workspace)
    p.return_file(key="report", path=path)
    p.return_folder(key="outputs", path="out")
    if fail_after_return == "return":
        p.return_file(key="missing", path="missing")
    elif fail_after_return == "run":
        p.run(id="fail", fn=harness_function(p, "missing", "fail"))
    with Client(server_url) as client:
        result = client.execute(p)
    assert result.completed == (fail_after_return is None)
    assert not Path(result["workspace"]).exists()
    result["report"].save(tmp_path / "report")
    result["outputs"].save(tmp_path / "outputs")
    assert (tmp_path / "report").read_bytes() == b"input report"
    assert (tmp_path / "outputs/nested/report").read_bytes() == b"input report"
    assert (tmp_path / "outputs/nested/empty").is_dir()
    assert (tmp_path / "outputs/.hidden").read_bytes() == b""
