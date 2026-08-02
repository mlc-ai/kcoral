from benchmark_server.engine import execute
from benchmark_server.keys import compute_blob_hash
from benchmark_server.schemas import Program, Return, Run, Upload
from benchmark_server.testing import FakeRuntime


def ref(handle):
    return {"$ref": handle}


def test_module_run_and_explicit_return():
    outcome = execute(
        Program(
            [
                Upload("fn", "module", source="def main(x):\n    return x + 1\n"),
                Run("answer", ref("fn"), [41]),
                Return("value", ref("answer")),
            ]
        ),
        FakeRuntime(),
    )
    assert outcome.status == "COMPLETED"
    assert outcome.results == {"value": {"type": "integer", "value": 42}}


def test_unreturned_values_are_not_serialized():
    outcome = execute(Program([Run("opaque", "builtin.opaque", [])]), FakeRuntime())
    assert outcome.status == "COMPLETED" and outcome.results == {}


def test_recursive_values_and_depth_first_binary_parts():
    source = "def main():\n    return [b'a', {'nested': b'b'}]\n"
    outcome = execute(
        Program(
            [
                Upload("fn", "module", source=source),
                Run("nested", ref("fn"), []),
                Return("nested", ref("nested")),
            ]
        ),
        FakeRuntime(),
    )
    encoded = outcome.results["nested"]
    assert encoded["type"] == "array"
    assert encoded["value"][0]["part"] == "return:0"
    assert encoded["value"][1]["value"]["nested"]["part"] == "return:1"
    assert outcome.binary_parts == {"return:0": b"a", "return:1": b"b"}


def test_tensor_return_has_metadata_hash_and_binary_part():
    raw = b"\x00\x00\x80?"
    digest = compute_blob_hash(raw)
    program = Program(
        [
            Upload("tensor", "tensor", blob=digest, dtype="float32", shape=[1]),
            Return("tensor", ref("tensor")),
        ],
        blob_bytes={digest: raw},
    )
    outcome = execute(program, FakeRuntime())
    assert outcome.results["tensor"] == {
        "type": "tensor",
        "dtype": "float32",
        "shape": [1],
        "part": "return:0",
        "sha256": digest,
    }
    assert outcome.binary_parts == {"return:0": raw}


def test_instruction_failure_stops_and_returns_no_results():
    outcome = execute(
        Program(
            [
                Run("ok", "builtin.structural", []),
                Run("bad", "builtin.nope", []),
                Return("ok", ref("ok")),
            ]
        ),
        FakeRuntime(),
    )
    assert outcome.status == "FAILED" and outcome.results == {}
    assert outcome.error["kind"] == "runtime" and outcome.error["instruction_index"] == 1


def test_unsupported_return_is_serialization_failure():
    outcome = execute(
        Program([Run("opaque", "builtin.opaque", []), Return("value", ref("opaque"))]),
        FakeRuntime(),
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "serialization"
    assert outcome.error["instruction_index"] == 1


def test_invalid_exported_tensor_is_serialization_failure():
    class InvalidTensorRuntime(FakeRuntime):
        def export_tensor(self, value):
            return "float32", [2], b"short"

    outcome = execute(
        Program([Run("opaque", "builtin.opaque", []), Return("value", ref("opaque"))]),
        InvalidTensorRuntime(),
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "serialization"


def test_stdout_and_stderr_are_captured_for_the_request():
    source = (
        "import os, sys\n"
        "print('module loaded', flush=True)\n"
        "def main():\n"
        "    os.write(1, b'fd output\\n')\n"
        "    print('error output', file=sys.stderr)\n"
        "    return 1\n"
    )
    outcome = execute(
        Program(
            [
                Upload("fn", "module", source=source),
                Run("value", ref("fn"), []),
                Return("value", ref("value")),
            ]
        ),
        FakeRuntime(),
    )
    assert outcome.stdout == "module loaded\nfd output\n"
    assert outcome.stderr == "error output\n"


def test_output_limit_is_shared_across_the_request():
    program = Program(
        [
            Upload("fn", "module", source="print('12345')\ndef main():\n    print('67890')\n"),
            Run("value", ref("fn"), []),
        ],
        options={"output_limit_bytes": 7},
    )
    outcome = execute(program, FakeRuntime())
    assert outcome.stdout == "12345\n6" and outcome.stdout_truncated
