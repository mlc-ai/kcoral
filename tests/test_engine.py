import pytest

from benchmark_server.engine import execute
from benchmark_server.keys import compute_blob_hash
from benchmark_server.schemas import Program, Ref, Return, Run, Upload
from benchmark_server.testing import UNSHARED_GPU, FakeRuntime


def ref(handle):
    return Ref(handle)


def call_module(source, entry=None):
    """Upload ``source``, call whatever object the handle bound, and return the outcome."""
    return execute(
        Program(
            [
                Upload("fn", "module", source=source, entry=entry),
                Run("answer", ref("fn"), [41]),
                Return("value", ref("answer")),
            ]
        ),
        FakeRuntime(),
        UNSHARED_GPU,
    )


@pytest.mark.parametrize(
    "source,entry",
    [
        # The sole top-level definition is the entry, whatever it is named.
        ("def matmul(x):\n    return x + 1\n", None),
        # A module-level constant is not a definition, so it does not compete.
        ("BLOCK = 127\n\ndef matmul(x):\n    return x + BLOCK - 126\n", None),
        # ``main`` still wins when the source defines several names.
        ("def helper(x):\n    return 0\n\ndef main(x):\n    return x + 1\n", None),
        # An explicit ``entry`` outranks ``main``.
        ("def main(x):\n    return 0\n\ndef matmul(x):\n    return x + 1\n", "matmul"),
    ],
)
def test_module_entry_resolution(source, entry):
    outcome = call_module(source, entry)
    assert outcome.status == "COMPLETED"
    assert outcome.results == {"value": {"type": "integer", "value": 42}}


@pytest.mark.parametrize(
    "source,entry,message",
    [
        (
            "def helper(x):\n    return 0\n\ndef matmul(x):\n    return x\n",
            None,
            "top-level names 'helper', 'matmul'",
        ),
        ("BLOCK = 128\n", None, "no top-level function or class"),
        ("def matmul(x):\n    return x\n", "typo", "does not define 'typo'"),
    ],
)
def test_unresolvable_module_entry_fails_the_upload(source, entry, message):
    outcome = call_module(source, entry)
    assert outcome.status == "FAILED"
    assert outcome.error["kind"] == "parse"
    assert outcome.error["instruction_op"] == "upload"
    assert message in outcome.error["message"]


def test_unreturned_values_are_not_serialized():
    outcome = execute(Program([Run("opaque", "builtin.opaque", [])]), FakeRuntime(), UNSHARED_GPU)
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
        UNSHARED_GPU,
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
    outcome = execute(program, FakeRuntime(), UNSHARED_GPU)
    assert outcome.results["tensor"] == {
        "type": "tensor",
        "dtype": "float32",
        "shape": [1],
        "part": "return:0",
        "sha256": digest,
    }
    assert outcome.binary_parts == {"return:0": raw}


def test_bytes_upload_is_available_to_later_instructions_and_returns():
    raw = b"file contents\x00\xff"
    digest = compute_blob_hash(raw)
    program = Program(
        [
            Upload("file", "bytes", blob=digest),
            Return("file", ref("file")),
        ],
        blob_bytes={digest: raw},
    )
    outcome = execute(program, FakeRuntime(), UNSHARED_GPU)
    assert outcome.results["file"] == {
        "type": "bytes",
        "part": "return:0",
        "sha256": digest,
    }
    assert outcome.binary_parts == {"return:0": raw}


def test_instruction_failure_stops_and_describes_the_instruction():
    outcome = execute(
        Program(
            [
                Run("ok", "builtin.structural", []),
                Run("bad", "builtin.nope", []),
                Return("ok", ref("ok")),
            ]
        ),
        FakeRuntime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED" and outcome.results == {}
    error = outcome.error
    assert error["kind"] == "runtime" and error["instruction_index"] == 1
    assert error["instruction_op"] == "run" and error["instruction_id"] == "bad"
    assert "Traceback" in error["traceback"]


def test_returns_that_ran_survive_a_later_failure():
    outcome = execute(
        Program(
            [
                Run("ok", "builtin.structural", []),
                Return("early", ref("ok")),
                Run("bad", "builtin.nope", []),
                Return("late", ref("ok")),
            ]
        ),
        FakeRuntime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED"
    assert set(outcome.results) == {"early"}
    assert outcome.results["early"]["value"]["ok"] == {"type": "boolean", "value": True}


def test_failed_return_rolls_back_only_its_own_binary_parts():
    # ``main`` encodes one part before hitting a value the encoder cannot handle.
    source = "def main():\n    return [b'partial', object()]\n"
    outcome = execute(
        Program(
            [
                Run("blob", "builtin.binary", []),
                Return("kept", ref("blob")),
                Upload("fn", "module", source=source),
                Run("mixed", ref("fn"), []),
                Return("dropped", ref("mixed")),
            ]
        ),
        FakeRuntime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED"
    assert set(outcome.results) == {"kept"}
    # Without rollback the aborted return would leave an orphan 'return:1'.
    assert outcome.binary_parts == {"return:0": b"binary-result"}
    error = outcome.error
    assert error["kind"] == "serialization" and error["instruction_index"] == 4
    assert error["instruction_op"] == "return" and error["instruction_id"] is None


def test_unsupported_return_is_serialization_failure():
    outcome = execute(
        Program([Run("opaque", "builtin.opaque", []), Return("value", ref("opaque"))]),
        FakeRuntime(),
        UNSHARED_GPU,
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
        UNSHARED_GPU,
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
        UNSHARED_GPU,
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
    outcome = execute(program, FakeRuntime(), UNSHARED_GPU)
    assert outcome.stdout == "12345\n6" and outcome.stdout_truncated


class RecordingLease:
    def __init__(self) -> None:
        self.held = False
        self.acquires = 0

    def acquire(self) -> None:
        if not self.held:
            self.acquires += 1
        self.held = True

    def release(self) -> None:
        self.held = False


class CudaAwareRuntime(FakeRuntime):
    """The fake runtime has no compiler; the real one binds CUDA source text."""

    def load_module(self, source, entry=None, language="python"):
        if language == "cuda":
            return object()
        return super().load_module(source, entry, language)


@pytest.mark.parametrize(
    "language,source,entry,acquires",
    [
        # A CUDA upload runs nothing, so it never waits for a GPU.
        ("cuda", "void go() {}", "go", 0),
        # A Python upload execs the client's source, which could touch one.
        ("python", "def main(x):\n    return x\n", None, 1),
    ],
)
def test_which_module_uploads_take_the_gpu(language, source, entry, acquires):
    lease = RecordingLease()
    program = Program([Upload("k", "module", source=source, entry=entry, language=language)])
    outcome = execute(program, CudaAwareRuntime(), lease)
    assert outcome.status == "COMPLETED"
    assert lease.acquires == acquires
