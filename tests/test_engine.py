import pytest
from support.programs import harness_call

from kcoral.engine import execute
from kcoral.keys import compute_blob_hash
from kcoral.schemas import FileUpload, GetFunction, Program, Ref, Return, Run, Upload
from kcoral.testing import UNSHARED_GPU, FakeRuntime, execute_for_test


def ref(handle):
    return Ref(handle)


def call_module(source, name):
    """Upload ``source``, select ``name``, call it, and return the outcome."""
    return execute_for_test(
        Program(
            [
                Upload("module", "module", source=source),
                GetFunction("fn", ref("module"), name),
                Run("answer", ref("fn"), [41]),
                Return("value", ref("answer")),
            ]
        ),
        FakeRuntime(),
        UNSHARED_GPU,
    )


def test_library_module_binds_multiple_functions_and_chains_results():
    digest = compute_blob_hash(b"fake-library")

    class LibraryRuntime(FakeRuntime):
        def load_library(self, data):
            assert data == b"fake-library"
            return {"add_one": lambda value: value + 1, "times_two": lambda value: value * 2}

    outcome = execute_for_test(
        Program(
            [
                Upload("module", "library", blob=digest),
                GetFunction("add_one", ref("module"), "add_one"),
                GetFunction("times_two", ref("module"), "times_two"),
                Run("incremented", ref("add_one"), [20]),
                Run("answer", ref("times_two"), [ref("incremented")]),
                Return("answer", ref("answer")),
            ],
            blob_bytes={digest: b"fake-library"},
        ),
        LibraryRuntime(),
        UNSHARED_GPU,
    )

    assert outcome.status == "COMPLETED"
    assert outcome.results == {"answer": {"type": "integer", "value": 42}}


def test_missing_library_function_is_attributed_to_get_function():
    digest = compute_blob_hash(b"fake-library")

    class LibraryRuntime(FakeRuntime):
        def load_library(self, data):
            return {}

    outcome = execute_for_test(
        Program(
            [
                Upload("module", "library", blob=digest),
                GetFunction("missing", ref("module"), "missing"),
            ],
            blob_bytes={digest: b"fake-library"},
        ),
        LibraryRuntime(),
        UNSHARED_GPU,
    )

    assert outcome.status == "FAILED"
    assert outcome.error["kind"] == "compile"
    assert outcome.error["instruction_op"] == "get_function"
    assert outcome.error["instruction_id"] == "missing"


def test_cleanup_failure_preserves_runtime_error_and_marks_worker_unhealthy():
    runtime = FakeRuntime()
    cleanup_errors = []
    outcome = execute_for_test(
        Program([*harness_call("bad", "poison", [])]),
        runtime,
        UNSHARED_GPU,
        cleanup_failed=cleanup_errors.append,
    )
    assert outcome.status == "FAILED"
    assert outcome.error["kind"] == "runtime"
    assert outcome.error["message"] == "simulated illegal memory access"
    assert len(cleanup_errors) == 1
    assert str(cleanup_errors[0]) == "simulated poisoned GPU context"


def test_cuda_last_error_is_attributed_to_the_current_request_without_poisoning_cleanup():
    runtime = FakeRuntime()
    cleanup_errors = []
    outcome = execute_for_test(
        Program([*harness_call("bad", "stale_cuda_error", [])]),
        runtime,
        UNSHARED_GPU,
        cleanup_failed=cleanup_errors.append,
    )

    assert outcome.status == "FAILED"
    assert outcome.error["kind"] == "runtime"
    assert outcome.error["message"] == "CUDA error cudaErrorInvalidValue (1): invalid argument"
    assert outcome.error["instruction_index"] == 2
    assert outcome.error["instruction_id"] == "bad"
    assert cleanup_errors == []
    assert runtime.take_last_error() is None


def test_cuda_last_error_overrides_cupti_unavailable_error():
    outcome = execute_for_test(
        Program([*harness_call("bad", "stale_cuda_error_unavailable", [])]),
        FakeRuntime(),
        UNSHARED_GPU,
    )

    assert outcome.status == "FAILED"
    assert outcome.error["kind"] == "runtime"
    assert outcome.error["message"] == "CUDA error cudaErrorInvalidValue (1): invalid argument"


@pytest.mark.parametrize(
    "source,name",
    [
        ("def matmul(x):\n    return x + 1\n", "matmul"),
        ("BLOCK = 127\n\ndef matmul(x):\n    return x + BLOCK - 126\n", "matmul"),
        ("def helper(x):\n    return 0\n\ndef main(x):\n    return x + 1\n", "main"),
        ("def main(x):\n    return 0\n\ndef matmul(x):\n    return x + 1\n", "matmul"),
    ],
)
def test_python_module_function_selection(source, name):
    outcome = call_module(source, name)
    assert outcome.status == "COMPLETED"
    assert outcome.results == {"value": {"type": "integer", "value": 42}}


@pytest.mark.parametrize(
    "source,name",
    [
        ("def helper(x):\n    return 0\n\ndef matmul(x):\n    return x\n", "typo"),
        ("BLOCK = 128\n", "main"),
        ("def matmul(x):\n    return x\n", "main"),
    ],
)
def test_missing_python_function_fails_get_function(source, name):
    outcome = call_module(source, name)
    assert outcome.status == "FAILED"
    assert outcome.error["kind"] == "parse"
    assert outcome.error["instruction_op"] == "get_function"
    assert outcome.error["instruction_id"] == "fn"
    assert f"defines no name {name!r}" in outcome.error["message"]


def test_unreturned_values_are_not_serialized():
    outcome = execute_for_test(
        Program([*harness_call("opaque", "opaque", [])]), FakeRuntime(), UNSHARED_GPU
    )
    assert outcome.status == "COMPLETED" and outcome.results == {}


def test_recursive_values_and_depth_first_binary_parts():
    source = "def main():\n    return [b'a', {'nested': b'b'}]\n"
    outcome = execute_for_test(
        Program(
            [
                Upload("module", "module", source=source),
                GetFunction("fn", ref("module"), "main"),
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
    outcome = execute_for_test(program, FakeRuntime(), UNSHARED_GPU)
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
    outcome = execute_for_test(program, FakeRuntime(), UNSHARED_GPU)
    assert outcome.results["file"] == {
        "type": "bytes",
        "part": "return:0",
        "sha256": digest,
    }
    assert outcome.binary_parts == {"return:0": raw}


def test_instruction_failure_stops_and_describes_the_instruction():
    outcome = execute_for_test(
        Program(
            [
                *harness_call("ok", "structural", []),
                *harness_call("bad", "nope", []),
                Return("ok", ref("ok")),
            ]
        ),
        FakeRuntime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED" and outcome.results == {}
    error = outcome.error
    assert error["kind"] == "runtime" and error["instruction_index"] == 5
    assert error["instruction_op"] == "run" and error["instruction_id"] == "bad"
    assert "Traceback" in error["traceback"]


def test_returns_that_ran_survive_a_later_failure():
    outcome = execute_for_test(
        Program(
            [
                *harness_call("ok", "structural", []),
                Return("early", ref("ok")),
                *harness_call("bad", "nope", []),
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
    outcome = execute_for_test(
        Program(
            [
                *harness_call("blob", "binary", []),
                Return("kept", ref("blob")),
                Upload("module", "module", source=source),
                GetFunction("fn", ref("module"), "main"),
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
    assert error["kind"] == "serialization" and error["instruction_index"] == 7
    assert error["instruction_op"] == "return" and error["instruction_id"] is None


def test_unsupported_return_is_serialization_failure():
    outcome = execute_for_test(
        Program([*harness_call("opaque", "opaque", []), Return("value", ref("opaque"))]),
        FakeRuntime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "serialization"
    assert outcome.error["instruction_index"] == 3


def test_invalid_exported_tensor_is_serialization_failure():
    class InvalidTensorRuntime(FakeRuntime):
        def export_tensor(self, value):
            return "float32", [2], b"short"

    outcome = execute_for_test(
        Program([*harness_call("opaque", "opaque", []), Return("value", ref("opaque"))]),
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
    outcome = execute_for_test(
        Program(
            [
                Upload("module", "module", source=source),
                GetFunction("fn", ref("module"), "main"),
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
            Upload("module", "module", source="print('12345')\ndef main():\n    print('67890')\n"),
            GetFunction("fn", ref("module"), "main"),
            Run("value", ref("fn"), []),
        ],
        options={"output_limit_bytes": 7},
    )
    outcome = execute_for_test(program, FakeRuntime(), UNSHARED_GPU)
    assert outcome.stdout == "12345\n6" and outcome.stdout_truncated


class RecordingLease:
    def __init__(self) -> None:
        self.held = False
        self.acquires = 0
        self.releases = 0

    def acquire(self) -> None:
        if not self.held:
            self.acquires += 1
        self.held = True

    def release(self) -> None:
        if self.held:
            self.releases += 1
        self.held = False


def test_file_upload_copies_nested_file_without_taking_the_gpu(tmp_path):
    raw = b"tensor contents\x00\xff"
    digest = compute_blob_hash(raw)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class CleanupLease(RecordingLease):
        def acquire(self):
            # File staging has already finished when final cleanup takes the GPU.
            assert (workspace / "nested/tensor").read_bytes() == raw
            super().acquire()

    lease = CleanupLease()

    outcome = execute(
        Program(
            [FileUpload(id="file", blob=digest, path="nested/tensor")], blob_bytes={digest: raw}
        ),
        FakeRuntime(),
        lease,
        workspace_dir=str(workspace),
    )

    assert outcome.status == "COMPLETED"
    assert (workspace / "nested/tensor").read_bytes() == raw
    assert lease.acquires == 1 and lease.releases == 0  # retain final cleanup ownership


def test_uploaded_module_reads_file_relative_to_request_workspace():
    raw = b"script input"
    digest = compute_blob_hash(raw)
    source = "DATA = open('input/data.bin', 'rb').read()\ndef main():\n    return DATA\n"
    outcome = execute_for_test(
        Program(
            [
                FileUpload(id="file", blob=digest, path="input/data.bin"),
                Upload("module", "module", source=source),
                GetFunction("fn", ref("module"), "main"),
                Run("value", ref("fn"), []),
                Return("value", ref("value")),
            ],
            blob_bytes={digest: raw},
        ),
        FakeRuntime(),
        UNSHARED_GPU,
    )

    assert outcome.status == "COMPLETED"
    part = outcome.results["value"]["part"]
    assert outcome.binary_parts[part] == raw


def test_file_upload_does_not_follow_workspace_symlink(tmp_path):
    raw = b"must stay contained"
    digest = compute_blob_hash(raw)
    outside = tmp_path / "outside"
    outside.mkdir()
    source = f"import os\nos.symlink({str(outside)!r}, 'escape')\n"

    outcome = execute_for_test(
        Program(
            [
                Upload("module", "module", source=source),
                FileUpload(id="file", blob=digest, path="escape/tensor"),
            ],
            blob_bytes={digest: raw},
        ),
        FakeRuntime(),
        UNSHARED_GPU,
    )

    assert outcome.status == "FAILED"
    assert outcome.error["kind"] == "runtime"
    assert outcome.error["instruction_index"] == 1
    assert outcome.error["instruction_id"] == "file"
    assert not (outside / "tensor").exists()


def test_python_module_upload_takes_the_gpu():
    lease = RecordingLease()
    program = Program([Upload("k", "module", source="def main(x): return x")])

    class PlacementRuntime(FakeRuntime):
        def load_module(self, source):
            assert lease.held
            return super().load_module(source)

    outcome = execute_for_test(program, PlacementRuntime(), lease)
    assert outcome.status == "COMPLETED"
    assert lease.acquires == 1 and lease.releases == 0


def test_a_cpu_only_function_hands_the_gpu_over_for_its_call():
    source = "def main(x):\n    return x + 1\n"
    lease = RecordingLease()
    program = Program(
        [
            Upload("module", "module", source=source),  # a Python upload takes the GPU
            GetFunction("host", ref("module"), "main", cpu_only=True),
            Run("off", ref("host"), [1]),  # released for the call
            GetFunction("device", ref("module"), "main"),  # taken back
            Run("on", ref("device"), [1]),
            Return("off", ref("off")),
            Return("on", ref("on")),
        ]
    )
    outcome = execute_for_test(program, FakeRuntime(), lease)
    assert outcome.status == "COMPLETED"
    assert outcome.results["off"] == outcome.results["on"] == {"type": "integer", "value": 2}
    assert lease.acquires == 2 and lease.releases == 1  # final ownership stays with caller


# Stands in for uploaded code that reaches the CUDA API despite its declaration.
CUDA_TOUCHING = (
    "from kcoral.testing import simulate_cuda_call\n\n"
    "def main():\n"
    "    simulate_cuda_call('cudaMalloc')\n"
    "    return 1\n"
)


def test_a_cpu_only_function_that_reaches_cuda_fails_naming_the_call():
    def call(cpu_only):
        return execute_for_test(
            Program(
                [
                    Upload("module", "module", source=CUDA_TOUCHING),
                    GetFunction("fn", ref("module"), "main", cpu_only=cpu_only),
                    Run("value", ref("fn"), []),
                    Return("value", ref("value")),
                ]
            ),
            FakeRuntime(),
            UNSHARED_GPU,
        )

    outcome = call(cpu_only=True)
    assert outcome.status == "FAILED" and outcome.results == {}
    error = outcome.error
    assert error["kind"] == "gpu_access" and error["instruction_id"] == "value"
    assert error["cuda_call"] == "cudaMalloc" and "cudaMalloc" in error["message"]
    assert error["location"] == "<uploaded>:1 in main"
    assert isinstance(error["detected_at_ns"], int)
    # Undeclared, the same function holds the GPU and may use it freely.
    assert call(cpu_only=False).status == "COMPLETED"


def test_a_gpu_access_error_keeps_its_kind_over_a_stale_cuda_error():
    class StaleErrorRuntime(FakeRuntime):
        def take_last_error(self):
            return "CUDA error cudaErrorInvalidValue (1): invalid argument"

    outcome = execute_for_test(
        Program(
            [
                Upload("module", "module", source=CUDA_TOUCHING),
                GetFunction("fn", ref("module"), "main", cpu_only=True),
                Run("value", ref("fn"), []),
            ]
        ),
        StaleErrorRuntime(),
        UNSHARED_GPU,
    )
    error = outcome.error
    assert error["kind"] == "gpu_access" and error["cuda_call"] == "cudaMalloc"
    assert error["message"].endswith(
        "; CUDA also reports CUDA error cudaErrorInvalidValue (1): invalid argument"
    )


def test_explicit_host_build_and_gpu_load_use_the_correct_lease():
    lease = RecordingLease()
    runtime = FakeRuntime()
    runtime.lease = lease
    source = """
def build():
    assert not _test_runtime.lease.held
    return b'artifact'

def load(artifact):
    assert _test_runtime.lease.held
    assert artifact == b'artifact'
    return lambda x: x + 1
"""
    program = Program(
        [
            Upload("module", "module", source=source),
            GetFunction("build", ref("module"), "build", cpu_only=True),
            GetFunction("load", ref("module"), "load"),
            Run("artifact", ref("build")),
            Run("kernel", ref("load"), [ref("artifact")]),
            Run("result", ref("kernel"), [41]),
            Return("result", ref("result")),
        ]
    )
    outcome = execute_for_test(program, runtime, lease)
    assert outcome.status == "COMPLETED", outcome.error
    assert outcome.results["result"] == {"type": "integer", "value": 42}
    assert lease.acquires == 2 and lease.releases == 1


def test_handle_destructors_run_before_gpu_lease_is_released():
    lease = RecordingLease()
    observed = []

    class Module:
        def finish(self):
            return None

        def __del__(self):
            observed.append(("destroy", lease.held))

    class Runtime(FakeRuntime):
        def load_module(self, source, language="python"):
            return Module()

        def reset(self):
            observed.append(("reset", lease.held))

    program = Program(
        [
            Upload("module", "module", source="unused"),
            GetFunction("fn", Ref("module"), "finish", cpu_only=True),
            Run("done", Ref("fn"), []),
        ]
    )
    result = execute_for_test(program, Runtime(), lease)
    assert result.status == "COMPLETED"
    assert observed == [("destroy", True), ("reset", True)]
    assert lease.held
