import pytest

from kcoral.errors import ValidationError
from kcoral.schemas import (
    FileUpload,
    FolderUpload,
    GetFunction,
    Ref,
    Return,
    Run,
    Upload,
    parse_program,
    strict_json_loads,
)

TENSOR_HASH = "0" * 64


def test_parse_complete_program():
    program = parse_program(
        {
            "instructions": [
                {
                    "op": "upload",
                    "id": "module",
                    "kind": "module",
                    "source": "def kernel(x):\n    return x\n",
                },
                {
                    "op": "get_function",
                    "id": "kernel",
                    "module": {"$ref": "module"},
                    "name": "kernel",
                    "cpu_only": True,
                },
                {
                    "op": "upload",
                    "id": "tensor",
                    "kind": "tensor",
                    "blob": TENSOR_HASH,
                    "dtype": "float32",
                    "shape": [2, 3],
                },
                {"op": "run", "id": "result", "fn": {"$ref": "kernel"}, "args": [1]},
                {"op": "return", "key": "answer", "value": {"$ref": "result"}},
            ],
            "options": {"timeout_seconds": 12, "output_limit_bytes": 0},
        }
    )
    assert isinstance(program.instructions[0], Upload)
    assert program.instructions[0].language == "python"  # the default when unset
    assert isinstance(program.instructions[1], GetFunction)
    assert program.instructions[1].cpu_only is True  # False when unset
    assert isinstance(program.instructions[3], Run)
    assert isinstance(program.instructions[4], Return)
    # References are resolved to ``Ref`` at parse time; literals stay untouched.
    assert program.instructions[1].module == Ref("module")
    assert program.instructions[3].fn == Ref("kernel")
    assert program.instructions[3].args == [1]
    assert program.instructions[4].value == Ref("result")
    assert program.options == {"timeout_seconds": 12.0, "output_limit_bytes": 0}
    assert program.blob_uploads()[0].blob == TENSOR_HASH


def test_parse_cuda_module_upload():
    program = parse_program(
        {
            "instructions": [
                {
                    "op": "upload",
                    "id": "kernel",
                    "kind": "module",
                    "language": "cuda",
                    "source": "void add(tvm::ffi::TensorView x) {}",
                },
                {
                    "op": "get_function",
                    "id": "add",
                    "module": {"$ref": "kernel"},
                    "name": "add",
                },
            ]
        }
    )
    assert program.instructions[0].language == "cuda"
    assert program.instructions[1] == GetFunction("add", Ref("kernel"), "add")


def test_parse_library_upload():
    program = parse_program(
        {
            "instructions": [
                {
                    "op": "upload",
                    "id": "kernel",
                    "kind": "library",
                    "blob": TENSOR_HASH,
                }
            ]
        }
    )
    upload = program.instructions[0]
    assert upload.kind == "library"
    # blob-backed, so it joins tensors in the cache-admission path
    assert program.blob_uploads() == [upload]


def test_parse_library_module_and_get_function():
    program = parse_program(
        {
            "instructions": [
                {
                    "op": "upload",
                    "id": "kernels",
                    "kind": "library",
                    "blob": TENSOR_HASH,
                },
                {
                    "op": "get_function",
                    "id": "step",
                    "module": {"$ref": "kernels"},
                    "name": "namespace.step",
                },
            ]
        }
    )
    upload, get_function = program.instructions
    assert isinstance(upload, Upload)
    assert isinstance(get_function, GetFunction)
    assert get_function.module == Ref("kernels") and get_function.name == "namespace.step"


def test_parse_bytes_upload():
    program = parse_program(
        {
            "instructions": [
                {
                    "op": "upload",
                    "id": "file",
                    "kind": "bytes",
                    "blob": TENSOR_HASH,
                }
            ]
        }
    )
    upload = program.instructions[0]
    assert upload.kind == "bytes" and upload.blob == TENSOR_HASH
    assert program.blob_uploads() == [upload]


def test_parse_file_upload_normalizes_relative_path_without_creating_a_handle():
    program = parse_program(
        {
            "instructions": [
                {
                    "op": "upload",
                    "kind": "file",
                    "blob": TENSOR_HASH,
                    "path": "./weights//tensor.bin",
                }
            ]
        }
    )
    upload = program.instructions[0]
    assert upload == FileUpload(blob=TENSOR_HASH, path="weights/tensor.bin")
    assert program.blob_uploads() == [upload]


@pytest.mark.parametrize(
    "path",
    ["", ".", "..", "../tensor", "data/../tensor", "/tmp/tensor", "data\\tensor", "x\x00y"],
)
def test_file_upload_rejects_unsafe_paths(path):
    with pytest.raises(ValidationError, match="filesystem 'path'"):
        parse_program(
            {"instructions": [{"op": "upload", "kind": "file", "blob": TENSOR_HASH, "path": path}]}
        )


@pytest.mark.parametrize(
    "paths,match",
    [
        (["a", "./a"], "duplicate"),
        (["a", "a/b"], "conflicting"),
        (["a/b", "a"], "conflicting"),
        (["a", "a-b", "a/b"], "conflicting"),
    ],
)
def test_file_upload_rejects_duplicate_and_file_directory_conflicts(paths, match):
    instructions = [
        {"op": "upload", "kind": "file", "blob": TENSOR_HASH, "path": path} for path in paths
    ]
    with pytest.raises(ValidationError, match=match):
        parse_program({"instructions": instructions})


@pytest.mark.parametrize(
    "instruction,match",
    [
        ({"op": "upload", "id": "x", "kind": "function", "key": "old"}, "unknown kind"),
        ({"op": "upload", "id": "x", "kind": "module"}, "missing field"),
        (
            {
                "op": "upload",
                "id": "x",
                "kind": "tensor",
                "blob": "sha256:old",
                "dtype": "float32",
                "shape": [1],
            },
            "lowercase SHA-256",
        ),
        (
            {
                "op": "upload",
                "id": "x",
                "kind": "tensor",
                "blob": TENSOR_HASH,
                "dtype": "float4_e2m1",
                "shape": [1],
            },
            "unsupported dtype",
        ),
        ({"op": "run", "id": "x", "fn": {"$ref": "x"}, "extra": 1}, "unknown field"),
        ({"op": "run", "id": "x", "fn": "builtin.zeros"}, "'fn' must be"),
        ({"op": "unknown", "id": "x"}, "unknown op"),
        (
            {"op": "upload", "id": "x", "kind": "module", "source": "", "entry": "main"},
            "unknown field",
        ),
        (
            {"op": "upload", "id": "x", "kind": "module", "source": "", "language": "rust"},
            "unsupported language",
        ),
        (
            {
                "op": "upload",
                "id": "x",
                "kind": "library",
                "blob": TENSOR_HASH,
                "entry": "add_one",
            },
            "unknown field",
        ),
        (
            {"op": "upload", "id": "x", "kind": "bytes", "blob": "sha256:old"},
            "lowercase SHA-256",
        ),
        (
            {
                "op": "upload",
                "id": "file-has-no-handle",
                "kind": "file",
                "blob": TENSOR_HASH,
                "path": "tensor",
            },
            "unknown field",
        ),
        (
            {"op": "upload", "kind": "file", "blob": "sha256:old", "path": "tensor"},
            "lowercase SHA-256",
        ),
    ],
)
def test_invalid_instruction_shapes_rejected(instruction, match):
    with pytest.raises(ValidationError, match=match):
        parse_program({"instructions": [instruction]})


def test_duplicate_handles_and_return_keys_rejected():
    with pytest.raises(ValidationError, match="duplicate instruction id"):
        parse_program(
            {
                "instructions": [
                    {"op": "upload", "id": "x", "kind": "module", "source": "def main(): pass"},
                    {"op": "upload", "id": "x", "kind": "module", "source": "def main(): pass"},
                ]
            }
        )
    with pytest.raises(ValidationError, match="duplicate return key"):
        parse_program(
            {
                "instructions": [
                    {"op": "upload", "id": "x", "kind": "module", "source": "def main(): pass"},
                    {"op": "return", "key": "x", "value": {"$ref": "x"}},
                    {"op": "return", "key": "x", "value": {"$ref": "x"}},
                ]
            }
        )


def test_returns_may_interleave_with_uploads_and_runs():
    program = parse_program(
        {
            "instructions": [
                {"op": "upload", "id": "x", "kind": "module", "source": "def main(): pass"},
                {"op": "return", "key": "x", "value": {"$ref": "x"}},
                {"op": "run", "id": "y", "fn": {"$ref": "x"}, "args": [{"$ref": "x"}]},
                {"op": "return", "key": "y", "value": {"$ref": "y"}},
            ]
        }
    )
    # The run after the first return still resolves its reference.
    assert program.instructions[2].args == [Ref("x")]


def test_forward_references_rejected():
    with pytest.raises(ValidationError, match="unknown/forward"):
        parse_program(
            {
                "instructions": [
                    {"op": "run", "id": "x", "fn": {"$ref": "x"}, "args": [{"$ref": "y"}]}
                ]
            }
        )
    # A return may not reach forward either, even though order is otherwise free.
    with pytest.raises(ValidationError, match="unknown/forward"):
        parse_program(
            {
                "instructions": [
                    {"op": "return", "key": "x", "value": {"$ref": "x"}},
                    {"op": "upload", "id": "x", "kind": "module", "source": "def main(): pass"},
                ]
            }
        )
    with pytest.raises(ValidationError, match="unknown/forward"):
        parse_program(
            {
                "instructions": [
                    {
                        "op": "get_function",
                        "id": "step",
                        "module": {"$ref": "module"},
                        "name": "step",
                    },
                    {
                        "op": "upload",
                        "id": "module",
                        "kind": "library",
                        "blob": TENSOR_HASH,
                    },
                ]
            }
        )


@pytest.mark.parametrize(
    "instruction,match",
    [
        (
            {"op": "get_function", "id": "fn", "module": "module", "name": "step"},
            "must be.*ref",
        ),
        (
            {"op": "get_function", "id": "fn", "module": {"$ref": "module"}, "name": ""},
            "non-empty string",
        ),
        (
            {
                "op": "get_function",
                "id": "fn",
                "module": {"$ref": "module"},
                "name": "step",
                "cpu_only": "yes",
            },
            "'cpu_only' must be a boolean",
        ),
        (
            {
                "op": "get_function",
                "id": "fn",
                "module": {"$ref": "module"},
                "name": "step",
                "extra": True,
            },
            "unknown field",
        ),
    ],
)
def test_invalid_get_function_shapes_rejected(instruction, match):
    with pytest.raises(ValidationError, match=match):
        parse_program(
            {
                "instructions": [
                    {
                        "op": "upload",
                        "id": "module",
                        "kind": "library",
                        "blob": TENSOR_HASH,
                    },
                    instruction,
                ]
            }
        )


@pytest.mark.parametrize(
    "options",
    [
        None,
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"output_limit_bytes": -1},
        {"output_limit_bytes": True},
        {"unknown": 1},
    ],
)
def test_invalid_options_rejected(options):
    with pytest.raises(ValidationError):
        parse_program(
            {
                "instructions": [
                    {"op": "upload", "id": "x", "kind": "module", "source": "def main(): pass"}
                ],
                "options": options,
            }
        )


def test_strict_json_rejects_duplicate_keys_and_non_finite_numbers():
    with pytest.raises(ValidationError, match="duplicate"):
        strict_json_loads('{"instructions": [], "instructions": []}')
    with pytest.raises(ValidationError, match="non-finite"):
        strict_json_loads('{"value": NaN}')


def test_folder_upload_has_no_handle_and_references_one_blob():
    wire = {"op": "upload", "kind": "folder", "blob": TENSOR_HASH, "path": "./data//"}
    program = parse_program({"instructions": [wire]})
    assert program.instructions == [FolderUpload(blob=TENSOR_HASH, path="data")]
    assert program.blob_uploads() == program.instructions
    for field, value in [("id", "folder"), ("source", "text"), ("dtype", "uint8")]:
        with pytest.raises(ValidationError, match="unknown field"):
            parse_program({"instructions": [{**wire, field: value}]})
    for field, value in [("blob", "invalid"), ("path", "../outside")]:
        with pytest.raises(ValidationError):
            parse_program({"instructions": [{**wire, field: value}]})
