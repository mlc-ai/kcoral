import pytest

from benchmark_server.errors import ValidationError
from benchmark_server.schemas import Ref, Return, Run, Upload, parse_program, strict_json_loads

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
                    "entry": "kernel",
                },
                {
                    "op": "upload",
                    "id": "tensor",
                    "kind": "tensor",
                    "blob": TENSOR_HASH,
                    "dtype": "float32",
                    "shape": [2, 3],
                },
                {"op": "run", "id": "result", "fn": {"$ref": "module"}, "args": [1]},
                {"op": "return", "key": "answer", "value": {"$ref": "result"}},
            ],
            "options": {"timeout_seconds": 12, "output_limit_bytes": 0},
        }
    )
    assert isinstance(program.instructions[0], Upload)
    assert program.instructions[0].entry == "kernel"
    assert isinstance(program.instructions[2], Run)
    assert isinstance(program.instructions[3], Return)
    # References are resolved to ``Ref`` at parse time; literals stay untouched.
    assert program.instructions[2].fn == Ref("module")
    assert program.instructions[2].args == [1]
    assert program.instructions[3].value == Ref("result")
    assert program.options == {"timeout_seconds": 12.0, "output_limit_bytes": 0}
    assert program.tensor_uploads()[0].blob == TENSOR_HASH


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
        ({"op": "run", "id": "x", "fn": "builtin.zeros", "extra": 1}, "unknown field"),
        ({"op": "unknown", "id": "x"}, "unknown op"),
        (
            {"op": "upload", "id": "x", "kind": "module", "source": "", "entry": "not an id"},
            "'entry' must be a Python identifier",
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
                    {"op": "run", "id": "x", "fn": "builtin.zeros"},
                    {"op": "run", "id": "x", "fn": "builtin.empty"},
                ]
            }
        )
    with pytest.raises(ValidationError, match="duplicate return key"):
        parse_program(
            {
                "instructions": [
                    {"op": "run", "id": "x", "fn": "builtin.zeros"},
                    {"op": "return", "key": "x", "value": {"$ref": "x"}},
                    {"op": "return", "key": "x", "value": {"$ref": "x"}},
                ]
            }
        )


def test_returns_may_interleave_with_uploads_and_runs():
    program = parse_program(
        {
            "instructions": [
                {"op": "run", "id": "x", "fn": "builtin.zeros"},
                {"op": "return", "key": "x", "value": {"$ref": "x"}},
                {"op": "run", "id": "y", "fn": "builtin.zeros", "args": [{"$ref": "x"}]},
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
                    {"op": "run", "id": "x", "fn": "builtin.zeros", "args": [{"$ref": "y"}]}
                ]
            }
        )
    # A return may not reach forward either, even though order is otherwise free.
    with pytest.raises(ValidationError, match="unknown/forward"):
        parse_program(
            {
                "instructions": [
                    {"op": "return", "key": "x", "value": {"$ref": "x"}},
                    {"op": "run", "id": "x", "fn": "builtin.zeros"},
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
                "instructions": [{"op": "run", "id": "x", "fn": "builtin.zeros"}],
                "options": options,
            }
        )


def test_strict_json_rejects_duplicate_keys_and_non_finite_numbers():
    with pytest.raises(ValidationError, match="duplicate"):
        strict_json_loads('{"instructions": [], "instructions": []}')
    with pytest.raises(ValidationError, match="non-finite"):
        strict_json_loads('{"value": NaN}')
