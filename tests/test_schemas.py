import pytest

from benchmark_server.errors import ValidationError
from benchmark_server.schemas import (
    is_json_structural,
    is_ref,
    parse_program,
    to_structural,
)


def test_parse_run_and_upload():
    p = parse_program(
        {
            "instructions": [
                {
                    "id": "k",
                    "op": "upload",
                    "kind": "function",
                    "key": "sha256:a",
                    "inline": {"source": "x=1"},
                },
                {"id": "x", "op": "run", "fn": "builtin.randn", "args": [{"shape": [4]}]},
            ]
        }
    )
    assert p.uploads()[0].id == "k"
    assert p.instructions[1].op == "run" and p.instructions[1].fn == "builtin.randn"


def test_duplicate_id_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        parse_program(
            {
                "instructions": [
                    {"id": "a", "op": "upload", "kind": "function", "key": "k"},
                    {"id": "a", "op": "run", "fn": "f", "args": []},
                ]
            }
        )


def test_unknown_op_and_kind():
    with pytest.raises(ValidationError, match="unknown op"):
        parse_program({"instructions": [{"id": "a", "op": "frob"}]})
    with pytest.raises(ValidationError, match="unknown kind"):
        parse_program({"instructions": [{"id": "a", "op": "upload", "kind": "weird", "key": "k"}]})


def test_forward_reference_rejected():
    with pytest.raises(ValidationError, match="forward handle"):
        parse_program(
            {
                "instructions": [
                    {"id": "a", "op": "run", "fn": "f", "args": [{"$ref": "later"}]},
                    {"id": "later", "op": "run", "fn": "g", "args": []},
                ]
            }
        )


def test_empty_instructions_rejected():
    with pytest.raises(ValidationError):
        parse_program({"instructions": []})


def test_helpers():
    assert is_ref({"$ref": "x"}) and not is_ref({"$ref": "x", "y": 1})
    assert is_json_structural({"a": [1, 2, "s"], "b": None})
    assert not is_json_structural(object())
    assert to_structural({"latency_ms": 1.0}, "p") == {"latency_ms": 1.0}
    assert to_structural(object(), "p") == {"handle": "p"}
