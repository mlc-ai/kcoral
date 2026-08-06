"""Validated wire types for the execution protocol."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import ValidationError
from .keys import is_blob_hash

DTYPE_ITEM_SIZES: dict[str, int] = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}


@dataclass
class Upload:
    id: str
    kind: Literal["module", "tensor", "library"]
    source: str | None = None
    entry: str | None = None
    language: Literal["python", "cuda"] = "python"
    blob: str | None = None
    dtype: str | None = None
    shape: list[int] | None = None
    op: Literal["upload"] = "upload"


@dataclass(frozen=True)
class Ref:
    """A validated reference to an earlier handle; wire form is ``{"$ref": id}``."""

    id: str


@dataclass
class Run:
    id: str
    fn: str | Ref
    args: list[Any] = field(default_factory=list)  # ``Ref`` or a JSON literal
    op: Literal["run"] = "run"


@dataclass
class Return:
    key: str
    value: Ref
    op: Literal["return"] = "return"


Instruction = Upload | Run | Return


@dataclass
class Program:
    instructions: list[Instruction]
    options: dict[str, Any] = field(default_factory=dict)
    # Filled by the HTTP front-end after multipart validation and cache lookup.
    blob_bytes: dict[str, bytes] = field(default_factory=dict)

    def blob_uploads(self) -> list[Upload]:
        """Uploads whose payload comes from the content-addressed blob cache."""
        return [
            instruction
            for instruction in self.instructions
            if isinstance(instruction, Upload) and instruction.blob is not None
        ]


@dataclass
class ProgramOutcome:
    """What running one program produced, before the front-end adds request metadata."""

    status: str
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    binary_parts: dict[str, bytes] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False


def is_ref(value: Any) -> bool:
    """Return whether ``value`` has the wire shape of a handle reference.

    Parsing turns every such value into a :class:`Ref`, so only ``parse_program``
    inspects the wire shape; everything downstream matches on ``Ref``.
    """
    return (
        isinstance(value, dict)
        and set(value) == {"$ref"}
        and isinstance(value["$ref"], str)
        and bool(value["$ref"])
    )


def strict_json_loads(data: bytes | str) -> Any:
    """Parse JSON while rejecting duplicate keys and non-finite numbers."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValidationError(f"non-finite number {value} is not allowed")

    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    except ValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"malformed JSON: {exc}") from exc


def parse_program(body: Any) -> Program:
    if not isinstance(body, dict):
        raise ValidationError("program must be a JSON object")
    _check_fields(body, {"instructions", "options"}, {"instructions"}, "program")

    raw_instructions = body["instructions"]
    if not isinstance(raw_instructions, list) or not raw_instructions:
        raise ValidationError("'instructions' must be a non-empty array")
    options = _parse_options(body.get("options", {}))

    handles: set[str] = set()
    return_keys: set[str] = set()
    instructions: list[Instruction] = []

    # Instructions may appear in any order; ``handles`` grows as they are parsed,
    # so a reference to a later instruction is rejected as a forward reference.
    for index, item in enumerate(raw_instructions):
        if not isinstance(item, dict):
            raise ValidationError(f"instruction {index} must be an object")
        op = item.get("op")
        if op == "return":
            instruction = _parse_return(item, index, handles, return_keys)
            return_keys.add(instruction.key)
        elif op in ("upload", "run"):
            instruction_id = item.get("id")
            if not isinstance(instruction_id, str) or not instruction_id:
                raise ValidationError(f"instruction {index} needs a non-empty string 'id'")
            if instruction_id in handles:
                raise ValidationError(f"duplicate instruction id: {instruction_id!r}")
            if op == "upload":
                instruction = _parse_upload(item, index)
            else:
                instruction = _parse_run(item, index, handles)
            handles.add(instruction_id)
        else:
            raise ValidationError(f"instruction {index}: unknown op {op!r}")
        instructions.append(instruction)

    return Program(instructions=instructions, options=options)


def expected_tensor_nbytes(dtype: str, shape: list[int]) -> int:
    try:
        item_size = DTYPE_ITEM_SIZES[dtype]
    except KeyError as exc:
        raise ValidationError(f"unsupported tensor dtype: {dtype!r}") from exc
    elements = math.prod(shape)
    return elements * item_size


def _parse_options(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("'options' must be an object")
    _check_fields(value, {"timeout_seconds", "output_limit_bytes"}, set(), "options")

    options: dict[str, Any] = {}
    if "timeout_seconds" in value:
        timeout = value["timeout_seconds"]
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValidationError("'timeout_seconds' must be a finite positive number")
        options["timeout_seconds"] = float(timeout)
    if "output_limit_bytes" in value:
        output_limit = value["output_limit_bytes"]
        if isinstance(output_limit, bool) or not isinstance(output_limit, int) or output_limit < 0:
            raise ValidationError("'output_limit_bytes' must be a non-negative integer")
        options["output_limit_bytes"] = output_limit
    return options


def _parse_upload(item: dict[str, Any], index: int) -> Upload:
    kind = item.get("kind")
    if kind == "module":
        _check_fields(
            item,
            {"op", "id", "kind", "source", "entry", "language"},
            {"op", "id", "kind", "source"},
            f"instruction {index}",
        )
        source = item["source"]
        if not isinstance(source, str):
            raise ValidationError(f"module upload {item['id']!r}: 'source' must be a string")
        language = item.get("language", "python")
        if language not in ("python", "cuda"):
            raise ValidationError(
                f"module upload {item['id']!r}: unsupported language {language!r}"
            )
        entry = item.get("entry")
        if entry is not None and not (isinstance(entry, str) and entry.isidentifier()):
            raise ValidationError(f"module upload {item['id']!r}: 'entry' must be an identifier")
        if language == "cuda":
            # Nothing executes a CUDA upload, so no entry can be inferred from it.
            if entry is None:
                raise ValidationError(
                    f"module upload {item['id']!r}: a 'cuda' module must name its 'entry'"
                )
            if entry == "main":
                raise ValidationError(
                    f"module upload {item['id']!r}: C++ reserves 'main'; name the entry otherwise"
                )
        return Upload(id=item["id"], kind="module", source=source, entry=entry, language=language)
    if kind == "tensor":
        _check_fields(
            item,
            {"op", "id", "kind", "blob", "dtype", "shape"},
            {"op", "id", "kind", "blob", "dtype", "shape"},
            f"instruction {index}",
        )
        blob = item["blob"]
        dtype = item["dtype"]
        shape = item["shape"]
        if not is_blob_hash(blob):
            raise ValidationError(
                f"tensor upload {item['id']!r}: 'blob' must be a lowercase SHA-256 digest"
            )
        if not isinstance(dtype, str) or dtype not in DTYPE_ITEM_SIZES:
            raise ValidationError(f"tensor upload {item['id']!r}: unsupported dtype {dtype!r}")
        if not isinstance(shape, list) or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
            for dimension in shape
        ):
            raise ValidationError(
                f"tensor upload {item['id']!r}: 'shape' must be an array of non-negative integers"
            )
        return Upload(id=item["id"], kind="tensor", blob=blob, dtype=dtype, shape=shape)
    if kind == "library":
        _check_fields(
            item,
            {"op", "id", "kind", "blob", "entry"},
            {"op", "id", "kind", "blob", "entry"},
            f"instruction {index}",
        )
        blob = item["blob"]
        entry = item["entry"]
        if not is_blob_hash(blob):
            raise ValidationError(
                f"library upload {item['id']!r}: 'blob' must be a lowercase SHA-256 digest"
            )
        if not (isinstance(entry, str) and entry.isidentifier()):
            raise ValidationError(f"library upload {item['id']!r}: 'entry' must be an identifier")
        return Upload(id=item["id"], kind="library", blob=blob, entry=entry)
    raise ValidationError(f"upload {item.get('id')!r}: unknown kind {kind!r}")


def _parse_run(item: dict[str, Any], index: int, handles: set[str]) -> Run:
    _check_fields(item, {"op", "id", "fn", "args"}, {"op", "id", "fn"}, f"instruction {index}")
    raw_fn = item["fn"]
    fn: str | Ref
    if isinstance(raw_fn, str):
        if not raw_fn:
            raise ValidationError(f"run {item['id']!r}: 'fn' must not be empty")
        fn = raw_fn
    elif is_ref(raw_fn):
        fn = _resolve_ref(raw_fn, handles, item["id"])
    else:
        raise ValidationError(f"run {item['id']!r}: 'fn' must be a name or {{'$ref': id}}")

    raw_args = item.get("args", [])
    if not isinstance(raw_args, list):
        raise ValidationError(f"run {item['id']!r}: 'args' must be an array")
    args: list[Any] = []
    for argument in raw_args:
        # Only a top-level argument is a reference; one nested inside a JSON
        # value stays a literal, as it does at execution time.
        if is_ref(argument):
            args.append(_resolve_ref(argument, handles, item["id"]))
        else:
            _validate_json_value(argument, f"run {item['id']!r} argument")
            args.append(argument)
    return Run(id=item["id"], fn=fn, args=args)


def _parse_return(
    item: dict[str, Any], index: int, handles: set[str], return_keys: set[str]
) -> Return:
    _check_fields(item, {"op", "key", "value"}, {"op", "key", "value"}, f"instruction {index}")
    key = item["key"]
    if not isinstance(key, str) or not key:
        raise ValidationError(f"return instruction {index} needs a non-empty string 'key'")
    if key in return_keys:
        raise ValidationError(f"duplicate return key: {key!r}")
    value = item["value"]
    if not is_ref(value):
        raise ValidationError(f"return {key!r}: 'value' must be {{'$ref': id}}")
    return Return(key=key, value=_resolve_ref(value, handles, f"return {key!r}"))


def _resolve_ref(reference: dict[str, Any], handles: set[str], owner: str) -> Ref:
    target = reference["$ref"]
    if target not in handles:
        raise ValidationError(f"{owner!r} references unknown/forward handle {target!r}")
    return Ref(target)


def _validate_json_value(value: Any, label: str) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for child in value:
            _validate_json_value(child, label)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValidationError(f"{label} contains a non-string object key")
        for child in value.values():
            _validate_json_value(child, label)
        return
    raise ValidationError(f"{label} contains a non-JSON value")


def _check_fields(value: dict[str, Any], allowed: set[str], required: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValidationError(f"{label} has unknown field(s): {', '.join(sorted(unknown))}")
    missing = required - set(value)
    if missing:
        raise ValidationError(f"{label} is missing field(s): {', '.join(sorted(missing))}")
