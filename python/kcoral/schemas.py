"""Validated wire types for the execution protocol."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import pairwise
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
    kind: Literal["module", "tensor", "bytes", "library"]
    source: str | None = None
    language: Literal["python", "cuda"] = "python"
    blob: str | None = None
    dtype: str | None = None
    shape: list[int] | None = None
    op: Literal["upload"] = "upload"


@dataclass
class FileUpload:
    """A blob materialized as a regular file in the request workspace."""

    blob: str
    path: str
    kind: Literal["file"] = "file"
    op: Literal["upload"] = "upload"


@dataclass
class FolderUpload:
    """One cached archive unpacked beneath a request-workspace directory."""

    blob: str
    path: str
    kind: Literal["folder"] = "folder"
    op: Literal["upload"] = "upload"


@dataclass(frozen=True)
class Ref:
    """A validated reference to an earlier handle; wire form is ``{"$ref": id}``."""

    id: str


@dataclass
class Run:
    id: str
    fn: Ref
    args: list[Any] = field(default_factory=list)  # ``Ref`` or a JSON literal
    op: Literal["run"] = "run"


@dataclass
class GetFunction:
    id: str
    module: Ref
    name: str
    cpu_only: bool = False
    op: Literal["get_function"] = "get_function"


@dataclass
class Return:
    key: str
    value: Ref
    op: Literal["return"] = "return"


@dataclass
class FileReturn:
    key: str
    kind: Literal["file", "folder"]
    path: str | Ref
    op: Literal["return"] = "return"


Instruction = Upload | FileUpload | FolderUpload | GetFunction | Run | Return | FileReturn


@dataclass
class Program:
    instructions: list[Instruction]
    options: dict[str, Any] = field(default_factory=dict)
    # Filled by the HTTP front-end after multipart validation and cache lookup.
    blob_bytes: dict[str, bytes] = field(default_factory=dict)

    # Trusted limits supplied by the front-end; never accepted in wire options.
    max_return_bytes: int = 256 * 1024**2

    # Validated archive members passed to the worker, never accepted from the wire.
    folder_entries: dict[str, list[tuple[str, int, int]]] = field(default_factory=dict)

    def blob_uploads(self) -> list[Upload | FileUpload | FolderUpload]:
        """Uploads whose payload comes from the content-addressed blob cache."""
        return [
            instruction
            for instruction in self.instructions
            if isinstance(instruction, (FileUpload, FolderUpload))
            or (isinstance(instruction, Upload) and instruction.blob is not None)
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
    """Validate decoded protocol data and build a server-side program.

    :param body: A decoded JSON object with ``instructions`` and optional ``options``.
    :returns: A :class:`kcoral.schemas.Program` containing validated instructions
        and parsed options, not the client-side :class:`kcoral.Program` builder.
    :raises ValidationError: If fields, instruction order, references or paths
        violate the protocol. References must name earlier instructions.

    This function does not execute instructions, resolve cached blobs or
    initialize a worker. It is intended for server integration.
    """
    if not isinstance(body, dict):
        raise ValidationError("program must be a JSON object")
    _check_fields(body, {"instructions", "options"}, {"instructions"}, "program")

    raw_instructions = body["instructions"]
    if not isinstance(raw_instructions, list) or not raw_instructions:
        raise ValidationError("'instructions' must be a non-empty array")
    options = _parse_options(body.get("options", {}))

    handles: set[str] = set()
    return_keys: set[str] = set()
    file_paths: list[str] = []
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
        elif op == "upload" and item.get("kind") in ("file", "folder"):
            instruction = _parse_filesystem_upload(item, index)
            if isinstance(instruction, FileUpload):
                file_paths.append(instruction.path)
        elif op in ("upload", "get_function", "run"):
            instruction_id = item.get("id")
            if not isinstance(instruction_id, str) or not instruction_id:
                raise ValidationError(f"instruction {index} needs a non-empty string 'id'")
            if instruction_id in handles:
                raise ValidationError(f"duplicate instruction id: {instruction_id!r}")
            if op == "upload":
                instruction = _parse_upload(item, index)
            elif op == "get_function":
                instruction = _parse_get_function(item, index, handles)
            else:
                instruction = _parse_run(item, index, handles)
            handles.add(instruction_id)
        else:
            raise ValidationError(f"instruction {index}: unknown op {op!r}")
        instructions.append(instruction)

    validate_and_add_file_paths(file_paths, set())
    return Program(instructions=instructions, options=options)


def normalize_file_path(value: Any) -> str:
    """Validate and canonicalize a request-workspace-relative POSIX path.

    Inspect the raw components before normalization so ``a/../b`` is rejected
    rather than silently turned into a path that appears safe afterwards.
    """
    if not isinstance(value, str) or not value:
        raise ValidationError("filesystem 'path' must be a non-empty string")
    if "\x00" in value:
        raise ValidationError("filesystem 'path' must not contain NUL")
    if "\\" in value:
        raise ValidationError("filesystem 'path' must use POSIX '/' separators")
    if value.startswith("/"):
        raise ValidationError("filesystem 'path' must be relative")

    raw_parts = value.split("/")
    if ".." in raw_parts:
        raise ValidationError("filesystem 'path' must not contain a '..' component")
    parts = [part for part in raw_parts if part not in ("", ".")]
    if not parts:
        raise ValidationError("filesystem 'path' must name a file or folder")
    for part in parts:
        if len(part.encode("utf-8")) > 255:
            raise ValidationError("filesystem 'path' contains a component longer than 255 bytes")
    normalized = "/".join(parts)
    if len(normalized.encode("utf-8")) > 4096:
        raise ValidationError("filesystem 'path' is longer than 4096 bytes")
    return normalized


def validate_and_add_file_path(path: str, paths: set[str]) -> None:
    """Reject destination conflicts, then add the normalized path to ``paths``."""
    for existing in paths:
        if path == existing:
            raise ValidationError(f"duplicate file upload path: {path!r}")
        if path.startswith(existing + "/") or existing.startswith(path + "/"):
            raise ValidationError(f"conflicting file upload paths: {existing!r} and {path!r}")
    paths.add(path)


def validate_and_add_file_paths(additions: list[str], paths: set[str]) -> None:
    """Validate a batch in O(n log n), then update the existing declarations."""
    # Component sorting puts a file immediately before its descendants, even
    # with intervening names such as "a-b" alongside "a" and "a/b".
    ordered = sorted([*paths, *additions], key=lambda path: path.split("/"))
    for previous, current in pairwise(ordered):
        if current == previous:
            raise ValidationError(f"duplicate file upload path: {current!r}")
        if current.startswith(previous + "/"):
            raise ValidationError(f"conflicting file upload paths: {previous!r} and {current!r}")
    paths.update(additions)


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
            {"op", "id", "kind", "source", "language"},
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
        return Upload(id=item["id"], kind="module", source=source, language=language)
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
    if kind == "bytes":
        _check_fields(
            item,
            {"op", "id", "kind", "blob"},
            {"op", "id", "kind", "blob"},
            f"instruction {index}",
        )
        blob = item["blob"]
        if not is_blob_hash(blob):
            raise ValidationError(
                f"bytes upload {item['id']!r}: 'blob' must be a lowercase SHA-256 digest"
            )
        return Upload(id=item["id"], kind="bytes", blob=blob)
    if kind == "library":
        _check_fields(
            item,
            {"op", "id", "kind", "blob"},
            {"op", "id", "kind", "blob"},
            f"instruction {index}",
        )
        blob = item["blob"]
        if not is_blob_hash(blob):
            raise ValidationError(
                f"library upload {item['id']!r}: 'blob' must be a lowercase SHA-256 digest"
            )
        return Upload(id=item["id"], kind="library", blob=blob)
    raise ValidationError(f"upload {item.get('id')!r}: unknown kind {kind!r}")


def _parse_filesystem_upload(item: dict[str, Any], index: int) -> FileUpload | FolderUpload:
    _check_fields(
        item,
        {"op", "kind", "blob", "path"},
        {"op", "kind", "blob", "path"},
        f"instruction {index}",
    )
    blob = item["blob"]
    if not is_blob_hash(blob):
        raise ValidationError(f"{item['kind']} upload: 'blob' must be a lowercase SHA-256 digest")
    cls = FileUpload if item["kind"] == "file" else FolderUpload
    return cls(blob=blob, path=normalize_file_path(item["path"]))


def _parse_get_function(item: dict[str, Any], index: int, handles: set[str]) -> GetFunction:
    _check_fields(
        item,
        {"op", "id", "module", "name", "cpu_only"},
        {"op", "id", "module", "name"},
        f"instruction {index}",
    )
    module = item["module"]
    if not is_ref(module):
        raise ValidationError(f"get_function {item['id']!r}: 'module' must be {{'$ref': id}}")
    name = item["name"]
    if not isinstance(name, str) or not name:
        raise ValidationError(f"get_function {item['id']!r}: 'name' must be a non-empty string")
    cpu_only = item.get("cpu_only", False)
    if not isinstance(cpu_only, bool):
        raise ValidationError(f"get_function {item['id']!r}: 'cpu_only' must be a boolean")
    return GetFunction(
        id=item["id"],
        module=_resolve_ref(module, handles, item["id"]),
        name=name,
        cpu_only=cpu_only,
    )


def _parse_run(item: dict[str, Any], index: int, handles: set[str]) -> Run:
    _check_fields(item, {"op", "id", "fn", "args"}, {"op", "id", "fn"}, f"instruction {index}")
    raw_fn = item["fn"]
    if not is_ref(raw_fn):
        raise ValidationError(f"run {item['id']!r}: 'fn' must be {{'$ref': id}}")
    fn = _resolve_ref(raw_fn, handles, item["id"])

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
) -> Return | FileReturn:
    fields = {"op", "key", "kind", "path"} if "kind" in item else {"op", "key", "value"}
    _check_fields(item, fields, fields, f"instruction {index}")
    key = item["key"]
    if not isinstance(key, str) or not key:
        raise ValidationError(f"return instruction {index} needs a non-empty string 'key'")
    if key in return_keys:
        raise ValidationError(f"duplicate return key: {key!r}")
    if "kind" in item:
        kind = item["kind"]
        if kind not in ("file", "folder"):
            raise ValidationError(f"return {key!r}: unknown kind {kind!r}")
        raw_path = item["path"]
        path = (
            _resolve_ref(raw_path, handles, f"return {key!r}")
            if is_ref(raw_path)
            else normalize_file_path(raw_path)
        )
        return FileReturn(key=key, kind=kind, path=path)
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
