"""Wire validation, blob identities, and multipart request/response framing."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
from dataclasses import dataclass
from email.parser import BytesHeaderParser
from email.policy import default
from itertools import pairwise
from typing import Any

from kcoral.errors import ValidationError
from kcoral.schemas import (
    DTYPE_ITEM_SIZES,
    FileReturn,
    FileUpload,
    GetFunction,
    Instruction,
    Program,
    Ref,
    Return,
    Run,
    Upload,
)

_BLOB_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def is_blob_hash(value: object) -> bool:
    return isinstance(value, str) and _BLOB_HASH_RE.fullmatch(value) is not None


def compute_blob_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_blob(blob_hash: str, data: bytes) -> None:
    if not is_blob_hash(blob_hash):
        raise ValidationError("blob name must contain a lowercase SHA-256 digest")
    actual = compute_blob_hash(data)
    if actual != blob_hash:
        raise ValidationError(f"blob hash mismatch: declared {blob_hash}, computed {actual}")


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
        elif op in ("upload", "get_function", "run"):
            instruction_id = item.get("id")
            if not isinstance(instruction_id, str) or not instruction_id:
                raise ValidationError(f"instruction {index} needs a non-empty string 'id'")
            if instruction_id in handles:
                raise ValidationError(f"duplicate instruction id: {instruction_id!r}")
            if op == "upload":
                instruction = _parse_upload(item, index)
                if isinstance(instruction, FileUpload):
                    file_paths.append(instruction.path)
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
    _check_fields(value, {"timeout_seconds", "output_limit_bytes", "gpu_count"}, set(), "options")

    options: dict[str, Any] = {}
    if "gpu_count" in value:
        count = value["gpu_count"]
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 8:
            raise ValidationError("'gpu_count' must be an integer between 1 and 8")
        options["gpu_count"] = count
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


def _parse_upload(item: dict[str, Any], index: int) -> Upload | FileUpload:
    kind = item.get("kind")
    if kind == "file":
        return _parse_file_upload(item, index)
    if kind == "module":
        _check_fields(
            item,
            {"op", "id", "kind", "source"},
            {"op", "id", "kind", "source"},
            f"instruction {index}",
        )
        source = item["source"]
        if not isinstance(source, str):
            raise ValidationError(f"module upload {item['id']!r}: 'source' must be a string")
        return Upload(id=item["id"], kind="module", source=source)
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


def _parse_file_upload(item: dict[str, Any], index: int) -> FileUpload:
    _check_fields(
        item,
        {"op", "id", "kind", "blob", "path"},
        {"op", "id", "kind", "blob", "path"},
        f"instruction {index}",
    )
    blob = item["blob"]
    if not is_blob_hash(blob):
        raise ValidationError("file upload: 'blob' must be a lowercase SHA-256 digest")
    return FileUpload(id=item["id"], blob=blob, path=normalize_file_path(item["path"]))


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


@dataclass(frozen=True)
class MultipartPart:
    name: str
    content_type: str
    data: bytes


def parse_multipart(content_type: str | None, body: bytes) -> list[MultipartPart]:
    if not content_type or not content_type.lower().startswith("multipart/form-data"):
        raise ValidationError("expected a multipart/form-data request")
    try:
        header = BytesHeaderParser(policy=default).parsebytes(
            f"Content-Type: {content_type}\r\n\r\n".encode("ascii")
        )
        boundary = header.get_boundary()
        if header.get_content_type() != "multipart/form-data" or not boundary:
            raise ValidationError("malformed multipart/form-data boundary")
        marker = b"--" + boundary.encode("ascii")
    except (UnicodeError, ValueError) as exc:
        raise ValidationError("malformed multipart/form-data boundary") from exc
    if b"\r" in marker or b"\n" in marker or header["content-type"].defects:
        raise ValidationError("malformed multipart/form-data boundary")

    # Scan framing as bytes. The email parser is used only for small MIME
    # headers, never to decode/copy a potentially gigabyte-sized binary body.
    def next_boundary(start: int) -> tuple[int, int, bool]:
        candidate = 0 if start == 0 and body.startswith(marker) else -1
        while True:
            if candidate < 0:
                found = body.find(b"\n" + marker, start)
                if found < 0:
                    raise ValidationError("malformed multipart/form-data body")
                candidate = found + 1
            tail = candidate + len(marker)
            closing = body[tail : tail + 2] == b"--"
            if closing:
                tail += 2
            while tail < len(body) and body[tail] in (32, 9):
                tail += 1
            if body[tail : tail + 2] == b"\r\n":
                return candidate, tail + 2, closing
            if body[tail : tail + 1] == b"\n":
                return candidate, tail + 1, closing
            if closing and tail == len(body):
                return candidate, tail, closing
            # A boundary prefix inside binary data is not a delimiter.
            start = candidate + len(marker)
            candidate = -1

    _, part_start, closing = next_boundary(0)
    parts: list[MultipartPart] = []
    while not closing:
        boundary_start, next_start, closing = next_boundary(part_start)
        part_end = boundary_start - 1  # boundary's preceding LF
        if part_end > part_start and body[part_end - 1 : part_end] == b"\r":
            part_end -= 1
        # Only scan the current part's header lines; payload bytes stay opaque.
        cursor = part_start
        while True:
            end = body.find(b"\n", cursor, part_end)
            if end < 0:
                raise ValidationError("malformed multipart part headers")
            if body[cursor:end] in (b"", b"\r"):
                payload_start = end + 1
                break
            cursor = end + 1
        item = BytesHeaderParser(policy=default).parsebytes(body[part_start:payload_start])
        if item.defects:
            raise ValidationError("malformed multipart part headers")
        part_start = next_start
        if len(item.get_all("content-disposition", [])) != 1:
            raise ValidationError("each multipart part must have one Content-Disposition")
        if len(item.get_all("content-type", [])) != 1:
            raise ValidationError("each multipart part must have one Content-Type")
        if item["content-disposition"].defects:
            raise ValidationError("multipart part has a malformed Content-Disposition")
        if item["content-type"].defects:
            raise ValidationError("multipart part has a malformed Content-Type")
        if item.get_content_maintype() == "multipart":
            raise ValidationError("nested multipart parts are not supported")
        if item.get_content_disposition() != "form-data":
            raise ValidationError("multipart parts must use form-data disposition")
        names = [
            value
            for key, value in item.get_params(header="content-disposition", unquote=True)[1:]
            if key.lower() == "name"
        ]
        if len(names) != 1 or not isinstance(names[0], str) or not names[0]:
            raise ValidationError("multipart part is missing its name")
        name = names[0]
        transfer_encoding = item.get("content-transfer-encoding")
        if transfer_encoding and transfer_encoding.lower() not in ("binary", "8bit"):
            raise ValidationError("encoded multipart parts are not supported")
        payload = body[payload_start:part_end]
        parts.append(MultipartPart(name=name, content_type=item.get_content_type(), data=payload))
    if not parts:
        raise ValidationError("multipart body has no parts")
    return parts


def encode_multipart(parts: list[MultipartPart]) -> tuple[bytes, str]:
    if not parts:
        raise ValueError("at least one multipart part is required")
    while True:
        boundary = f"kcoral-{secrets.token_hex(16)}"
        delimiter = f"\r\n--{boundary}".encode("ascii")
        if all(delimiter not in part.data for part in parts):
            break

    chunks: list[bytes] = []
    for part in parts:
        if not part.name or any(character in part.name for character in '"\r\n'):
            raise ValueError(f"invalid multipart part name: {part.name!r}")
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{part.name}"\r\n'.encode("ascii"),
                f"Content-Type: {part.content_type}\r\n\r\n".encode("ascii"),
                part.data,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _encode_response(
    payload: dict[str, object], binary_parts: dict[str, bytes]
) -> tuple[bytes, str]:
    result_bytes = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if not binary_parts:
        return result_bytes, "application/json"
    parts = [MultipartPart("result", "application/json", result_bytes)]
    parts.extend(
        MultipartPart(name, "application/octet-stream", data) for name, data in binary_parts.items()
    )
    return encode_multipart(parts)


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [item for item in items if not (item in seen or seen.add(item))]


def parse_request(content_type: str | None, body: bytes) -> tuple[Program, dict[str, bytes], bytes]:
    parts = parse_multipart(content_type, body)
    program_bytes: bytes | None = None
    supplied_blobs: dict[str, bytes] = {}

    for part in parts:
        if part.name == "program":
            if program_bytes is not None:
                raise ValidationError("duplicate multipart part: 'program'")
            if part.content_type != "application/json":
                raise ValidationError("the 'program' part must use application/json")
            program_bytes = part.data
            continue
        if not part.name.startswith("blob:"):
            raise ValidationError(f"unsupported multipart part: {part.name!r}")
        blob_hash = part.name.removeprefix("blob:")
        if not is_blob_hash(blob_hash):
            raise ValidationError(f"malformed blob part name: {part.name!r}")
        if blob_hash in supplied_blobs:
            raise ValidationError(f"duplicate blob part: {blob_hash}")
        if part.content_type != "application/octet-stream":
            raise ValidationError(f"blob part {blob_hash} must use application/octet-stream")
        verify_blob(blob_hash, part.data)
        supplied_blobs[blob_hash] = part.data

    if program_bytes is None:
        raise ValidationError("missing multipart part: 'program'")
    program = parse_program(strict_json_loads(program_bytes))

    return program, supplied_blobs, program_bytes
