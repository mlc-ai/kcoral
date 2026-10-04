"""Validate execution responses and decode explicitly returned values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import httpx
import ml_dtypes
import numpy as np

from kcoral.artifacts import ReturnedFile, ReturnedFolder, validate_manifest
from kcoral.errors import ProtocolError
from kcoral.protocol import (
    expected_tensor_nbytes,
    is_blob_hash,
    parse_multipart,
    strict_json_loads,
    verify_blob,
)
from kcoral.schemas import DTYPE_ITEM_SIZES


@dataclass
class ProgramResult:
    """Decoded execution outcome, including any results returned before failure.

    :param status: ``COMPLETED`` or ``FAILED``. The client handles cache misses
        internally before producing an outcome.
    :param request_id: Server-generated request identifier for log correlation.
    :param queue_ms: Milliseconds waiting for a worker.
    :param elapsed_ms: Execution elapsed time in milliseconds.
    :param lease_wait_ms: Execution time spent waiting for the exclusive GPU lease.
    :param lease_held_ms: Execution time holding the exclusive GPU lease.
    :param results: Explicitly returned values, keyed by the program's return keys.
        Binary values decode to bytes and tensors to CPU NumPy arrays.
    :param stdout: Captured standard output.
    :param stderr: Captured standard error.
    :param stdout_truncated: Whether standard output exceeded the capture limit.
    :param stderr_truncated: Whether standard error exceeded the capture limit.
    :param error: Structured instruction failure, or ``None`` on success.

    All parameters are available as attributes. When present, ``error`` includes
    the kind, message, instruction index and identifier, and traceback.
    """

    status: str
    request_id: str
    queue_ms: float
    elapsed_ms: float
    # Of `elapsed_ms`: waiting for the GPU, then holding it.
    lease_wait_ms: float
    lease_held_ms: float
    results: dict[str, Any]
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    error: dict[str, Any] | None = None

    @property
    def completed(self) -> bool:
        """Whether every instruction completed successfully."""
        return self.status == "COMPLETED"

    def __getitem__(self, key: str) -> Any:
        """Read an explicitly returned value by key; raise KeyError if absent."""
        return self.results[key]


def _response_body(response: httpx.Response) -> tuple[dict[str, Any], dict[str, bytes]]:
    media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type == "application/json":
        return _json_body(response), {}
    if media_type != "multipart/form-data":
        raise ProtocolError(f"unsupported response content type: {media_type!r}")
    try:
        parts = parse_multipart(response.headers.get("content-type"), response.content)
    except Exception as exc:
        raise ProtocolError(f"malformed multipart response: {exc}") from exc

    result: dict[str, Any] | None = None
    binary_parts: dict[str, bytes] = {}
    for part in parts:
        if part.name == "result":
            if result is not None:
                raise ProtocolError("duplicate 'result' response part")
            if part.content_type != "application/json":
                raise ProtocolError("the 'result' response part must use application/json")
            try:
                parsed = strict_json_loads(part.data)
            except Exception as exc:
                raise ProtocolError(f"the 'result' part is not valid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ProtocolError("the 'result' response part must contain an object")
            result = parsed
        else:
            if part.name in binary_parts:
                raise ProtocolError(f"duplicate response part: {part.name!r}")
            if part.content_type != "application/octet-stream":
                raise ProtocolError(f"response part {part.name!r} has the wrong content type")
            binary_parts[part.name] = part.data
    if result is None:
        raise ProtocolError("multipart response is missing the 'result' part")
    return result, binary_parts


def _json_body(response: httpx.Response) -> dict[str, Any]:
    try:
        body = strict_json_loads(response.content)
    except Exception as exc:
        raise ProtocolError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ProtocolError("JSON response must be an object")
    return body


def _parse_program_result(body: dict[str, Any], binary_parts: dict[str, bytes]) -> ProgramResult:
    try:
        status = body["status"]
        if status not in ("COMPLETED", "FAILED"):
            raise ValueError(f"unexpected program status {status!r}")
        request_id = body["request_id"]
        queue_ms = body["queue_ms"]
        elapsed_ms = body["elapsed_ms"]
        lease_wait_ms = body["lease_wait_ms"]
        lease_held_ms = body["lease_held_ms"]
        stdout = body["stdout"]
        stderr = body["stderr"]
        stdout_truncated = body.get("stdout_truncated", False)
        stderr_truncated = body.get("stderr_truncated", False)
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        timings = (queue_ms, elapsed_ms, lease_wait_ms, lease_held_ms)
        if not all(_is_number(value) for value in timings):
            raise ValueError("the reported timings must be finite numbers")
        if not isinstance(stdout, str) or not isinstance(stderr, str):
            raise ValueError("stdout and stderr must be strings")
        if not isinstance(stdout_truncated, bool) or not isinstance(stderr_truncated, bool):
            raise ValueError("output truncation flags must be booleans")

        used_parts: set[str] = set()
        # A FAILED program still reports every return that ran before the failure.
        encoded_results = body["results"]
        if not isinstance(encoded_results, dict):
            raise ValueError("results must be an object")
        results = {
            key: _decode_value(value, binary_parts, used_parts)
            for key, value in encoded_results.items()
            if isinstance(key, str)
        }
        if len(results) != len(encoded_results):
            raise ValueError("result keys must be strings")

        if status == "COMPLETED":
            if "error" in body:
                raise ValueError("COMPLETED response must not contain error details")
            error = None
        else:
            error = _parse_error(body["error"])
        unreferenced = set(binary_parts) - used_parts
        if unreferenced:
            raise ValueError(f"unreferenced binary response parts: {sorted(unreferenced)}")
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"malformed execution response: {exc}") from exc
    return ProgramResult(
        status=status,
        request_id=request_id,
        queue_ms=float(queue_ms),
        elapsed_ms=float(elapsed_ms),
        lease_wait_ms=float(lease_wait_ms),
        lease_held_ms=float(lease_held_ms),
        results=results,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        error=error,
    )


def _parse_error(error: Any) -> dict[str, Any]:
    if not isinstance(error, dict):
        raise ValueError("error must be an object")
    expected = {
        "kind",
        "message",
        "instruction_index",
        "instruction_op",
        "instruction_id",
        "traceback",
    }
    if error.get("kind") == "gpu_access":
        expected |= {"cuda_call", "location", "interfered_request_id"}
    if set(error) != expected:
        raise ValueError("error details have unexpected fields")
    if error["kind"] not in {
        "parse",
        "compile",
        "runtime",
        "gpu_access",
        "correctness",
        "serialization",
        "unavailable",
        "engine",
    }:
        raise ValueError("error kind is invalid")
    if error["kind"] == "gpu_access" and not (
        isinstance(error["cuda_call"], str)
        and isinstance(error["location"], str)
        and (
            error["interfered_request_id"] is None
            or isinstance(error["interfered_request_id"], str)
        )
    ):
        raise ValueError("gpu_access error details have the wrong types")
    if not isinstance(error["message"], str) or not isinstance(error["traceback"], str):
        raise ValueError("error message and traceback must be strings")
    if isinstance(error["instruction_index"], bool) or not isinstance(
        error["instruction_index"], int
    ):
        raise ValueError("error instruction_index must be an integer")
    if error["instruction_op"] not in ("upload", "get_function", "run", "return"):
        raise ValueError("error instruction_op is invalid")
    if error["instruction_id"] is not None and not isinstance(error["instruction_id"], str):
        raise ValueError("error instruction_id must be a string or null")
    return error


def _decode_value(encoded: Any, binary_parts: dict[str, bytes], used_parts: set[str]) -> Any:
    if not isinstance(encoded, dict) or not isinstance(encoded.get("type"), str):
        raise ValueError("encoded values must be objects with a type")
    value_type = encoded["type"]
    if value_type == "null":
        _expect_fields(encoded, {"type"})
        return None
    if value_type == "boolean":
        _expect_fields(encoded, {"type", "value"})
        if not isinstance(encoded["value"], bool):
            raise ValueError("boolean value has the wrong type")
        return encoded["value"]
    if value_type == "integer":
        _expect_fields(encoded, {"type", "value"})
        if isinstance(encoded["value"], bool) or not isinstance(encoded["value"], int):
            raise ValueError("integer value has the wrong type")
        return encoded["value"]
    if value_type == "number":
        _expect_fields(encoded, {"type", "value"})
        if not _is_number(encoded["value"]):
            raise ValueError("number value has the wrong type")
        return float(encoded["value"])
    if value_type == "string":
        _expect_fields(encoded, {"type", "value"})
        if not isinstance(encoded["value"], str):
            raise ValueError("string value has the wrong type")
        return encoded["value"]
    if value_type == "array":
        _expect_fields(encoded, {"type", "value"})
        if not isinstance(encoded["value"], list):
            raise ValueError("array value has the wrong type")
        return [_decode_value(child, binary_parts, used_parts) for child in encoded["value"]]
    if value_type == "object":
        _expect_fields(encoded, {"type", "value"})
        if not isinstance(encoded["value"], dict):
            raise ValueError("object value has the wrong type")
        return {
            key: _decode_value(child, binary_parts, used_parts)
            for key, child in encoded["value"].items()
        }
    if value_type == "bytes":
        _expect_fields(encoded, {"type", "part", "sha256"})
        return _binary_part(encoded, binary_parts, used_parts)
    if value_type == "file":
        _expect_fields(encoded, {"type", "size", "part", "sha256"})
        size = encoded["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("file size must be a non-negative integer")
        data = _binary_part(encoded, binary_parts, used_parts)
        if len(data) != size:
            raise ValueError("file binary length does not match size")
        return ReturnedFile(data)
    if value_type == "folder":
        _expect_fields(encoded, {"type", "files", "directories"})
        files, directories = encoded["files"], encoded["directories"]
        if not isinstance(files, dict) or not isinstance(directories, list):
            raise ValueError("folder files/directories have the wrong type")
        validate_manifest(files, directories)
        if any(
            not isinstance(child, dict) or child.get("type") != "file" for child in files.values()
        ):
            raise ValueError("folder files must contain file values")
        return ReturnedFolder(
            {path: _decode_value(child, binary_parts, used_parts) for path, child in files.items()},
            tuple(directories),
        )
    if value_type == "tensor":
        _expect_fields(encoded, {"type", "dtype", "shape", "part", "sha256"})
        dtype = encoded["dtype"]
        shape = encoded["shape"]
        if (
            dtype not in DTYPE_ITEM_SIZES
            or not isinstance(shape, list)
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
                for dimension in shape
            )
        ):
            raise ValueError("tensor metadata is invalid")
        data = _binary_part(encoded, binary_parts, used_parts)
        if len(data) != expected_tensor_nbytes(dtype, shape):
            raise ValueError("tensor binary length does not match dtype and shape")
        return _decode_tensor(dtype, shape, data)
    raise ValueError(f"unknown encoded value type: {value_type!r}")


def _binary_part(
    encoded: dict[str, Any], binary_parts: dict[str, bytes], used_parts: set[str]
) -> bytes:
    part_name = encoded["part"]
    blob_hash = encoded["sha256"]
    if not isinstance(part_name, str) or not part_name.startswith("return:"):
        raise ValueError("binary value has an invalid part name")
    part_index = part_name.removeprefix("return:")
    if not part_index.isdigit() or part_name != f"return:{int(part_index)}":
        raise ValueError("binary value has an invalid part index")
    if part_name != f"return:{len(used_parts)}":
        raise ValueError("binary values are not numbered in depth-first order")
    if part_name in used_parts:
        raise ValueError(f"binary response part is referenced more than once: {part_name!r}")
    if not is_blob_hash(blob_hash):
        raise ValueError("binary value has an invalid SHA-256 digest")
    try:
        data = binary_parts[part_name]
    except KeyError as exc:
        raise ValueError(f"missing binary response part: {part_name!r}") from exc
    try:
        verify_blob(blob_hash, data)
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    used_parts.add(part_name)
    return data


# Every protocol dtype as a numpy dtype. Explicit '<' pins the little-endian wire
# layout independently of host endianness; the ml_dtypes entries (numpy has no
# native scalar type for them) come only in native order, so those assume a
# little-endian host, as does every target the server runs on.
_NUMPY_DTYPES = {
    "bool": np.dtype("bool"),
    "uint8": np.dtype("uint8"),
    "int8": np.dtype("int8"),
    "int16": np.dtype("<i2"),
    "int32": np.dtype("<i4"),
    "int64": np.dtype("<i8"),
    "float16": np.dtype("<f2"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "float8_e4m3fn": np.dtype(ml_dtypes.float8_e4m3fn),
    "float8_e5m2": np.dtype(ml_dtypes.float8_e5m2),
}


def _decode_tensor(dtype: str, shape: list[int], data: bytes) -> Any:
    try:
        numpy_dtype = _NUMPY_DTYPES[dtype]
    except KeyError:
        raise ValueError(f"cannot decode tensor dtype {dtype!r}") from None
    # frombuffer aliases the read-only response bytes; copy() gives the caller a
    # writable array that owns its storage and outlives the response.
    return np.frombuffer(data, dtype=numpy_dtype).reshape(shape).copy()


def _expect_fields(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError(
            f"encoded {value.get('type')!r} value has fields {sorted(value)}, "
            f"expected {sorted(expected)}"
        )


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
