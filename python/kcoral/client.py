"""Synchronous Python client for the multipart execution protocol."""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import ml_dtypes
import numpy as np

from .keys import compute_blob_hash, is_blob_hash, verify_blob
from .multipart import parse_multipart
from .schemas import (
    DTYPE_ITEM_SIZES,
    _add_file_path,
    _add_file_paths,
    expected_tensor_nbytes,
    normalize_file_path,
    strict_json_loads,
)


class KCoralError(Exception):
    """A non-200 response from the server."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        kind: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.kind = kind
        self.request_id = request_id


class TransportError(Exception):
    """The request did not produce an HTTP response."""


class ProtocolError(Exception):
    """The server response does not follow the protocol."""


@dataclass(frozen=True)
class Register:
    id: str


@dataclass
class Program:
    _instructions: list[dict[str, Any]] = field(default_factory=list, init=False)
    _blobs: dict[str, bytes] = field(default_factory=dict, init=False)
    _ids: set[str] = field(default_factory=set, init=False)
    _return_keys: set[str] = field(default_factory=set, init=False)
    _file_paths: set[str] = field(default_factory=set, init=False)

    @property
    def instructions(self) -> list[dict[str, Any]]:
        return list(self._instructions)

    def upload_folder(self, folder: str | os.PathLike[str], *, path: str) -> None:
        """Snapshot a directory as ordinary file uploads at this program position.

        Includes hidden files; empty directories and original file metadata are
        not uploaded. Symbolic links, special files, and repeated directories
        are rejected. Failed calls leave the program unchanged.
        """
        destination = normalize_file_path(path)
        instructions = []
        blobs = {}
        for remote, data in _folder_files(folder, destination):
            digest = compute_blob_hash(data)
            blobs.setdefault(digest, data)
            instructions.append({"op": "upload", "kind": "file", "blob": digest, "path": remote})
        # One batch validation avoids a quadratic scan for folders with many
        # files and commits no state until traversal and validation both succeed.
        _add_file_paths([item["path"] for item in instructions], self._file_paths)
        self._instructions.extend(sorted(instructions, key=lambda item: item["path"]))
        for digest, data in blobs.items():
            self._blobs.setdefault(digest, data)

    def upload(
        self,
        *,
        id: str | None = None,
        kind: str,
        source: str | None = None,
        language: str = "python",
        value: Any = None,
        dtype: str | None = None,
        shape: list[int] | None = None,
        blob: Any = None,
        path: str | None = None,
    ) -> Register | None:
        if kind == "file":
            if id is not None:
                raise TypeError("file upload does not accept 'id'")
            if source is not None or value is not None or dtype is not None or shape is not None:
                raise TypeError("file upload accepts only 'blob' and 'path'")
            if language != "python":
                raise TypeError("file upload does not accept 'language'")
            normalized_path = normalize_file_path(path)
            try:
                raw = blob if isinstance(blob, bytes) else bytes(memoryview(blob))
            except TypeError as exc:
                raise TypeError("file upload requires a bytes-like 'blob'") from exc
            _add_file_path(normalized_path, self._file_paths)
            blob_hash = compute_blob_hash(raw)
            self._blobs.setdefault(blob_hash, raw)
            self._instructions.append(
                {"op": "upload", "kind": "file", "blob": blob_hash, "path": normalized_path}
            )
            return None

        if blob is not None or path is not None:
            raise TypeError(f"{kind} upload does not accept file fields")
        if kind == "module":
            if not isinstance(source, str):
                raise TypeError("module upload requires string 'source'")
            if value is not None or dtype is not None or shape is not None:
                raise TypeError("module upload does not accept tensor fields")
            if language not in ("python", "cuda"):
                raise ValueError("module upload 'language' must be 'python' or 'cuda'")
            instruction = {"op": "upload", "id": id, "kind": "module", "source": source}
            if language != "python":
                instruction["language"] = language
        elif kind == "tensor":
            if source is not None:
                raise TypeError("tensor upload does not accept 'source'")
            if language != "python":
                raise TypeError("tensor upload does not accept 'language'")
            tensor_dtype, tensor_shape, raw = _tensor_fields(value, dtype=dtype, shape=shape)
            blob_hash = compute_blob_hash(raw)
            self._blobs.setdefault(blob_hash, raw)
            instruction = {
                "op": "upload",
                "id": id,
                "kind": "tensor",
                "blob": blob_hash,
                "dtype": tensor_dtype,
                "shape": tensor_shape,
            }
        elif kind == "bytes":
            if source is not None:
                raise TypeError("bytes upload does not accept 'source'")
            if dtype is not None or shape is not None:
                raise TypeError("bytes upload does not accept tensor fields")
            if language != "python":
                raise TypeError("bytes upload does not accept 'language'")
            try:
                raw = value if isinstance(value, bytes) else bytes(memoryview(value))
            except TypeError as exc:
                raise TypeError("bytes upload requires a bytes-like 'value'") from exc
            blob_hash = compute_blob_hash(raw)
            self._blobs.setdefault(blob_hash, raw)
            instruction = {
                "op": "upload",
                "id": id,
                "kind": "bytes",
                "blob": blob_hash,
            }
        elif kind == "library":
            if source is not None:
                raise TypeError("library upload does not accept 'source'")
            if dtype is not None or shape is not None:
                raise TypeError("library upload does not accept tensor fields")
            raw = value if isinstance(value, bytes) else bytes(memoryview(value))
            blob_hash = compute_blob_hash(raw)
            self._blobs.setdefault(blob_hash, raw)
            instruction = {
                "op": "upload",
                "id": id,
                "kind": "library",
                "blob": blob_hash,
            }
        else:
            raise ValueError(
                "upload kind must be 'module', 'tensor', 'bytes', 'library', or 'file'"
            )
        self._add_id(id)
        self._instructions.append(instruction)
        return Register(id)

    def get_function(
        self,
        *,
        id: str,
        module: Register | dict[str, str],
        name: str,
        cpu_only: bool = False,
    ) -> Register:
        reference = _reference(module) if isinstance(module, Register) else module
        if not (
            isinstance(reference, dict)
            and set(reference) == {"$ref"}
            and isinstance(reference["$ref"], str)
        ):
            raise TypeError("module must be a Register or {'$ref': id}")
        if reference["$ref"] not in self._ids:
            raise ValueError(f"get_function {id!r} references unknown handle {reference['$ref']!r}")
        if not isinstance(name, str) or not name:
            raise ValueError("function name must be a non-empty string")
        if not isinstance(cpu_only, bool):
            raise TypeError("cpu_only must be a bool")
        instruction = {"op": "get_function", "id": id, "module": reference, "name": name}
        if cpu_only:
            instruction["cpu_only"] = True
        self._add_id(id)
        self._instructions.append(instruction)
        return Register(id)

    def run(
        self, *, id: str, fn: str | Register | dict[str, str], args: list[Any] | None = None
    ) -> Register:
        self._add_id(id)
        wire_fn: Any = _reference(fn) if isinstance(fn, Register) else fn
        wire_args = [
            _reference(argument) if isinstance(argument, Register) else argument
            for argument in (args or [])
        ]
        self._instructions.append({"op": "run", "id": id, "fn": wire_fn, "args": wire_args})
        return Register(id)

    def return_(self, *, key: str, value: Register | dict[str, str]) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("return key must be a non-empty string")
        if key in self._return_keys:
            raise ValueError(f"duplicate return key: {key!r}")
        reference = _reference(value) if isinstance(value, Register) else value
        if not (
            isinstance(reference, dict)
            and set(reference) == {"$ref"}
            and isinstance(reference["$ref"], str)
        ):
            raise TypeError("return value must be a Register or {'$ref': id}")
        if reference["$ref"] not in self._ids:
            raise ValueError(f"return {key!r} references unknown handle {reference['$ref']!r}")
        self._return_keys.add(key)
        self._instructions.append({"op": "return", "key": key, "value": reference})

    def _add_id(self, instruction_id: str | None) -> None:
        if not isinstance(instruction_id, str) or not instruction_id:
            raise ValueError("instruction id must be a non-empty string")
        if instruction_id in self._ids:
            raise ValueError(f"duplicate instruction id: {instruction_id!r}")
        self._ids.add(instruction_id)


@dataclass
class ProgramResult:
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
        return self.status == "COMPLETED"

    def __getitem__(self, key: str) -> Any:
        return self.results[key]


class Client:
    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        connect_timeout_seconds: float = 10.0,
    ) -> None:
        timeout = httpx.Timeout(None, connect=connect_timeout_seconds)
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def execute(
        self,
        program: Program,
        *,
        timeout_seconds: float | None = None,
        output_limit_bytes: int | None = None,
    ) -> ProgramResult:
        if not isinstance(program, Program):
            raise TypeError("execute expects a Program")
        options: dict[str, Any] = {}
        if timeout_seconds is not None:
            options["timeout_seconds"] = timeout_seconds
        if output_limit_bytes is not None:
            options["output_limit_bytes"] = output_limit_bytes

        body, binary_parts = self._post_program(program, options, include_blobs=set())
        if body.get("status") == "CACHE_MISS":
            if binary_parts:
                raise ProtocolError("CACHE_MISS response must not contain binary parts")
            missing = _parse_missing_blobs(body)
            unavailable = missing - set(program._blobs)
            if unavailable:
                raise ProtocolError(
                    f"server misses blobs with no local bytes to send: {sorted(unavailable)}"
                )
            body, binary_parts = self._post_program(program, options, include_blobs=missing)
            if body.get("status") == "CACHE_MISS":
                body, binary_parts = self._post_program(
                    program, options, include_blobs=set(program._blobs)
                )
            if body.get("status") == "CACHE_MISS":
                raise ProtocolError("server still reports CACHE_MISS after a complete blob resend")
        return _parse_program_result(body, binary_parts)

    def health(self) -> dict[str, Any]:
        response = self._request("GET", "/health")
        if response.status_code != 200:
            raise _server_error(response)
        body = _json_body(response)
        if body.get("status") != "ok":
            raise ProtocolError("health status is not ok")
        return body

    def target(self) -> dict[str, str]:
        """What an uploaded library must be built for, e.g. ``{"arch": "sm_100a"}``."""
        target = self.health().get("target")
        if not isinstance(target, dict) or "arch" not in target:
            raise ProtocolError("the server reported no compilation target")
        return target

    def _post_program(
        self, program: Program, options: dict[str, Any], include_blobs: set[str]
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        payload: dict[str, Any] = {"instructions": program.instructions}
        if options:
            payload["options"] = options
        files: list[tuple[str, tuple[None, bytes, str]]] = [
            (
                "program",
                (
                    None,
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8"),
                    "application/json",
                ),
            )
        ]
        files.extend(
            (f"blob:{blob_hash}", (None, data, "application/octet-stream"))
            for blob_hash, data in program._blobs.items()
            if blob_hash in include_blobs
        )
        response = self._request("POST", "/execute", files=files)
        if response.status_code != 200:
            raise _server_error(response)
        return _response_body(response)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise TransportError(str(exc)) from exc


def _folder_files(folder: str | os.PathLike[str], destination: str) -> Iterator[tuple[str, bytes]]:
    """Yield snapshots of regular files, never following links or recursing in Python.

    Directory descriptors anchor each descent even if a local path is replaced
    during traversal. Only the active ancestry stays open, so a wide tree does
    not exhaust descriptors. File reads are bounded by their initial size.
    """
    source = Path(folder)
    if source.is_symlink():
        raise ValueError(f"upload_folder rejects symbolic links: {source}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    seen = set()
    stack = []

    def enter(fd: int, remote: str) -> None:
        try:
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
            if identity in seen:
                raise ValueError(f"upload_folder encountered a repeated directory: {remote}")
            seen.add(identity)
            stack.append((fd, os.scandir(fd), remote))
        except BaseException:
            os.close(fd)
            raise

    enter(os.open(source, flags), destination)
    try:
        while stack:
            parent_fd, entries, remote = stack[-1]
            entry = next(entries, None)
            if entry is None:
                entries.close()
                os.close(parent_fd)
                stack.pop()
                continue
            path = normalize_file_path(f"{remote}/{entry.name}")
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode):
                enter(os.open(entry.name, flags, dir_fd=parent_fd), path)
            elif stat.S_ISREG(mode):
                fd = os.open(
                    entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd
                )
                with os.fdopen(fd, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode):
                        raise ValueError(f"upload_folder requires a regular file: {path}")
                    data = stream.read(before.st_size)
                    after = os.fstat(stream.fileno())
                    if (
                        len(data) != before.st_size
                        or after.st_size != before.st_size
                        or after.st_mtime_ns != before.st_mtime_ns
                        or after.st_ctime_ns != before.st_ctime_ns
                    ):
                        raise ValueError(f"upload_folder file changed while reading: {path}")
                yield path, data
            else:
                raise ValueError(f"upload_folder rejects symbolic links and special files: {path}")
    finally:
        for fd, entries, _ in reversed(stack):
            entries.close()
            os.close(fd)


def _reference(register: Register) -> dict[str, str]:
    return {"$ref": register.id}


def _tensor_fields(
    value: Any, *, dtype: str | None, shape: list[int] | None
) -> tuple[str, list[int], bytes]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        if dtype is None or shape is None:
            raise TypeError("raw tensor bytes require 'dtype' and 'shape'")
        tensor_dtype, tensor_shape, raw = dtype, list(shape), bytes(value)
    else:
        tensor_dtype, tensor_shape, raw = _array_fields(value)
        if dtype is not None and dtype != tensor_dtype:
            raise ValueError(
                f"declared dtype {dtype!r} does not match value dtype {tensor_dtype!r}"
            )
        if shape is not None and list(shape) != tensor_shape:
            raise ValueError(
                f"declared shape {shape!r} does not match value shape {tensor_shape!r}"
            )
    if tensor_dtype not in DTYPE_ITEM_SIZES:
        raise ValueError(f"unsupported tensor dtype: {tensor_dtype!r}")
    try:
        expected_size = expected_tensor_nbytes(tensor_dtype, tensor_shape)
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    if len(raw) != expected_size:
        raise ValueError(f"tensor metadata expects {expected_size} bytes, got {len(raw)}")
    return tensor_dtype, tensor_shape, raw


def _array_fields(value: Any) -> tuple[str, list[int], bytes]:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            return (
                str(tensor.dtype).removeprefix("torch."),
                [int(dimension) for dimension in tensor.shape],
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
            )
    except ImportError:
        pass

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.byteorder == ">" or (
            array.dtype.byteorder == "=" and sys.byteorder == "big"
        ):
            array = array.astype(array.dtype.newbyteorder("<"))
        return array.dtype.name, [int(dimension) for dimension in array.shape], array.tobytes()
    if hasattr(value, "__dlpack__"):
        try:
            array = np.from_dlpack(value)
        except Exception:
            array = None
        if array is not None:
            return _array_fields(array)

    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        return (
            str(value.dtype),
            [int(dimension) for dimension in value.shape],
            value.tobytes(),
        )
    raise TypeError(f"cannot upload {type(value).__name__!r} as a tensor")


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


def _parse_missing_blobs(body: dict[str, Any]) -> set[str]:
    missing = body.get("missing_blobs")
    if not isinstance(missing, list) or not missing:
        raise ProtocolError("CACHE_MISS response needs a non-empty 'missing_blobs' array")
    if any(not is_blob_hash(blob_hash) for blob_hash in missing):
        raise ProtocolError("CACHE_MISS response contains an invalid blob hash")
    if len(set(missing)) != len(missing):
        raise ProtocolError("CACHE_MISS response contains duplicate blob hashes")
    return set(missing)


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


def _server_error(response: httpx.Response) -> KCoralError:
    try:
        body = _json_body(response)
    except ProtocolError:
        body = {}
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message", "server error"))
        kind = error.get("kind")
    else:
        message = str(error) if error else f"HTTP {response.status_code}"
        kind = None
    request_id = body.get("request_id") if isinstance(body, dict) else None
    return KCoralError(response.status_code, message, kind=kind, request_id=request_id)
