from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path
from typing import Any

import httpx

from ._dlpack import tensor_from_bytes
from .models import (
    BenchmarkServerError,
    Entry,
    ExecutionResult,
    ExecutionWarning,
    FileContent,
    Health,
    PreparedFiles,
    ProtocolError,
    TransportError,
    WorkerHealth,
)
from .validation import (
    make_job_payload,
    make_program_payload,
    strict_json_loads,
    validate_hash,
    validate_manifest_paths,
    validate_program_payload,
    validate_remote_path,
)


class Client:
    def __init__(
        self,
        base_url: str,
        *,
        headers: Mapping[str, str] | None = None,
        connect_timeout_seconds: float = 10,
    ) -> None:
        if connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be positive")
        timeout = httpx.Timeout(None, connect=connect_timeout_seconds)
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def prepare_files(self, files: Mapping[str, FileContent]) -> PreparedFiles:
        manifest, content = _encode_files(files)
        self._prepare(content)
        return PreparedFiles(manifest)

    def execute(
        self,
        files: Mapping[str, FileContent],
        *,
        language: str = "python",
        entry: Entry = Entry(),
        timeout_seconds: float | None = None,
        stdout_limit_bytes: int | None = None,
        stderr_limit_bytes: int | None = None,
    ) -> ExecutionResult:
        manifest, content = _encode_files(files)
        job = make_job_payload(
            manifest,
            language,
            entry,
            timeout_seconds,
            stdout_limit_bytes,
            stderr_limit_bytes,
        )
        return self._execute_with_content(job, content)

    def execute_instructions(
        self,
        instructions: list[Mapping[str, Any]],
        blobs: Mapping[str, FileContent],
        *,
        timeout_seconds: float | None = None,
        stdout_limit_bytes: int | None = None,
        stderr_limit_bytes: int | None = None,
    ) -> ExecutionResult:
        job = make_program_payload(
            instructions,
            timeout_seconds,
            stdout_limit_bytes,
            stderr_limit_bytes,
        )
        validated_program = validate_program_payload(job)
        content = _encode_blobs(blobs)
        unreferenced = sorted(set(content) - set(validated_program.blob_digests))
        if unreferenced:
            raise ValueError(
                "blobs contains hashes not referenced by instructions: " + ", ".join(unreferenced)
            )
        return self._execute_with_content(job, content)

    def execute_prepared(
        self,
        prepared: PreparedFiles,
        *,
        language: str = "python",
        entry: Entry = Entry(),
        timeout_seconds: float | None = None,
        stdout_limit_bytes: int | None = None,
        stderr_limit_bytes: int | None = None,
    ) -> ExecutionResult:
        if not isinstance(prepared, PreparedFiles):
            raise ValueError("prepared must be a PreparedFiles instance")
        job = make_job_payload(
            prepared.manifest,
            language,
            entry,
            timeout_seconds,
            stdout_limit_bytes,
            stderr_limit_bytes,
        )
        return _decode_execution_response(self._post_execute(job, {}))

    def _execute_with_content(
        self, job: Mapping[str, Any], content: Mapping[str, bytes]
    ) -> ExecutionResult:
        self._prepare(content)
        try:
            response = self._post_execute(job, {})
            return _decode_execution_response(response)
        except BenchmarkServerError as exc:
            if exc.code != "blob_not_found" or not exc.missing_blobs:
                raise
            missing: dict[str, bytes] = {}
            for digest in exc.missing_blobs:
                if digest not in content:
                    raise ProtocolError(
                        "server requested a missing blob not referenced by this execution"
                    ) from exc
                missing[digest] = content[digest]
            response = self._post_execute(job, missing)
            return _decode_execution_response(response)

    def health(self) -> Health:
        response = self._request("GET", "/health")
        if response.status_code != 200:
            raise _decode_server_error(response, require_request_id=False)
        raw = _json_response(response)
        try:
            if raw.get("status") != "ok":
                raise ValueError("status is not ok")
            workers = tuple(
                WorkerHealth(
                    int(item["gpu_id"]),
                    str(item["status"]),
                    float(item["uptime_seconds"]),
                )
                for item in raw["workers"]
            )
            health = Health(
                str(raw["status"]),
                int(raw["gpu_count"]),
                int(raw["queue_length"]),
                workers,
            )
            if health.gpu_count != len(workers) or health.queue_length < 0:
                raise ValueError("inconsistent health fields")
            return health
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"malformed health response: {exc}") from exc

    def _prepare(self, content: Mapping[str, bytes]) -> None:
        response = self._request("POST", "/blobs/check", json={"blobs": list(content)})
        if response.status_code != 200:
            raise _decode_server_error(response, require_request_id=False)
        raw = _json_response(response)
        try:
            missing_values = raw["missing"]
            if not isinstance(missing_values, list):
                raise ValueError("missing is not an array")
            missing = [validate_hash(value) for value in missing_values]
            if len(set(missing)) != len(missing) or any(
                digest not in content for digest in missing
            ):
                raise ValueError("missing contains duplicate or unrequested hashes")
        except (KeyError, ValueError) as exc:
            raise ProtocolError(f"malformed blob-check response: {exc}") from exc
        if not missing:
            return
        files = [
            (f"blob:{digest}", (digest, content[digest], "application/octet-stream"))
            for digest in missing
        ]
        upload = self._request("POST", "/blobs", files=files)
        if upload.status_code != 200:
            raise _decode_server_error(upload, require_request_id=False)
        raw_upload = _json_response(upload)
        try:
            stored = [validate_hash(value) for value in raw_upload["stored"]]
            already = [validate_hash(value) for value in raw_upload["already_present"]]
            if (
                raw_upload.get("status") != "ok"
                or len(set(stored + already)) != len(stored) + len(already)
                or set(stored + already) != set(missing)
            ):
                raise ValueError("stored hashes do not match the upload")
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"malformed blob-upload response: {exc}") from exc

    def _post_execute(self, job: Mapping[str, Any], inline: Mapping[str, bytes]) -> httpx.Response:
        files: list[tuple[str, tuple[str | None, bytes, str]]] = [
            (
                "job",
                (
                    None,
                    json.dumps(job, allow_nan=False, separators=(",", ":")).encode(),
                    "application/json",
                ),
            )
        ]
        files.extend(
            (f"blob:{digest}", (digest, data, "application/octet-stream"))
            for digest, data in inline.items()
        )
        response = self._request("POST", "/execute", files=files)
        if response.status_code != 200:
            raise _decode_server_error(response, require_request_id=True)
        return response

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise TransportError(str(exc)) from exc


def _encode_files(
    files: Mapping[str, FileContent],
) -> tuple[dict[str, str], dict[str, bytes]]:
    if not isinstance(files, Mapping):
        raise ValueError("files must be a mapping")
    paths: list[str] = []
    manifest: dict[str, str] = {}
    content: dict[str, bytes] = {}
    for remote_path, value in files.items():
        path = validate_remote_path(remote_path)
        paths.append(path)
        data = _read_content(value, path)
        digest = hashlib.sha256(data).hexdigest()
        manifest[path] = digest
        content.setdefault(digest, data)
    validate_manifest_paths(paths)
    return manifest, content


def _encode_blobs(blobs: Mapping[str, FileContent]) -> dict[str, bytes]:
    if not isinstance(blobs, Mapping):
        raise ValueError("blobs must be a mapping")
    content: dict[str, bytes] = {}
    for declared_digest, value in blobs.items():
        digest = validate_hash(declared_digest)
        data = _read_content(value, digest)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"blob content does not match declared hash {digest}")
        content[digest] = data
    return content


def _read_content(value: FileContent, name: str) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, Path):
        return value.read_bytes()
    raise ValueError(f"unsupported content for {name!r}")


def _decode_execution_response(response: httpx.Response) -> ExecutionResult:
    header_id = response.headers.get("x-request-id")
    _validate_request_id(header_id)
    content_type = response.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    binary_parts: dict[str, bytes] = {}
    if media_type == "application/json":
        raw = _json_response(response)
    elif media_type == "multipart/form-data":
        raw, binary_parts = _parse_multipart_response(response.content, content_type)
    else:
        raise ProtocolError(f"unexpected execution response content type {content_type!r}")
    try:
        if raw.get("status") != "ok":
            raise ValueError("status is not ok")
        body_id = raw["request_id"]
        _validate_request_id(body_id)
        if body_id != header_id:
            raise ValueError("request identifier does not match response header")
        referenced: set[str] = set()
        value = _decode_node(raw["return"], binary_parts, referenced, "$", 0)
        if referenced != set(binary_parts):
            raise ValueError("multipart response contains unreferenced binary parts")
        warnings = tuple(
            ExecutionWarning(
                str(item["code"]),
                tuple(validate_hash(value) for value in item.get("blobs", [])),
            )
            for item in raw.get("warnings", [])
        )
        return ExecutionResult(
            request_id=body_id,
            value=value,
            elapsed_ms=float(raw["elapsed_ms"]),
            queue_ms=float(raw["queue_ms"]),
            stdout=str(raw["stdout"]),
            stderr=str(raw["stderr"]),
            stdout_truncated=bool(raw.get("stdout_truncated", False)),
            stderr_truncated=bool(raw.get("stderr_truncated", False)),
            warnings=warnings,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError(f"malformed execution response: {exc}") from exc


def _decode_node(
    node: Any,
    binary_parts: Mapping[str, bytes],
    referenced: set[str],
    path: str,
    depth: int,
) -> Any:
    if depth > 256 or not isinstance(node, dict) or not isinstance(node.get("type"), str):
        raise ValueError(f"invalid return-value node at {path}")
    node_type = node["type"]
    if node_type == "json":
        value = node.get("value")
        json.dumps(value, allow_nan=False)
        return value
    if node_type in {"bytes", "tensor"}:
        part_name = node.get("part")
        if not isinstance(part_name, str) or part_name not in binary_parts:
            raise ValueError(f"missing binary part at {path}")
        data = binary_parts[part_name]
        size = node.get("size")
        digest = node.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size != len(data):
            raise ValueError(f"binary size mismatch at {path}")
        validate_hash(digest)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"binary hash mismatch at {path}")
        referenced.add(part_name)
        if node_type == "bytes":
            return data
        return _decode_tensor(node, data, path)
    if node_type in {"list", "tuple"}:
        items = node.get("items")
        if not isinstance(items, list):
            raise ValueError(f"items must be an array at {path}")
        decoded = [
            _decode_node(item, binary_parts, referenced, f"{path}[{index}]", depth + 1)
            for index, item in enumerate(items)
        ]
        return tuple(decoded) if node_type == "tuple" else decoded
    if node_type == "dict":
        items = node.get("items")
        if not isinstance(items, dict) or any(not isinstance(key, str) for key in items):
            raise ValueError(f"items must be an object at {path}")
        return {
            key: _decode_node(value, binary_parts, referenced, f"{path}.{key}", depth + 1)
            for key, value in items.items()
        }
    raise ValueError(f"unknown return-value node type {node_type!r} at {path}")


def _decode_tensor(node: Mapping[str, Any], data: bytes, path: str) -> Any:
    try:
        import tvm_ffi

        dtype_name = node["dtype"]
        shape = node["shape"]
        if not isinstance(dtype_name, str) or not isinstance(shape, list):
            raise ValueError("dtype or shape has the wrong type")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape):
            raise ValueError("shape contains an invalid dimension")
        ffi_dtype = tvm_ffi.dtype(dtype_name)
        expected = math.prod(shape) * int(ffi_dtype.itemsize)
        if expected != len(data):
            raise ValueError("tensor size does not match shape and dtype")
        return tensor_from_bytes(data, dtype_name, tuple(shape))
    except Exception as exc:
        raise ValueError(f"invalid tensor metadata at {path}: {exc}") from exc


def _parse_multipart_response(
    content: bytes, content_type: str
) -> tuple[dict[str, Any], dict[str, bytes]]:
    message = BytesParser(policy=email_policy).parsebytes(
        f"MIME-Version: 1.0\r\nContent-Type: {content_type}\r\n\r\n".encode("ascii") + content
    )
    if not message.is_multipart():
        raise ProtocolError("malformed multipart execution response")
    metadata: dict[str, Any] | None = None
    binaries: dict[str, bytes] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        part_type = part.get_content_type().lower()
        data = part.get_payload(decode=True)
        if not isinstance(name, str) or data is None:
            raise ProtocolError("malformed multipart response part")
        if name == "result":
            if metadata is not None or part_type != "application/json":
                raise ProtocolError("duplicate or malformed result part")
            try:
                value = strict_json_loads(data)
            except ValueError as exc:
                raise ProtocolError(str(exc)) from exc
            if not isinstance(value, dict):
                raise ProtocolError("result part must contain a JSON object")
            metadata = value
        else:
            if (
                not name.startswith("return:")
                or part_type != "application/octet-stream"
                or name in binaries
            ):
                raise ProtocolError("duplicate or malformed binary response part")
            binaries[name] = data
    if metadata is None:
        raise ProtocolError("multipart response is missing the result part")
    return metadata, binaries


def _decode_server_error(
    response: httpx.Response, *, require_request_id: bool
) -> BenchmarkServerError:
    raw = _json_response(response)
    try:
        if (
            raw.get("status") != "error"
            or not isinstance(raw["error"], str)
            or not isinstance(raw["message"], str)
        ):
            raise ValueError("missing structured error fields")
        request_id = raw.get("request_id")
        if require_request_id:
            header_id = response.headers.get("x-request-id")
            _validate_request_id(header_id)
            _validate_request_id(request_id)
            if request_id != header_id:
                raise ValueError("request identifier mismatch")
        elif request_id is not None:
            _validate_request_id(request_id)
        missing = tuple(validate_hash(value) for value in raw.get("missing_blobs", []))
        instruction_index = raw.get("instruction_index")
        if instruction_index is not None and (
            isinstance(instruction_index, bool)
            or not isinstance(instruction_index, int)
            or instruction_index < 0
        ):
            raise ValueError("instruction_index must be a non-negative integer")
        return BenchmarkServerError(
            response.status_code,
            raw["error"],
            raw["message"],
            request_id=request_id,
            stdout=str(raw.get("stdout", "")),
            stderr=str(raw.get("stderr", "")),
            traceback=raw.get("traceback"),
            missing_blobs=missing,
            instruction_index=instruction_index,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"malformed server error response: {exc}") from exc


def _json_response(response: httpx.Response) -> dict[str, Any]:
    if (
        response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        != "application/json"
    ):
        raise ProtocolError("response content type is not application/json")
    try:
        raw = strict_json_loads(response.content)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc
    if not isinstance(raw, dict):
        raise ProtocolError("JSON response must be an object")
    return raw


def _validate_request_id(value: Any) -> None:
    if not isinstance(value, str):
        raise ProtocolError("missing request identifier")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ProtocolError("invalid request identifier") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ProtocolError("request identifier is not a canonical UUID v4")
