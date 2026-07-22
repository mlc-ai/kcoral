from __future__ import annotations

import json
import math
import posixpath
import re
from collections.abc import Mapping
from typing import Any

from .models import (
    CallInstruction,
    Entry,
    RandomTensorInstruction,
    RegisterReference,
    ReturnInstruction,
    ServerConfig,
    UploadModuleInstruction,
    UploadTensorInstruction,
    ValidatedJob,
    ValidatedProgram,
    ValidatedRequest,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_INSTRUCTIONS = 10_000
MAX_REGISTER_INDEX = 2**31 - 1
MAX_NAME_LENGTH = 1_024
MAX_CLIENT_LIMIT = 2**63 - 1
TENSOR_DTYPE_SIZES = {
    "bool": 1,
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
}
TENSOR_DEVICES = {"cpu", "cuda:0"}
RANDOM_TENSOR_DTYPES = {"bfloat16", "float16", "float32", "float64"}


class DuplicateKeyError(ValueError):
    pass


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(data: bytes | str) -> Any:
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(
            data,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite number {value} is not allowed")
            ),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        DuplicateKeyError,
        ValueError,
    ) as exc:
        raise ValueError(f"malformed JSON: {exc}") from exc


def validate_hash(value: Any) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError("blob hash must contain exactly 64 lowercase hexadecimal characters")
    return value


def validate_remote_path(path: Any) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("file path must be a non-empty string")
    if "\x00" in path or "\\" in path or path.startswith("/"):
        raise ValueError(f"unsafe file path: {path!r}")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"unsafe file path: {path!r}")
    if posixpath.normpath(path) != path:
        raise ValueError(f"non-canonical file path: {path!r}")
    if parts[0] == ".benchmark-server":
        raise ValueError("file path uses a server-reserved name")
    return path


def validate_manifest_paths(paths: list[str]) -> None:
    path_set = set(paths)
    if len(path_set) != len(paths):
        raise ValueError("duplicate file path")
    for path in paths:
        segments = path.split("/")
        for index in range(1, len(segments)):
            if "/".join(segments[:index]) in path_set:
                raise ValueError(f"file/directory path collision involving {path!r}")


def validate_job(raw: Any, config: ServerConfig) -> ValidatedRequest:
    if not isinstance(raw, dict):
        raise ValueError("job must be a JSON object")
    if "instructions" in raw:
        return _validate_program(raw, config)
    allowed = {
        "language",
        "entry",
        "files",
        "timeout_seconds",
        "stdout_limit_bytes",
        "stderr_limit_bytes",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown job field(s): {', '.join(sorted(unknown))}")
    if "files" not in raw or not isinstance(raw["files"], dict):
        raise ValueError("files is required and must be an object")

    language = raw.get("language", "python")
    if language != "python":
        raise ValueError("language must be 'python'")

    entry_raw = raw.get("entry", {})
    if not isinstance(entry_raw, dict):
        raise ValueError("entry must be an object")
    entry_unknown = set(entry_raw) - {"file", "function"}
    if entry_unknown:
        raise ValueError(f"unknown entry field(s): {', '.join(sorted(entry_unknown))}")
    entry_file = validate_remote_path(entry_raw.get("file", "main.py"))
    function = entry_raw.get("function", "main")
    if not isinstance(function, str) or not function.isidentifier():
        raise ValueError("entry.function must be a Python identifier")

    manifest: dict[str, str] = {}
    for remote_path, reference in raw["files"].items():
        path = validate_remote_path(remote_path)
        if not isinstance(reference, dict) or set(reference) != {"blob"}:
            raise ValueError(f"files[{remote_path!r}] must contain only a blob field")
        manifest[path] = validate_hash(reference["blob"])
    validate_manifest_paths(list(manifest))
    if entry_file not in manifest:
        raise ValueError("entry.file must be present in files")

    timeout, stdout_limit, stderr_limit = _validate_execution_limits(raw, config)
    return ValidatedJob(
        language=language,
        entry=Entry(entry_file, function),
        files=manifest,
        timeout_seconds=timeout,
        stdout_limit_bytes=stdout_limit,
        stderr_limit_bytes=stderr_limit,
    )


def _validate_program(raw: dict[str, Any], config: ServerConfig) -> ValidatedProgram:
    allowed = {
        "instructions",
        "timeout_seconds",
        "stdout_limit_bytes",
        "stderr_limit_bytes",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown instruction job field(s): {', '.join(sorted(unknown))}")
    raw_instructions = raw.get("instructions")
    if not isinstance(raw_instructions, list):
        raise ValueError("instructions is required and must be an array")
    if len(raw_instructions) > MAX_INSTRUCTIONS:
        raise ValueError(f"instructions cannot contain more than {MAX_INSTRUCTIONS} items")

    instructions = []
    blob_digests: set[str] = set()
    defined_registers: set[int] = set()
    return_keys: set[str] = set()
    for instruction_index, raw_instruction in enumerate(raw_instructions):
        path = f"instructions[{instruction_index}]"
        if not isinstance(raw_instruction, dict):
            raise ValueError(f"{path} must be an object")
        operation = raw_instruction.get("op")
        if operation == "upload_module":
            _require_instruction_fields(raw_instruction, {"op", "blob"}, path)
            blob = validate_hash(raw_instruction["blob"])
            blob_digests.add(blob)
            instructions.append(UploadModuleInstruction(blob))
        elif operation == "upload_tensor":
            _require_instruction_fields(
                raw_instruction,
                {"op", "dst", "blob", "shape", "dtype", "device"},
                path,
            )
            destination = _validate_register(raw_instruction["dst"], f"{path}.dst")
            blob = validate_hash(raw_instruction["blob"])
            shape = _validate_tensor_shape(raw_instruction["shape"], path)
            dtype = raw_instruction["dtype"]
            if not isinstance(dtype, str) or dtype not in TENSOR_DTYPE_SIZES:
                raise ValueError(f"{path}.dtype is not supported")
            device = raw_instruction["device"]
            if not isinstance(device, str) or device not in TENSOR_DEVICES:
                raise ValueError(f"{path}.device must be 'cpu' or 'cuda:0'")
            tensor_size = math.prod(shape) * TENSOR_DTYPE_SIZES[dtype]
            if tensor_size > config.max_binary_value_bytes:
                raise ValueError(f"{path} tensor exceeds the configured binary-size limit")
            blob_digests.add(blob)
            defined_registers.add(destination)
            instructions.append(UploadTensorInstruction(destination, blob, shape, dtype, device))
        elif operation == "random_tensor":
            _require_instruction_fields(
                raw_instruction,
                {"op", "dst", "shape", "dtype", "seed", "device"},
                path,
            )
            destination = _validate_register(raw_instruction["dst"], f"{path}.dst")
            shape = _validate_tensor_shape(raw_instruction["shape"], path)
            dtype = raw_instruction["dtype"]
            if not isinstance(dtype, str) or dtype not in RANDOM_TENSOR_DTYPES:
                raise ValueError(f"{path}.dtype is not supported for random_tensor")
            seed = raw_instruction["seed"]
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 2**63 - 1:
                raise ValueError(f"{path}.seed must be an integer between 0 and {2**63 - 1}")
            device = raw_instruction["device"]
            if not isinstance(device, str) or device not in TENSOR_DEVICES:
                raise ValueError(f"{path}.device must be 'cpu' or 'cuda:0'")
            tensor_size = math.prod(shape) * TENSOR_DTYPE_SIZES[dtype]
            if tensor_size > config.max_binary_value_bytes:
                raise ValueError(f"{path} tensor exceeds the configured binary-size limit")
            defined_registers.add(destination)
            instructions.append(RandomTensorInstruction(destination, shape, dtype, seed, device))
        elif operation == "call":
            _require_instruction_fields(raw_instruction, {"op", "dst", "func", "args"}, path)
            destination_raw = raw_instruction["dst"]
            destination = (
                None
                if destination_raw is None
                else _validate_register(destination_raw, f"{path}.dst")
            )
            function = raw_instruction["func"]
            if (
                not isinstance(function, str)
                or not function
                or len(function) > MAX_NAME_LENGTH
                or "\x00" in function
            ):
                raise ValueError(f"{path}.func must be a valid non-empty function name")
            raw_arguments = raw_instruction["args"]
            if not isinstance(raw_arguments, list):
                raise ValueError(f"{path}.args must be an array")
            arguments = tuple(
                _validate_operand(
                    argument,
                    defined_registers,
                    f"{path}.args[{argument_index}]",
                    0,
                    config.max_nesting_depth,
                )
                for argument_index, argument in enumerate(raw_arguments)
            )
            if destination is not None:
                defined_registers.add(destination)
            instructions.append(CallInstruction(destination, function, arguments))
        elif operation == "return":
            _require_instruction_fields(raw_instruction, {"op", "reg", "key"}, path)
            register = _validate_register(raw_instruction["reg"], f"{path}.reg")
            if register not in defined_registers:
                raise ValueError(f"{path}.reg references undefined register r{register}")
            key = raw_instruction["key"]
            if not isinstance(key, str) or not key or len(key) > MAX_NAME_LENGTH or "\x00" in key:
                raise ValueError(f"{path}.key must be a valid non-empty string")
            if key in return_keys:
                raise ValueError(f"{path}.key duplicates return key {key!r}")
            return_keys.add(key)
            instructions.append(ReturnInstruction(register, key))
        else:
            raise ValueError(f"{path}.op is unknown")

    timeout, stdout_limit, stderr_limit = _validate_execution_limits(raw, config)
    return ValidatedProgram(
        instructions=tuple(instructions),
        blob_digests=frozenset(blob_digests),
        timeout_seconds=timeout,
        stdout_limit_bytes=stdout_limit,
        stderr_limit_bytes=stderr_limit,
    )


def _require_instruction_fields(instruction: dict[str, Any], expected: set[str], path: str) -> None:
    missing = expected - set(instruction)
    unknown = set(instruction) - expected
    if missing:
        raise ValueError(f"{path} is missing field(s): {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{path} has unknown field(s): {', '.join(sorted(unknown))}")


def _validate_register(value: Any, path: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > MAX_REGISTER_INDEX
    ):
        raise ValueError(f"{path} must be an integer between 0 and {MAX_REGISTER_INDEX}")
    return value


def _validate_tensor_shape(value: Any, path: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path}.shape must be an array")
    shape = []
    for dimension_index, dimension in enumerate(value):
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            or dimension > 2**63 - 1
        ):
            raise ValueError(f"{path}.shape[{dimension_index}] must be a non-negative integer")
        shape.append(dimension)
    return tuple(shape)


def _validate_operand(
    value: Any,
    defined_registers: set[int],
    path: str,
    depth: int,
    maximum_depth: int,
) -> Any:
    if depth > maximum_depth:
        raise ValueError(f"{path} exceeds the maximum nesting depth")
    if isinstance(value, dict):
        if set(value) != {"reg"}:
            raise ValueError(f"{path} object must contain only a reg field")
        register = _validate_register(value["reg"], f"{path}.reg")
        if register not in defined_registers:
            raise ValueError(f"{path} references undefined register r{register}")
        return RegisterReference(register)
    if isinstance(value, list):
        return [
            _validate_operand(
                item,
                defined_registers,
                f"{path}[{index}]",
                depth + 1,
                maximum_depth,
            )
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if value < -(2**63) or value > 2**63 - 1:
            raise ValueError(f"{path} integer is outside the signed 64-bit range")
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{path} is not a supported operand")


def _validate_execution_limits(
    raw: Mapping[str, Any], config: ServerConfig
) -> tuple[float, int, int]:
    timeout = raw.get("timeout_seconds", config.default_timeout_seconds)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout_seconds must be a positive number")
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0 or timeout > config.max_timeout_seconds:
        raise ValueError("timeout_seconds is outside the allowed range")

    stdout_limit = _validate_limit(
        raw.get("stdout_limit_bytes", config.default_stdout_limit_bytes),
        "stdout_limit_bytes",
        config.max_stdout_limit_bytes,
    )
    stderr_limit = _validate_limit(
        raw.get("stderr_limit_bytes", config.default_stderr_limit_bytes),
        "stderr_limit_bytes",
        config.max_stderr_limit_bytes,
    )
    return timeout, stdout_limit, stderr_limit


def _validate_limit(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > maximum:
        raise ValueError(f"{name} must be a non-negative integer no greater than {maximum}")
    return value


def make_job_payload(
    manifest: Mapping[str, str],
    language: str,
    entry: Entry,
    timeout_seconds: float | None,
    stdout_limit_bytes: int | None,
    stderr_limit_bytes: int | None,
) -> dict[str, Any]:
    if language != "python":
        raise ValueError("language must be 'python'")
    entry_file = validate_remote_path(entry.file)
    if not isinstance(entry.function, str) or not entry.function.isidentifier():
        raise ValueError("entry.function must be a Python identifier")
    paths = [validate_remote_path(path) for path in manifest]
    validate_manifest_paths(paths)
    clean_manifest = {path: validate_hash(digest) for path, digest in manifest.items()}
    if entry_file not in clean_manifest:
        raise ValueError("entry.file must be present in files")
    payload: dict[str, Any] = {
        "language": language,
        "entry": {"file": entry_file, "function": entry.function},
        "files": {path: {"blob": digest} for path, digest in clean_manifest.items()},
    }
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("timeout_seconds must be a positive number")
        value = float(timeout_seconds)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        payload["timeout_seconds"] = timeout_seconds
    for name, value in (
        ("stdout_limit_bytes", stdout_limit_bytes),
        ("stderr_limit_bytes", stderr_limit_bytes),
    ):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            payload[name] = value
    return payload


def make_program_payload(
    instructions: list[Mapping[str, Any]],
    timeout_seconds: float | None,
    stdout_limit_bytes: int | None,
    stderr_limit_bytes: int | None,
) -> dict[str, Any]:
    if not isinstance(instructions, list) or any(
        not isinstance(instruction, Mapping) for instruction in instructions
    ):
        raise ValueError("instructions must be an array of objects")
    payload: dict[str, Any] = {"instructions": [dict(instruction) for instruction in instructions]}
    if timeout_seconds is not None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("timeout_seconds must be a positive number")
        value = float(timeout_seconds)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        payload["timeout_seconds"] = timeout_seconds
    for name, value in (
        ("stdout_limit_bytes", stdout_limit_bytes),
        ("stderr_limit_bytes", stderr_limit_bytes),
    ):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            payload[name] = value
    return payload


def validate_program_payload(raw: dict[str, Any]) -> ValidatedProgram:
    config = ServerConfig(
        max_timeout_seconds=float("inf"),
        max_stdout_limit_bytes=MAX_CLIENT_LIMIT,
        max_stderr_limit_bytes=MAX_CLIENT_LIMIT,
        max_binary_value_bytes=MAX_CLIENT_LIMIT,
    )
    return _validate_program(raw, config)
