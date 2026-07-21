from __future__ import annotations

import json
import math
import posixpath
import re
from collections.abc import Mapping
from typing import Any

from .models import Entry, ServerConfig, ValidatedJob


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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
        raise ValueError(
            "blob hash must contain exactly 64 lowercase hexadecimal characters"
        )
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


def validate_job(raw: Any, config: ServerConfig) -> ValidatedJob:
    if not isinstance(raw, dict):
        raise ValueError("job must be a JSON object")
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

    timeout = raw.get("timeout_seconds", config.default_timeout_seconds)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout_seconds must be a positive number")
    timeout = float(timeout)
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or timeout > config.max_timeout_seconds
    ):
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
    return ValidatedJob(
        language=language,
        entry=Entry(entry_file, function),
        files=manifest,
        timeout_seconds=timeout,
        stdout_limit_bytes=stdout_limit,
        stderr_limit_bytes=stderr_limit,
    )


def _validate_limit(value: Any, name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > maximum
    ):
        raise ValueError(
            f"{name} must be a non-negative integer no greater than {maximum}"
        )
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
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
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
