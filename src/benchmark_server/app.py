"""HTTP front-end for ``POST /execute`` and ``GET /health``."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .cache import ByteCache
from .config import ServerConfig
from .errors import ValidationError
from .events import EventLogger
from .keys import is_blob_hash, verify_blob
from .multipart import MultipartPart, encode_multipart, parse_multipart
from .pool import PoolBusy, WorkerPool
from .schemas import (
    Program,
    ProgramOutcome,
    expected_tensor_nbytes,
    parse_program,
    strict_json_loads,
)
from .worker import WorkerCrashed, WorkerTimeout


class _CacheMiss(Exception):
    def __init__(self, blobs: list[str]):
        self.blobs = blobs


def create_app(
    config: ServerConfig | None = None,
    *,
    runtime_factory: Callable,
) -> FastAPI:
    """Build an application whose workers construct ``runtime_factory()``."""
    config = config or ServerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cache = ByteCache(config.cache_capacity_bytes)
        app.state.pool = WorkerPool(
            config.gpus, runtime_factory, config.worker_termination_grace_seconds
        )
        app.state.events = EventLogger(config.log_dir)
        app.state.events.emit("server_started", gpus=list(config.gpus))
        try:
            yield
        finally:
            app.state.pool.shutdown()
            app.state.events.emit("server_stopped")
            app.state.events.close()

    app = FastAPI(title="Benchmark Server", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict[str, object]:
        pool_health = request.app.state.pool.health()
        return {"status": "ok", "gpu_count": len(pool_health["workers"]), **pool_health}

    @app.post("/execute")
    async def execute_request(request: Request):
        request_id = str(uuid.uuid4())
        headers = {"X-Request-ID": request_id}
        events: EventLogger = request.app.state.events
        events.emit("request_started", request_id=request_id)

        def finished(http_status: int, *, level: str = "INFO", **fields: object) -> None:
            events.emit(
                "request_finished",
                level=level,
                request_id=request_id,
                http_status=http_status,
                **fields,
            )

        declared_length = request.headers.get("content-length", "")
        if declared_length.isdigit() and int(declared_length) > config.max_request_bytes:
            finished(413, level="WARNING", error="request_too_large")
            return _error_response(
                413, "request_too_large", "request exceeds the configured size limit", request_id
            )
        body_bytes = await request.body()
        if len(body_bytes) > config.max_request_bytes:
            finished(413, level="WARNING", error="request_too_large")
            return _error_response(
                413, "request_too_large", "request exceeds the configured size limit", request_id
            )

        cache: ByteCache = request.app.state.cache
        try:
            program, cache_keys = _parse_execute_request(
                request.headers.get("content-type"), body_bytes, cache
            )
        except ValidationError as exc:
            finished(400, level="WARNING", error="invalid_request")
            return _error_response(400, "parse", str(exc), request_id)
        except _CacheMiss as miss:
            finished(200, status="CACHE_MISS")
            return JSONResponse(
                {
                    "status": "CACHE_MISS",
                    "request_id": request_id,
                    "missing_blobs": miss.blobs,
                },
                headers=headers,
            )

        timeout = _resolve_timeout(program, config)
        program.options["output_limit_bytes"] = _resolve_output_limit(program, config)
        cache.pin(cache_keys)
        loop = asyncio.get_running_loop()
        try:
            outcome = await loop.run_in_executor(
                None,
                request.app.state.pool.submit,
                program,
                timeout,
                config.worker_wait_timeout_seconds,
            )
        except PoolBusy as exc:
            finished(503, level="WARNING", error="busy", queue_ms=exc.queue_ms)
            response = _error_response(503, "busy", "server saturated", request_id)
            response.headers["Retry-After"] = "1"
            return response
        except WorkerTimeout as exc:
            events.emit(
                "worker_restarted", request_id=request_id, gpu_id=exc.gpu_id, reason="timeout"
            )
            finished(
                504,
                level="WARNING",
                error="timeout",
                gpu_id=exc.gpu_id,
                queue_ms=exc.queue_ms,
                elapsed_ms=exc.elapsed_ms,
            )
            return _error_response(504, "timeout", "execution timed out", request_id)
        except WorkerCrashed as exc:
            events.emit(
                "worker_restarted", request_id=request_id, gpu_id=exc.gpu_id, reason="crash"
            )
            finished(
                500,
                level="ERROR",
                error="worker_crashed",
                gpu_id=exc.gpu_id,
                queue_ms=exc.queue_ms,
                elapsed_ms=exc.elapsed_ms,
            )
            return _error_response(500, "engine", "worker crashed", request_id)
        finally:
            cache.unpin(cache_keys)

        execution = outcome.execution
        if not isinstance(execution, ProgramOutcome):
            finished(500, level="ERROR", error="invalid_worker_response")
            return _error_response(500, "engine", "invalid worker response", request_id)
        payload: dict[str, object] = {
            "status": execution.status,
            "request_id": request_id,
            "queue_ms": outcome.queue_ms,
            "elapsed_ms": outcome.elapsed_ms,
            "stdout": execution.stdout,
            "stderr": execution.stderr,
            "stdout_truncated": execution.stdout_truncated,
            "stderr_truncated": execution.stderr_truncated,
        }
        payload["results"] = execution.results
        if execution.status != "COMPLETED":
            payload["error"] = execution.error

        response_body, content_type = _encode_response(payload, execution.binary_parts)
        if len(response_body) > config.max_response_bytes:
            finished(
                500,
                level="ERROR",
                error="response_too_large",
                gpu_id=outcome.gpu_id,
                queue_ms=outcome.queue_ms,
                elapsed_ms=outcome.elapsed_ms,
            )
            return _error_response(
                500,
                "response_too_large",
                "results exceed the configured response-size limit",
                request_id,
            )

        finished(
            200,
            status=execution.status,
            gpu_id=outcome.gpu_id,
            queue_ms=outcome.queue_ms,
            elapsed_ms=outcome.elapsed_ms,
        )
        response_headers = {**headers, "Content-Type": content_type}
        return Response(content=response_body, status_code=200, headers=response_headers)

    return app


def _parse_execute_request(
    content_type: str | None, body: bytes, cache: ByteCache
) -> tuple[Program, list[str]]:
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
    uploads = program.blob_uploads()
    referenced_blobs = _dedup([upload.blob for upload in uploads if upload.blob is not None])
    unreferenced = set(supplied_blobs) - set(referenced_blobs)
    if unreferenced:
        raise ValidationError(f"unreferenced blob part(s): {', '.join(sorted(unreferenced))}")

    for blob_hash, data in supplied_blobs.items():
        cache.put(blob_hash, data)

    missing: list[str] = []
    for upload in uploads:
        assert upload.blob is not None
        data = supplied_blobs.get(upload.blob)
        if data is None:
            data = cache.get(upload.blob)
        if data is None:
            missing.append(upload.blob)
            continue
        if upload.kind == "tensor":
            assert upload.dtype is not None and upload.shape is not None
            expected_size = expected_tensor_nbytes(upload.dtype, upload.shape)
            if len(data) != expected_size:
                raise ValidationError(
                    f"tensor upload {upload.id!r} expects {expected_size} bytes, got {len(data)}"
                )
        program.blob_bytes[upload.blob] = data
    if missing:
        raise _CacheMiss(_dedup(missing))
    return program, referenced_blobs


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


def _resolve_timeout(program: Program, config: ServerConfig) -> float:
    return min(
        float(program.options.get("timeout_seconds", config.default_timeout_seconds)),
        config.max_timeout_seconds,
    )


def _resolve_output_limit(program: Program, config: ServerConfig) -> int:
    return min(
        int(program.options.get("output_limit_bytes", config.output_limit_bytes)),
        config.max_output_limit_bytes,
    )


def _error_response(status: int, kind: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ERROR",
            "request_id": request_id,
            "error": {"kind": kind, "message": message},
        },
        status_code=status,
        headers={"X-Request-ID": request_id},
    )
