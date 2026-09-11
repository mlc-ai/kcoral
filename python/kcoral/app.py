"""HTTP front-end for ``POST /execute`` and ``GET /health``."""

from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .cache import ByteCache, DiskFileCache
from .config import ServerConfig
from .errors import ValidationError
from .events import EventLogger
from .keys import is_blob_hash, verify_blob
from .multipart import MultipartPart, encode_multipart, parse_multipart
from .pool import PoolBusy, SubmitOutcome, WorkerPool
from .schemas import (
    FileUpload,
    Program,
    ProgramOutcome,
    Run,
    Upload,
    expected_tensor_nbytes,
    parse_program,
    strict_json_loads,
)
from .worker import FINISH_REASON_LEVEL, WorkerCrashed, WorkerTimeout

_TRACEBACK_LIMIT = 8192
_MESSAGE_LIMIT = 2048

# The worker's finish reasons, plus the ones the front-end can answer with before
# any worker is involved.
_FINISH_REASON_LEVEL = {
    **FINISH_REASON_LEVEL,
    "no_worker": "WARNING",
    "rejected": "WARNING",
    "cache_miss": "INFO",
    "server_error": "ERROR",
}


class _CacheMiss(Exception):
    def __init__(self, blobs: list[str]):
        self.blobs = blobs


def create_app(
    config: ServerConfig | None = None,
    *,
    runtime_factory: Callable | None = None,
) -> FastAPI:
    """Build an application, optionally overriding its mode-specific runtime."""
    config = config or ServerConfig()
    if config.device not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {config.device!r}")
    if config.disk_cache_capacity_mbytes < 0:
        raise ValueError("disk cache capacity must be non-negative")
    worker_gpus = config.gpus if config.device == "gpu" else []
    cpu_workers = config.num_workers if config.device == "cpu" else None
    worker_count = (
        cpu_workers if cpu_workers is not None else len(worker_gpus) * config.workers_per_gpu
    )
    if runtime_factory is None:
        if config.device == "cpu":
            from .cpu_runtime import cpu_runtime_factory

            runtime_factory = cpu_runtime_factory
        else:
            from .gpu_runtime import gpu_runtime_factory

            runtime_factory = gpu_runtime_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Before the pool, not after: bringing workers up is where a deployment
        # fails first, and a log opened afterwards would never record it.
        events = EventLogger(config.log_dir, console=config.log_console)
        app.state.events = events
        events.emit(
            "server_started",
            run_dir=str(events.run_dir) if events.run_dir else None,
            config=_describe(config),
        )
        app.state.cache = ByteCache(config.cache_capacity_bytes)
        app.state.file_cache = DiskFileCache(
            config.disk_cache_dir, config.disk_cache_capacity_mbytes * 1024**2
        )
        try:
            app.state.pool = WorkerPool(
                worker_gpus,
                runtime_factory,
                termination_grace_seconds=config.worker_termination_grace_seconds,
                workers_per_gpu=config.workers_per_gpu,
                max_requests_per_worker=config.max_requests_per_worker,
                cpu_workers=cpu_workers,
                events=events,
            )
        except BaseException as exc:
            events.emit(
                "server_start_failed",
                level="ERROR",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc()[-_TRACEBACK_LIMIT:],
            )
            events.close()
            raise
        events.emit(
            "pool_ready",
            mode=config.device,
            target=app.state.pool.target(),
            versions=app.state.pool.versions(),
            workers=worker_count,
        )
        try:
            yield
        finally:
            cleanup = asyncio.create_task(app.state.pool.shutdown_async())
            cancelled = False
            try:
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        cancelled = True
                cleanup.result()
                events.emit("server_stopped")
            finally:
                events.close()
            if cancelled:
                raise asyncio.CancelledError

    app = FastAPI(title="KCoral", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict[str, object]:
        pool_health = request.app.state.pool.health()
        return {"status": "ok", "gpu_count": len(pool_health["gpus"]), **pool_health}

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception):
        """Close out a request the endpoint could not, so a front-end bug leaves
        more than a ``request_received`` with no end."""
        request_id = getattr(request.state, "request_id", None)
        request.app.state.events.emit(
            "request_finished",
            level="ERROR",
            request_id=request_id,
            http_status=500,
            finish_reason="server_error",
            error_kind="unhandled",
            error_message=f"{type(exc).__name__}: {exc}"[:_MESSAGE_LIMIT],
            traceback=traceback.format_exc()[-_TRACEBACK_LIMIT:],
        )
        return _error_response(500, "engine", "internal server error", request_id or "")

    @app.post("/execute")
    async def execute_request(request: Request):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id  # so unhandled_error can name it
        headers = {"X-Request-ID": request_id}
        events: EventLogger = request.app.state.events
        client = request.client
        events.emit(
            "request_received",
            request_id=request_id,
            client=client.host if client else None,
            content_length=request.headers.get("content-length"),
        )

        def finished(http_status: int, *, finish_reason: str, **fields: object) -> None:
            events.emit(
                "request_finished",
                level=_FINISH_REASON_LEVEL.get(finish_reason, "INFO"),
                request_id=request_id,
                http_status=http_status,
                finish_reason=finish_reason,
                **fields,
            )

        declared_length = request.headers.get("content-length", "")
        if declared_length.isdigit() and int(declared_length) > config.max_request_bytes:
            finished(413, finish_reason="rejected", error_kind="request_too_large")
            return _error_response(
                413, "request_too_large", "request exceeds the configured size limit", request_id
            )
        body_bytes = await request.body()
        if len(body_bytes) > config.max_request_bytes:
            finished(413, finish_reason="rejected", error_kind="request_too_large")
            return _error_response(
                413, "request_too_large", "request exceeds the configured size limit", request_id
            )

        cache: ByteCache = request.app.state.cache
        try:
            program, cache_keys, program_bytes = _parse_execute_request(
                request.headers.get("content-type"), body_bytes, cache, request.app.state.file_cache
            )
        except ValidationError as exc:
            finished(
                400,
                finish_reason="rejected",
                error_kind="invalid_request",
                error_message=str(exc)[:_MESSAGE_LIMIT],
            )
            return _error_response(400, "parse", str(exc), request_id)
        except _CacheMiss as miss:
            finished(200, finish_reason="cache_miss", status="CACHE_MISS", missing=len(miss.blobs))
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
        if events.enabled:  # describing the workload is the one cost worth a branch
            events.emit(
                "request_accepted",
                request_id=request_id,
                request_bytes=len(body_bytes),
                timeout_seconds=timeout,
                program=_keep_program(events, request_id, program_bytes)
                if config.log_programs
                else None,
                **_program_shape(program),
            )
        cache.pin(cache_keys)
        # A crash the engine can attribute to an instruction still answers 200
        # with a FAILED program; these carry its cause onto that record.
        crash_exitcode: int | None = None
        crash_tail = ""
        loop = asyncio.get_running_loop()
        try:
            outcome = await loop.run_in_executor(
                None,
                request.app.state.pool.submit,
                program,
                timeout,
                config.worker_wait_timeout_seconds,
                request_id,
            )
        except PoolBusy as exc:
            finished(503, finish_reason="no_worker", queue_ms=exc.queue_ms)
            response = _error_response(503, "busy", "server saturated", request_id)
            response.headers["Retry-After"] = "1"
            return response
        except WorkerTimeout as exc:
            finished(
                504,
                finish_reason="timeout",
                worker_id=exc.worker_id,
                gpu_id=exc.gpu_id,
                queue_ms=exc.queue_ms,
                elapsed_ms=exc.elapsed_ms,
                timeout_seconds=timeout,
                output_tail=exc.output_tail or None,
            )
            return _error_response(504, "timeout", "execution timed out", request_id)
        except WorkerCrashed as exc:
            if exc.instruction_index is None or not 0 <= exc.instruction_index < len(
                program.instructions
            ):
                finished(
                    500,
                    finish_reason="crashed",
                    worker_id=exc.worker_id,
                    gpu_id=exc.gpu_id,
                    exitcode=exc.exitcode,
                    queue_ms=exc.queue_ms,
                    elapsed_ms=exc.elapsed_ms,
                    output_tail=exc.output_tail or None,
                )
                return _error_response(500, "engine", "worker crashed", request_id)
            instruction = program.instructions[exc.instruction_index]
            execution = ProgramOutcome(
                status="FAILED",
                error={
                    "kind": "runtime",
                    "message": "worker exited while executing the instruction",
                    "instruction_index": exc.instruction_index,
                    "instruction_op": instruction.op,
                    "instruction_id": getattr(instruction, "id", None),
                    "traceback": "",
                },
            )
            outcome = SubmitOutcome(
                execution=execution,
                gpu_id=exc.gpu_id,
                queue_ms=exc.queue_ms or 0.0,
                elapsed_ms=exc.elapsed_ms or 0.0,
                lease_wait_ms=exc.lease_wait_ms,
                lease_held_ms=exc.lease_held_ms,
                worker_id=exc.worker_id or "",
                finish_reason="crashed",
            )
            crash_exitcode, crash_tail = exc.exitcode, exc.output_tail
        finally:
            cache.unpin(cache_keys)

        execution = outcome.execution
        if not isinstance(execution, ProgramOutcome):
            finished(
                500,
                finish_reason="server_error",
                error_kind="invalid_worker_response",
                worker_id=outcome.worker_id,
                gpu_id=outcome.gpu_id,
            )
            return _error_response(500, "engine", "invalid worker response", request_id)
        if execution.error is not None and execution.error.get("kind") == "gpu_access":
            execution.error["interfered_request_id"] = outcome.interfered_request_id
            events.emit(
                "gpu_access_violation",
                level="WARNING",
                request_id=request_id,
                worker_id=outcome.worker_id,
                gpu_id=outcome.gpu_id,
                instruction_id=execution.error.get("instruction_id"),
                cuda_call=execution.error.get("cuda_call"),
                location=execution.error.get("location"),
                interfered_request_id=outcome.interfered_request_id,
            )
        payload: dict[str, object] = {
            "status": execution.status,
            "request_id": request_id,
            "queue_ms": outcome.queue_ms,
            "elapsed_ms": outcome.elapsed_ms,
            "lease_wait_ms": outcome.lease_wait_ms,
            "lease_held_ms": outcome.lease_held_ms,
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
                finish_reason="server_error",
                error_kind="response_too_large",
                response_bytes=len(response_body),
                worker_id=outcome.worker_id,
                gpu_id=outcome.gpu_id,
                queue_ms=outcome.queue_ms,
                elapsed_ms=outcome.elapsed_ms,
                lease_held_ms=outcome.lease_held_ms,
            )
            return _error_response(
                500,
                "response_too_large",
                "results exceed the configured response-size limit",
                request_id,
            )

        error = execution.error or {}
        finished(
            200,
            finish_reason=outcome.finish_reason,
            exitcode=crash_exitcode,
            output_tail=crash_tail or None,
            status=execution.status,
            worker_id=outcome.worker_id,
            gpu_id=outcome.gpu_id,
            error_kind=error.get("kind"),
            error_message=str(error.get("message", ""))[:_MESSAGE_LIMIT] or None,
            instruction_index=error.get("instruction_index"),
            instruction_op=error.get("instruction_op"),
            instruction_id=error.get("instruction_id"),
            queue_ms=outcome.queue_ms,
            elapsed_ms=outcome.elapsed_ms,
            lease_wait_ms=outcome.lease_wait_ms,
            lease_held_ms=outcome.lease_held_ms,
            response_bytes=len(response_body),
        )
        response_headers = {**headers, "Content-Type": content_type}
        return Response(content=response_body, status_code=200, headers=response_headers)

    return app


def _parse_execute_request(
    content_type: str | None, body: bytes, cache: ByteCache, file_cache: DiskFileCache
) -> tuple[Program, list[str], bytes]:
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

    file_keys = {upload.blob for upload in uploads if isinstance(upload, FileUpload)}
    memory_keys = {upload.blob for upload in uploads if not isinstance(upload, FileUpload)}
    for blob_hash, data in supplied_blobs.items():
        if blob_hash in memory_keys:
            cache.put(blob_hash, data)
    file_cache.put_many({key: data for key, data in supplied_blobs.items() if key in file_keys})

    missing: list[str] = []
    for blob_hash in referenced_blobs:
        data = supplied_blobs.get(blob_hash)
        if data is None and blob_hash in file_keys:
            data = file_cache.get(blob_hash)
        if data is None and blob_hash in memory_keys:
            data = cache.get(blob_hash)
        if data is None:
            missing.append(blob_hash)
        else:
            # Owned bytes survive disk eviction while queued or executing. If a
            # hash has both kinds of upload, the request can share these bytes.
            program.blob_bytes[blob_hash] = data
    for upload in uploads:
        assert upload.blob is not None
        data = program.blob_bytes.get(upload.blob)
        if data is None:
            continue
        if upload.kind == "tensor":
            assert upload.dtype is not None and upload.shape is not None
            expected_size = expected_tensor_nbytes(upload.dtype, upload.shape)
            if len(data) != expected_size:
                raise ValidationError(
                    f"tensor upload {upload.id!r} expects {expected_size} bytes, got {len(data)}"
                )
    if missing:
        raise _CacheMiss(missing)
    return program, [key for key in referenced_blobs if key in memory_keys], program_bytes


def _describe(config: ServerConfig) -> dict[str, object]:
    """The settings a run was started with, so a log explains its own behaviour."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _program_shape(program: Program) -> dict[str, object]:
    """The shape of the workload, for reading the log without opening the
    program it describes."""
    ops: Counter[str] = Counter()
    uploads: Counter[str] = Counter()
    builtins: set[str] = set()
    for instruction in program.instructions:
        ops[instruction.op] += 1
        if isinstance(instruction, (Upload, FileUpload)):
            kind = instruction.kind
            uploads[f"{kind}:{instruction.language}" if kind == "module" else kind] += 1
        elif isinstance(instruction, Run) and isinstance(instruction.fn, str):
            builtins.add(instruction.fn)
    return {
        "instructions": len(program.instructions),
        "ops": dict(ops),
        "uploads": dict(uploads) or None,
        "builtins": sorted(builtins) or None,
        "blob_bytes": sum(len(data) for data in program.blob_bytes.values()) or None,
    }


def _keep_program(events: EventLogger, request_id: str, program_bytes: bytes) -> str | None:
    """Write the program beside the log and answer with its name. The bytes
    arrived over the wire, so nothing is re-serialized."""
    directory = events.subdir("programs")
    if directory is None:
        return None
    name = f"{request_id}.json"
    try:
        (directory / name).write_bytes(program_bytes)
    except OSError:
        return None  # best-effort, like every other write the log makes
    return name


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
