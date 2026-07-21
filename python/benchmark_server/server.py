from __future__ import annotations

import json
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .cache import BlobCache
from .events import EventLogger
from .models import ServerConfig
from .multipart import MultipartBody, encode_multipart, parse_multipart_request
from .validation import strict_json_loads, validate_hash, validate_job
from .worker import Scheduler, run_job


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra: Any) -> None:
        self.status = status
        self.code = code
        self.message = message
        self.extra = extra
        super().__init__(message)


class ServerRuntime:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config.validated()
        self.cache = BlobCache(config.cache_dir, config.cache_capacity_bytes)
        self.scheduler = Scheduler(config.devices)
        self.events: EventLogger | None = None
        self.work_root = config.work_dir.resolve()

    async def start(self) -> None:
        self.work_root.mkdir(parents=True, exist_ok=True)
        for child in self.work_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        self.events = EventLogger(self.config.log_dir)
        await self.cache.start()
        self.events.emit("server_started", devices=list(self.config.devices))

    async def close(self) -> None:
        await self.cache.close()
        if self.events is not None:
            self.events.emit("server_stopped")
            self.events.close()


def create_app(config: ServerConfig | None = None) -> FastAPI:
    runtime = ServerRuntime(config or ServerConfig())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await runtime.start()
        app.state.runtime = runtime
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="Benchmark Server", version="2", lifespan=lifespan)
    app.state.runtime = runtime

    @app.post("/execute")
    async def execute(request: Request) -> Response:
        request_id = str(uuid.uuid4())
        headers = {"X-Request-ID": request_id}
        events = _events(runtime)
        stdout_path, stderr_path = events.request_paths(request_id)
        events.emit("request_started", request_id=request_id)
        body: MultipartBody | None = None
        acquired: set[str] = set()
        work_dir: Path | None = None
        slot = None
        assigned_gpu_id: int | None = None
        queue_ms: float | None = None
        elapsed_ms: float | None = None
        try:
            try:
                body = await parse_multipart_request(request, runtime.cache.tmp)
                job_parts = [part for part in body.parts if part.name == "job"]
                if len(job_parts) != 1:
                    raise ValueError("request must contain exactly one job part")
                job_part = job_parts[0]
                if job_part.content_type != "application/json":
                    raise ValueError("job part content type must be application/json")
                raw_job = strict_json_loads(
                    job_part.read(runtime.config.max_json_metadata_bytes)
                )
                job = validate_job(raw_job, runtime.config)

                inline: dict[str, Path] = {}
                for part in body.parts:
                    if part.name == "job":
                        continue
                    if not part.name.startswith("blob:"):
                        raise ValueError(f"unexpected multipart part {part.name!r}")
                    digest = validate_hash(part.name[5:])
                    if part.content_type != "application/octet-stream":
                        raise ValueError(
                            f"blob part {part.name!r} must use application/octet-stream"
                        )
                    if digest in inline:
                        raise ValueError(f"duplicate blob part {digest}")
                    inline[digest] = part.path
                manifest_digests = set(job.files.values())
                missing, unused = await runtime.cache.ingest_and_acquire(
                    manifest_digests, inline
                )
                if missing:
                    raise APIError(
                        404,
                        "blob_not_found",
                        "one or more referenced blobs are missing",
                        missing_blobs=missing,
                    )
                acquired = manifest_digests
            except APIError:
                raise
            except ValueError as exc:
                raise APIError(400, "invalid_request", str(exc)) from exc
            except Exception as exc:
                raise APIError(
                    500, "internal_error", f"could not parse or cache request: {exc}"
                ) from exc

            queued_at = time.perf_counter()
            slot = await runtime.scheduler.acquire()
            assigned_gpu_id = slot.index
            queue_ms = (time.perf_counter() - queued_at) * 1000
            work_dir = runtime.work_root / request_id
            try:
                await runtime.cache.materialize(job.files, work_dir)
            except Exception as exc:
                raise APIError(
                    500, "internal_error", f"could not materialize request files: {exc}"
                ) from exc
            result_dir = work_dir / ".benchmark-server" / "result"
            outcome = await run_job(
                slot,
                job,
                request_id,
                work_dir,
                result_dir,
                stdout_path,
                stderr_path,
                runtime.config,
            )
            elapsed_ms = outcome.metadata.get("elapsed_ms")
            restarted = outcome.kind in {"timeout", "crash"}
            await runtime.scheduler.release(slot, restarted=restarted)
            if restarted:
                events.emit(
                    "worker_restarted",
                    request_id=request_id,
                    gpu_id=slot.index,
                    reason=outcome.kind,
                )
            slot = None

            stdout, stdout_truncated = _read_output(stdout_path, job.stdout_limit_bytes)
            stderr, stderr_truncated = _read_output(stderr_path, job.stderr_limit_bytes)
            if outcome.kind != "ok":
                mapping = {
                    "execution_failed": (400, "execution_failed"),
                    "invalid_return_value": (400, "invalid_return_value"),
                    "timeout": (408, "timeout"),
                    "crash": (503, "unavailable"),
                }
                status, code = mapping.get(outcome.kind, (500, "internal_error"))
                extra: dict[str, Any] = {"stdout": stdout, "stderr": stderr}
                if outcome.metadata.get("traceback") is not None:
                    extra["traceback"] = outcome.metadata["traceback"]
                if stdout_truncated:
                    extra["stdout_truncated"] = True
                if stderr_truncated:
                    extra["stderr_truncated"] = True
                raise APIError(
                    status, code, outcome.metadata.get("message", code), **extra
                )

            metadata: dict[str, Any] = {
                "status": "ok",
                "request_id": request_id,
                "return": outcome.metadata["return"],
                "elapsed_ms": float(outcome.metadata["elapsed_ms"]),
                "queue_ms": queue_ms,
                "stdout": stdout,
                "stderr": stderr,
            }
            if stdout_truncated:
                metadata["stdout_truncated"] = True
            if stderr_truncated:
                metadata["stderr_truncated"] = True
            if unused:
                metadata["warnings"] = [{"code": "unused_blob", "blobs": unused}]
            metadata_bytes = json.dumps(
                metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            if len(metadata_bytes) > runtime.config.max_json_metadata_bytes:
                raise APIError(
                    400,
                    "invalid_return_value",
                    "response metadata exceeds the configured limit",
                )

            if not outcome.binaries:
                response = Response(
                    metadata_bytes, 200, headers=headers, media_type="application/json"
                )
            else:
                binary_payloads = [
                    (part.name, part.path.read_bytes()) for part in outcome.binaries
                ]
                response_body, boundary = encode_multipart(
                    metadata_bytes, binary_payloads
                )
                if len(response_body) > runtime.config.max_response_bytes:
                    raise APIError(
                        400,
                        "invalid_return_value",
                        "response exceeds the configured size limit",
                    )
                response = Response(
                    response_body,
                    200,
                    headers=headers,
                    media_type=f"multipart/form-data; boundary={boundary}",
                )
            events.emit(
                "request_finished",
                level="WARNING" if unused else "INFO",
                request_id=request_id,
                http_status=200,
                error=None,
                gpu_id=assigned_gpu_id,
                queue_ms=queue_ms,
                elapsed_ms=elapsed_ms,
                stdout_path=str(stdout_path.relative_to(events.run_dir)),
                stderr_path=str(stderr_path.relative_to(events.run_dir)),
                **({"warnings": ["unused_blob"]} if unused else {}),
            )
            return response
        except APIError as exc:
            payload = {
                "status": "error",
                "error": exc.code,
                "message": exc.message,
                "request_id": request_id,
                **exc.extra,
            }
            events.emit(
                "request_finished",
                level="WARNING" if exc.status < 500 else "ERROR",
                request_id=request_id,
                http_status=exc.status,
                error=exc.code,
                gpu_id=assigned_gpu_id,
                queue_ms=queue_ms,
                elapsed_ms=elapsed_ms,
                stdout_path=str(stdout_path.relative_to(events.run_dir)),
                stderr_path=str(stderr_path.relative_to(events.run_dir)),
            )
            return JSONResponse(payload, status_code=exc.status, headers=headers)
        except Exception as exc:
            events.emit(
                "server_error", level="ERROR", request_id=request_id, message=str(exc)
            )
            events.emit(
                "request_finished",
                level="ERROR",
                request_id=request_id,
                http_status=500,
                error="internal_error",
                gpu_id=assigned_gpu_id,
                queue_ms=queue_ms,
                elapsed_ms=elapsed_ms,
                stdout_path=str(stdout_path.relative_to(events.run_dir)),
                stderr_path=str(stderr_path.relative_to(events.run_dir)),
            )
            payload = {
                "status": "error",
                "error": "internal_error",
                "message": "an internal server error occurred",
                "request_id": request_id,
            }
            return JSONResponse(payload, status_code=500, headers=headers)
        finally:
            if slot is not None:
                await runtime.scheduler.release(slot)
            if work_dir is not None:
                shutil.rmtree(work_dir, ignore_errors=True)
            if acquired:
                await runtime.cache.release(acquired)
            if body is not None:
                body.close()

    @app.post("/blobs/check")
    async def blobs_check(request: Request) -> Response:
        try:
            if _base_content_type(request) != "application/json":
                raise ValueError("content type must be application/json")
            raw = strict_json_loads(await request.body())
            if (
                not isinstance(raw, dict)
                or set(raw) != {"blobs"}
                or not isinstance(raw["blobs"], list)
            ):
                raise ValueError("body must contain only a blobs array")
            digests = [validate_hash(value) for value in raw["blobs"]]
            if len(set(digests)) != len(digests):
                raise ValueError("blobs must not contain duplicate hashes")
            return JSONResponse({"missing": await runtime.cache.check(digests)})
        except ValueError as exc:
            return _simple_error(400, "invalid_request", str(exc))
        except Exception:
            return _simple_error(
                500, "internal_error", "could not inspect the blob cache"
            )

    @app.post("/blobs")
    async def blobs_upload(request: Request) -> Response:
        body: MultipartBody | None = None
        try:
            body = await parse_multipart_request(request, runtime.cache.tmp)
            if not body.parts:
                raise ValueError("at least one blob part is required")
            sources: dict[str, Path] = {}
            for part in body.parts:
                if not part.name.startswith("blob:"):
                    raise ValueError(f"unexpected multipart part {part.name!r}")
                digest = validate_hash(part.name[5:])
                if part.content_type != "application/octet-stream":
                    raise ValueError(
                        f"blob part {part.name!r} must use application/octet-stream"
                    )
                if digest in sources:
                    raise ValueError(f"duplicate blob part {digest}")
                sources[digest] = part.path
            stored, already = await runtime.cache.upload(sources)
            return JSONResponse(
                {"status": "ok", "stored": stored, "already_present": already}
            )
        except ValueError as exc:
            return _simple_error(400, "invalid_request", str(exc))
        except Exception:
            return _simple_error(
                500, "internal_error", "could not store uploaded content"
            )
        finally:
            if body is not None:
                body.close()

    @app.get("/health")
    async def health(request: Request) -> Response:
        if request.url.query or await request.body():
            return _simple_error(
                400, "invalid_request", "health accepts no body or query parameters"
            )
        workers = runtime.scheduler.health()
        healthy = [
            worker
            for worker in workers
            if worker["status"] in {"idle", "busy", "restarting"}
        ]
        if not healthy:
            return _simple_error(
                503, "unavailable", "no healthy GPU worker is available"
            )
        return JSONResponse(
            {
                "status": "ok",
                "gpu_count": len(workers),
                "queue_length": runtime.scheduler.queue_length,
                "workers": workers,
            }
        )

    return app


def _events(runtime: ServerRuntime) -> EventLogger:
    if runtime.events is None:
        raise RuntimeError("server has not started")
    return runtime.events


def _read_output(path: Path, limit: int) -> tuple[str, bool]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        data = stream.read(limit)
    return data.decode("utf-8", errors="replace"), size > limit


def _simple_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"status": "error", "error": code, "message": message}, status_code=status
    )


def _base_content_type(request: Request) -> str:
    return request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
