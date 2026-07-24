"""HTTP front-end: `POST /benchmark` (synchronous, stateless) + `GET /health`.

The front-end touches no GPU. It validates the program, resolves each upload's
bytes (from the shared cache or inline), dispatches the program to a GPU worker
under a deadline, and returns the per-instruction results. `CACHE_MISS` is caught
here before any worker is touched.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .cache import ByteCache
from .config import ServerConfig
from .errors import ValidationError
from .events import EventLogger
from .keys import verify_key
from .pool import PoolBusy, WorkerPool
from .schemas import Program, parse_program
from .worker import WorkerCrashed, WorkerTimeout


class _CacheMiss(Exception):
    def __init__(self, keys: list[str]):
        self.keys = keys


def create_app(
    config: ServerConfig | None = None,
    *,
    runtime_factory: Callable,
) -> FastAPI:
    """Build the app. ``runtime_factory`` is a picklable, no-arg callable that
    each worker process calls to construct its Runtime."""
    config = config or ServerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cache = ByteCache(config.cache_capacity_bytes, config.cache_dir)
        app.state.pool = WorkerPool(
            config.gpus, runtime_factory, config.worker_termination_grace_seconds
        )
        app.state.events = EventLogger(config.log_dir)
        app.state.events.emit("server_started", gpus=list(config.gpus))
        try:
            yield
        finally:
            app.state.pool.shutdown()
            app.state.cache.close()
            app.state.events.emit("server_stopped")
            app.state.events.close()

    app = FastAPI(title="Benchmark Server", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict:
        pool_health = request.app.state.pool.health()
        return {"status": "ok", "gpu_count": len(pool_health["workers"]), **pool_health}

    @app.post("/benchmark")
    async def benchmark(request: Request):
        request_id = str(uuid.uuid4())
        headers = {"X-Request-ID": request_id}
        events: EventLogger = request.app.state.events
        events.emit("request_started", request_id=request_id)

        def finished(http_status: int, *, level: str = "INFO", **fields) -> None:
            events.emit(
                "request_finished",
                level=level,
                request_id=request_id,
                http_status=http_status,
                **fields,
            )

        try:
            body = await request.json()
        except Exception:
            finished(400, level="WARNING", error="invalid_json")
            return _error(400, "request body must be valid JSON", request_id)

        cache: ByteCache = request.app.state.cache
        try:
            program = parse_program(body)
            keys = _resolve_uploads(program, cache)
        except ValidationError as exc:
            finished(400, level="WARNING", error="invalid_request")
            return _error(400, str(exc), request_id)
        except _CacheMiss as miss:
            finished(200, status="CACHE_MISS")
            return JSONResponse(
                {"status": "CACHE_MISS", "missing_keys": miss.keys, "request_id": request_id},
                headers=headers,
            )

        timeout = _resolve_timeout(program, config)
        program.options["output_limit_bytes"] = _resolve_output_limit(program, config)
        cache.pin(keys)
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
            return JSONResponse(
                {"error": "server saturated", "request_id": request_id},
                status_code=503,
                headers={"Retry-After": "1", **headers},
            )
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
            return JSONResponse(
                {
                    "status": "ERROR",
                    "error": {"kind": "timeout", "message": "execution timed out"},
                    "request_id": request_id,
                },
                status_code=504,
                headers=headers,
            )
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
            return JSONResponse(
                {
                    "status": "ERROR",
                    "error": {"kind": "engine", "message": "worker crashed"},
                    "request_id": request_id,
                },
                status_code=500,
                headers=headers,
            )
        finally:
            cache.unpin(keys)

        # HTTP 200 means "we ran your program"; the body status reflects whether
        # every instruction succeeded (a FAILED instruction skips the rest).
        status = "FAILED" if any(r.status == "FAILED" for r in outcome.results) else "COMPLETED"
        finished(
            200,
            status=status,
            gpu_id=outcome.gpu_id,
            queue_ms=outcome.queue_ms,
            elapsed_ms=outcome.elapsed_ms,
        )
        return JSONResponse(
            {
                "status": status,
                "request_id": request_id,
                "queue_ms": outcome.queue_ms,
                "elapsed_ms": outcome.elapsed_ms,
                "results": [_result_dict(r) for r in outcome.results],
            },
            headers=headers,
        )

    return app


def _resolve_uploads(program: Program, cache: ByteCache) -> list[str]:
    """Resolve each upload to its canonical bytes (verify+cache inline; else read
    the cache) and fill ``program.upload_bytes``. Raises :class:`_CacheMiss` (with
    the de-duplicated missing keys) if any referenced bytes are absent — before
    any worker runs. Returns the de-duplicated keys to pin.
    """
    missing: list[str] = []
    keys: list[str] = []
    for up in program.uploads():
        if up.inline is not None:
            # Use the verified inline bytes directly; caching them is a
            # best-effort optimization (the cache may decline, e.g. a blob
            # bigger than its per-object cap).
            data = verify_key(up.key, up.kind, up.inline)
            cache.put(up.key, data)
        else:
            data = cache.get(up.key)
        if data is None:
            missing.append(up.key)
        else:
            program.upload_bytes[up.id] = data
            keys.append(up.key)
    if missing:
        raise _CacheMiss(_dedup(missing))
    return _dedup(keys)


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]


def _resolve_timeout(program: Program, config: ServerConfig) -> float:
    requested = program.options.get("timeout_seconds")
    if requested is None:
        return config.default_timeout_seconds
    try:
        return min(float(requested), config.max_timeout_seconds)
    except (TypeError, ValueError):
        return config.default_timeout_seconds


def _resolve_output_limit(program: Program, config: ServerConfig) -> int:
    """Per-instruction stdout/stderr capture cap; a client may lower or disable
    (<= 0) it but cannot exceed the server maximum."""
    requested = program.options.get("output_limit_bytes")
    if requested is None:
        return config.output_limit_bytes
    try:
        return min(int(requested), config.max_output_limit_bytes)
    except (TypeError, ValueError):
        return config.output_limit_bytes


def _error(status: int, message: str, request_id: str | None = None) -> JSONResponse:
    body: dict = {"error": message}
    headers = None
    if request_id is not None:
        body["request_id"] = request_id
        headers = {"X-Request-ID": request_id}
    return JSONResponse(body, status_code=status, headers=headers)


def _result_dict(r) -> dict:
    d = {"id": r.id, "op": r.op, "status": r.status}
    if r.value is not None:
        d["value"] = r.value
    if r.error is not None:
        d["error"] = r.error
    if r.stdout:
        d["stdout"] = r.stdout
    if r.stderr:
        d["stderr"] = r.stderr
    if r.stdout_truncated:
        d["stdout_truncated"] = True
    if r.stderr_truncated:
        d["stderr_truncated"] = True
    return d
