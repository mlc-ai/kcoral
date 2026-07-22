"""HTTP front-end: `POST /benchmark` (synchronous, stateless) + `GET /health`.

The front-end touches no GPU. It validates the program, resolves each upload's
bytes (from the shared cache or inline), dispatches the program to a GPU worker
under a deadline, and returns the per-instruction results. `CACHE_MISS` is caught
here before any worker is touched.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .cache import ByteCache
from .config import ServerConfig
from .errors import ValidationError
from .keys import verify_key
from .pool import PoolBusy, WorkerPool
from .schemas import Program, parse_program
from .worker import WorkerCrashed, WorkerTimeout


class _CacheMiss(Exception):
    def __init__(self, keys: list[str]):
        self.keys = keys


def create_app(
    config: Optional[ServerConfig] = None,
    *,
    runtime_factory: Callable,
) -> FastAPI:
    """Build the app. ``runtime_factory`` is a picklable, no-arg callable that
    each worker process calls to construct its Runtime."""
    config = config or ServerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cache = ByteCache(config.cache_capacity_bytes)
        app.state.pool = WorkerPool(config.gpus, runtime_factory)
        try:
            yield
        finally:
            app.state.pool.shutdown()

    app = FastAPI(title="Benchmark Server", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/benchmark")
    async def benchmark(request: Request):
        try:
            body = await request.json()
        except Exception:
            return _error(400, "request body must be valid JSON")

        cache: ByteCache = request.app.state.cache
        try:
            program = parse_program(body)
            keys = _resolve_uploads(program, cache)
        except ValidationError as exc:
            return _error(400, str(exc))
        except _CacheMiss as miss:
            return JSONResponse({"status": "CACHE_MISS", "missing_keys": miss.keys})

        timeout = _resolve_timeout(program, config)
        cache.pin(keys)
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None,
                request.app.state.pool.submit,
                program,
                timeout,
                config.worker_wait_timeout_seconds,
            )
        except PoolBusy:
            return JSONResponse(
                {"error": "server saturated"}, status_code=503, headers={"Retry-After": "1"}
            )
        except WorkerTimeout:
            return JSONResponse(
                {"status": "ERROR", "error": {"kind": "timeout", "message": "execution timed out"}},
                status_code=504,
            )
        except WorkerCrashed:
            return JSONResponse(
                {"status": "ERROR", "error": {"kind": "engine", "message": "worker crashed"}},
                status_code=500,
            )
        finally:
            cache.unpin(keys)

        # HTTP 200 means "we ran your program"; the body status reflects whether
        # every instruction succeeded (a FAILED instruction skips the rest).
        status = "FAILED" if any(r.status == "FAILED" for r in results) else "COMPLETED"
        return {"status": status, "results": [_result_dict(r) for r in results]}

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
            cache.put(up.key, verify_key(up.key, up.kind, up.inline))
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


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


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
    return d
