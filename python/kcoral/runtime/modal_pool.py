"""Front-end admission and authenticated HTTP forwarding to a Modal server."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, is_dataclass

import httpx

from kcoral import __version__
from kcoral.client.result import _json_body, _parse_program_result, _response_body
from kcoral.errors import ProtocolError
from kcoral.runtime.pool import PoolBusy, SubmitOutcome
from kcoral.runtime.worker import WorkerCleanupError, WorkerCrashed, WorkerTimeout
from kcoral.schemas import ProgramOutcome, Ref

# The deployed Modal server's contract; modal_app.py configures it with these.
REMOTE_MAX_REQUEST_BYTES = 256 * 1024**2
REMOTE_MAX_RESPONSE_BYTES = 256 * 1024**2
REMOTE_MAX_TIMEOUT_SECONDS = 900.0
# Each request retires the remote worker, and a timeout or crash respawns it before the
# answer. Either costs a kill plus prepare and initialize waits of up to 60 s each.
REMOTE_RESPAWN_SECONDS = 180.0
_PART_OVERHEAD_BYTES = 512  # multipart boundary and part headers, rounded up
# Errors the Modal server returns after finishing a request cleanly.
_REQUEST_ERRORS = {(400, "parse"), (413, "request_too_large"), (500, "response_too_large")}


class ModalRejected(Exception):
    """The Modal server refused this request but stays usable for the next one."""

    def __init__(self, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status, self.kind, self.message = status, kind, message


class ModalPool:
    """One admitted execution on a single, authenticated Modal GPU server."""

    def __init__(self, url: str, max_response_bytes: int, *, http_client=None) -> None:
        endpoint = httpx.URL(url)
        if endpoint.scheme != "https" or endpoint.userinfo or endpoint.query or endpoint.fragment:
            raise ValueError("modal_url must be an HTTPS endpoint without credentials or query")
        if http_client is None:
            key = os.environ.get("KCORAL_MODAL_KEY")
            secret = os.environ.get("KCORAL_MODAL_SECRET")
            if not key or not secret:
                raise ValueError(
                    "the Modal backend requires KCORAL_MODAL_KEY and KCORAL_MODAL_SECRET"
                )
            http_client = httpx.Client(
                base_url=url.rstrip("/"),
                headers={"Modal-Key": key, "Modal-Secret": secret, "Accept-Encoding": "identity"},
                follow_redirects=False,
                trust_env=False,
            )
        self._http = http_client
        self._max_response_bytes = max_response_bytes
        self._slots = threading.BoundedSemaphore(1)
        self._closing = threading.Event()
        self._condition = threading.Condition()
        self._active = self._waiting = 0
        self._shutdown_done = False
        # The instance whose last outcome or cleanup is unknown, until the remote recovers.
        self._lost: str | None = None
        self._needs_replacement = False
        try:
            info = self._health()
            self._check_version()
            self._instance_id = info["instance_id"]
            self._target, self._versions = info["target"], info["versions"]
        except BaseException:
            self._http.close()
            raise

    def _health(self) -> dict:
        response = self._request("GET", "/health", budget=30)
        response.raise_for_status()
        info = _json_body(response)
        if info["status"] != "ok" or info["gpu_count"] != 1:
            raise ValueError("Modal server must have one ready GPU")
        if info["load"]["request_capacity"] < 1:
            raise ValueError("Modal server has no serviceable worker")
        instance_id, target = info["instance_id"], info["target"]
        if not isinstance(instance_id, str) or not instance_id or "arch" not in target:
            raise ValueError("Modal server has no instance identity or compilation target")
        for metadata in (target, info["versions"]):
            if not isinstance(metadata, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
            ):
                raise ValueError("invalid Modal runtime metadata")
        return info

    def _check_version(self) -> None:
        response = self._request("GET", "/openapi.json", budget=30)
        response.raise_for_status()
        remote = _json_body(response)["info"]["version"]
        if remote != __version__:
            raise ValueError(
                f"Modal server runs KCoral {remote} but this server is {__version__}; "
                "redeploy modal_app.py"
            )

    def _mark_lost(self, *, needs_replacement: bool) -> None:
        with self._condition:
            self._lost = self._instance_id
            self._needs_replacement = self._needs_replacement or needs_replacement

    def _recover(self, queue_ms: float) -> None:
        """Resume once the remote is idle, and replaced if its cleanup was unconfirmed.

        The lost request is never replayed: an idle remote has finished it either way.
        """
        try:
            info = self._health()
            replaced = info["instance_id"] != self._lost
            if replaced:
                self._check_version()
        except (httpx.HTTPError, ProtocolError, KeyError, TypeError, ValueError):
            raise PoolBusy("Modal server unavailable", queue_ms=queue_ms) from None
        if (
            info["load"].get("requests_in_progress") != 0
            or (self._needs_replacement and not replaced)
            or info["target"] != self._target
        ):
            raise PoolBusy("Modal server has not recovered", queue_ms=queue_ms)
        with self._condition:
            self._instance_id, self._versions = info["instance_id"], info["versions"]
            self._lost, self._needs_replacement = None, False

    def _request(self, method: str, path: str, *, budget: float, **kwargs) -> httpx.Response:
        deadline = time.monotonic() + budget
        with self._http.stream(
            method, path, timeout=httpx.Timeout(budget, connect=10), **kwargs
        ) as response:
            if response.headers.get("content-encoding", "identity") != "identity":
                raise ProtocolError("compressed Modal responses are not accepted")
            content = bytearray()
            for chunk in response.iter_bytes():
                if time.monotonic() > deadline:
                    raise httpx.ReadTimeout("Modal transport deadline exceeded")
                if len(content) + len(chunk) > self._max_response_bytes:
                    raise ProtocolError("Modal response exceeds the configured size limit")
                content.extend(chunk)
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                content=bytes(content),
                request=response.request,
            )

    def submit(
        self,
        program,
        timeout: float,
        worker_wait_timeout: float = 0.0,
        request_id: str | None = None,
    ) -> SubmitOutcome:
        if program.options.get("gpu_count", 1) != 1:
            raise ValueError("the Modal backend supports one GPU per request")
        with self._condition:
            if self._closing.is_set():
                raise PoolBusy("server is shutting down")
            self._waiting += 1
        started = time.monotonic()
        deadline = started + max(0.0, worker_wait_timeout)
        acquired = False
        try:
            while not self._closing.is_set():
                if self._slots.acquire(timeout=min(0.1, max(0.0, deadline - time.monotonic()))):
                    acquired = True
                    break
                if time.monotonic() >= deadline:
                    break
        finally:
            with self._condition:
                self._waiting -= 1
                if acquired and self._closing.is_set():
                    self._slots.release()
                    acquired = False
                if acquired:
                    self._active += 1
                self._condition.notify_all()
        queue_ms = (time.monotonic() - started) * 1000
        if not acquired:
            raise PoolBusy("all Modal workers busy or shutting down", queue_ms=queue_ms)

        run_started = time.monotonic()
        try:
            if self._lost is not None:
                self._recover(queue_ms)
            payload = {
                "instructions": [_wire_value(instruction) for instruction in program.instructions],
                "options": {**program.options, "timeout_seconds": timeout},
            }
            program_json = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode()
            files = [("program", (None, program_json, "application/json"))]
            files.extend(
                (f"blob:{digest}", (None, data, "application/octet-stream"))
                for digest, data in program.blob_bytes.items()
            )
            # Cached blobs make the hop larger than what the client uploaded.
            forwarded_bytes = len(program_json) + sum(map(len, program.blob_bytes.values()))
            if forwarded_bytes + _PART_OVERHEAD_BYTES * len(files) > REMOTE_MAX_REQUEST_BYTES:
                raise ModalRejected(
                    413,
                    "request_too_large",
                    "request with its cached blobs exceeds the Modal server's size limit",
                )
            response = self._request(
                "POST",
                "/execute",
                # Waiting for the last replacement, running, then respawning after a failure.
                budget=timeout + 2 * REMOTE_RESPAWN_SECONDS + 60,
                files=files,
            )
            # Modal's proxy replaces X-Request-ID, so the remote names this request itself.
            if response.status_code == 503:
                raise PoolBusy("Modal worker capacity exhausted", queue_ms=queue_ms)
            if response.status_code != 200:
                body = _json_body(response)
                error = body.get("error", {})
                if not isinstance(error, dict):
                    raise ProtocolError("Modal error details must be an object")
                if (response.status_code, error.get("kind")) in _REQUEST_ERRORS:
                    raise ModalRejected(
                        response.status_code, error["kind"], str(error.get("message", ""))
                    )
                if response.status_code == 504 and error.get("kind") == "timeout":
                    failure = WorkerTimeout("Modal worker execution timed out")
                elif (
                    response.status_code == 500
                    and error.get("kind") == "engine"
                    and error.get("message") == "worker crashed"
                ):
                    failure = WorkerCrashed("Modal worker crashed")
                else:
                    unconfirmed = "cleanup was not confirmed" in str(error.get("message", ""))
                    # Delivery or cleanup is unknown; hold requests until the remote recovers.
                    self._mark_lost(needs_replacement=unconfirmed)
                    if unconfirmed:
                        raise WorkerCleanupError("Modal worker cleanup was not confirmed")
                    failure = WorkerCrashed(f"Modal server returned HTTP {response.status_code}")
                failure.worker_id = f"modal/{self._instance_id}"
                failure.queue_ms = queue_ms
                failure.elapsed_ms = (time.monotonic() - run_started) * 1000
                raise failure
            body, binary_parts = _response_body(response)
            result = _parse_program_result(body, binary_parts)
            return SubmitOutcome(
                execution=ProgramOutcome(
                    status=result.status,
                    results=body["results"],
                    error=result.error,
                    binary_parts=binary_parts,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    stdout_truncated=result.stdout_truncated,
                    stderr_truncated=result.stderr_truncated,
                ),
                gpu_id=None,
                queue_ms=queue_ms + result.queue_ms,
                elapsed_ms=result.elapsed_ms,
                lease_wait_ms=result.lease_wait_ms,
                lease_held_ms=result.lease_held_ms,
                worker_id=f"modal/{self._instance_id}",
                finish_reason="completed" if result.completed else "program_failed",
                interfered_request_id=(result.error or {}).get("interfered_request_id"),
                gpu_ids=result.gpu_ids,
            )
        except (httpx.HTTPError, ProtocolError, KeyError, TypeError, ValueError) as exc:
            self._mark_lost(needs_replacement=False)
            failure = WorkerCrashed(f"Modal transport failed: {exc}")
            failure.queue_ms = queue_ms
            failure.elapsed_ms = (time.monotonic() - run_started) * 1000
            raise failure from exc
        finally:
            with self._condition:
                self._active -= 1
                self._slots.release()
                self._condition.notify_all()

    def load(self) -> dict[str, int]:
        with self._condition:
            return {
                "request_capacity": 0 if self._closing.is_set() or self._lost else 1,
                "requests_in_progress": self._active,
                "requests_waiting": self._waiting,
            }

    def worker_status(self) -> dict[str, int]:
        with self._condition:
            return {"worker_count": 1, "busy_workers": self._active}

    def target(self) -> dict[str, str]:
        return self._target

    def versions(self) -> dict[str, str]:
        return self._versions

    @property
    def closing(self) -> bool:
        return self._closing.is_set()

    @property
    def active_requests(self) -> int:
        with self._condition:
            return self._active

    async def shutdown_async(self) -> None:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kcoral-modal-shutdown")
        try:
            await asyncio.get_running_loop().run_in_executor(executor, self.shutdown)
        finally:
            executor.shutdown(wait=False)

    def begin_shutdown(self) -> None:
        with self._condition:
            self._closing.set()
            self._condition.notify_all()

    def shutdown(self) -> None:
        self.begin_shutdown()
        with self._condition:
            while self._active or self._waiting:
                self._condition.wait()
            if not self._shutdown_done:
                self._http.close()
                self._shutdown_done = True


def _wire_value(value):
    if isinstance(value, Ref):
        return {"$ref": value.id}
    if is_dataclass(value):
        return {
            field.name: _wire_value(getattr(value, field.name))
            for field in fields(value)
            if getattr(value, field.name) is not None
        }
    if isinstance(value, list):
        return [_wire_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _wire_value(item) for key, item in value.items()}
    return value
