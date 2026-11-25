"""Handles for durable jobs submitted to a KCoral router."""

from __future__ import annotations

import math
import time
from contextlib import closing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from kcoral.client.result import ProgramResult, _json_body, _parse_program_result, _response_body
from kcoral.errors import ProtocolError, TransportError

if TYPE_CHECKING:
    from kcoral.client.http import Client


@dataclass(frozen=True)
class JobStatus:
    """A job snapshot. Timestamps are Unix milliseconds, absent until the event occurs.

    ``DISPATCHED`` includes waiting for a worker or GPU on the selected node.
    ``FAILED`` has a program result; ``ERROR`` has an HTTP execution error.
    """

    job_id: str
    state: str
    submitted_at: int
    dispatched_at: int | None
    finished_at: int | None
    node_id: str | None
    request_id: str | None
    error: dict[str, Any] | None


@dataclass(frozen=True)
class JobHandle:
    """A job ID and the client used to query it. Keep that client open."""

    id: str
    _client: Client = field(repr=False, compare=False)

    def status(self) -> JobStatus:
        """Fetch a lightweight snapshot without downloading the result."""
        from kcoral.client.http import _server_error

        with closing(self._client._request("GET", f"/jobs/{self.id}")) as response:
            if response.status_code != 200:
                raise _server_error(response)
            body = _json_body(response)
        if body.get("job_id") != self.id or body.get("state") not in (
            "QUEUED",
            "DISPATCHED",
            "COMPLETED",
            "FAILED",
            "ERROR",
        ):
            raise ProtocolError("invalid job status")
        for name in ("submitted_at", "dispatched_at", "finished_at"):
            value = body.get(name)
            if value is None and name != "submitted_at":
                continue
            if type(value) is not int or value < 0:
                raise ProtocolError(f"invalid job timestamp {name}")
        return JobStatus(
            job_id=self.id,
            state=body["state"],
            submitted_at=body["submitted_at"],
            dispatched_at=body.get("dispatched_at"),
            finished_at=body.get("finished_at"),
            node_id=body.get("node_id"),
            request_id=body.get("request_id"),
            error=body.get("error"),
        )

    def result(self, timeout: float | None = None, *, poll_interval: float = 1.0) -> ProgramResult:
        """Wait for and download the existing ProgramResult, including FAILED results.

        ``timeout`` is a positive, finite client waiting budget in seconds.
        ``None`` waits indefinitely. A waiting timeout raises ``TimeoutError``;
        the server keeps the job. Server execution errors raise ``KCoralError``.
        Results can be read repeatedly until the configured retention expires.
        """
        from kcoral.client.http import _server_error

        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be finite and positive")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"timed out waiting for job {self.id}")
            kwargs = {} if remaining is None else {"timeout": httpx.Timeout(remaining)}
            try:
                with closing(
                    self._client._request("GET", f"/jobs/{self.id}/result", stream=True, **kwargs)
                ) as response:
                    if response.status_code == 200:
                        body, binary_parts = _response_body(response)
                        return _parse_program_result(body, binary_parts)
                    if response.status_code != 202:
                        raise _server_error(response)
            except TransportError as exc:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for job {self.id}") from exc
                raise
            delay = poll_interval
            if deadline is not None:
                delay = min(delay, max(0, deadline - time.monotonic()))
            time.sleep(delay)
