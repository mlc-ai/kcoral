"""Synchronous HTTP client and content-addressed cache negotiation."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import httpx

from kcoral.client.program import Program
from kcoral.client.result import ProgramResult, _json_body, _parse_program_result, _response_body
from kcoral.errors import KCoralError, ProtocolError, TransportError
from kcoral.protocol import (
    is_blob_hash,
)

if TYPE_CHECKING:
    from kcoral.client.functions import RemoteFunction


_Parameters = ParamSpec("_Parameters")


_ReturnType = TypeVar("_ReturnType")


class Client:
    """Synchronous HTTP client for a running KCoral server.

    Use as a context manager to close connections automatically. The client
    requires no GPU libraries; computation happens on the remote worker.
    """

    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        connect_timeout_seconds: float = 10.0,
    ) -> None:
        """Create a client without contacting the server.

        :param base_url: Server address, such as ``http://127.0.0.1:8000``.
        :param headers: Optional HTTP headers sent with every request.
        :param connect_timeout_seconds: Limit for establishing a connection.

        Response reading has no client-side timeout. Pass ``timeout_seconds``
        to :meth:`execute` to request a server-side execution limit.
        """
        timeout = httpx.Timeout(None, connect=connect_timeout_seconds)
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def __enter__(self) -> Client:
        """Return this client for use in a ``with`` block."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Close connections when leaving a ``with`` block."""
        self.close()

    def close(self) -> None:
        """Release the underlying HTTP client's connections."""
        self._http.close()

    def function(
        self,
        *,
        timeout: float | None = None,
        output_limit_bytes: int | None = None,
        cpu_only: bool = False,
    ) -> Callable[[Callable[_Parameters, _ReturnType]], RemoteFunction[_Parameters, _ReturnType]]:
        """Decorate a self-contained Python function for this server.

        :param timeout: Server execution limit in seconds, subject to its maximum.
        :param output_limit_bytes: Captured output limit per stream.
        :param cpu_only: Whether the function touches no GPU.
        :returns: A decorator producing a :class:`RemoteFunction`. Its
            ``remote()`` method returns the decoded value; ``execute()`` returns
            the full :class:`ProgramResult`. Ordinary calls execute locally.

        The function must have available Python source, no captured variables,
        and no external globals. Import dependencies inside the function.
        Calls reuse this client's server address, headers and connections. Keep it open
        for the duration of remote calls.
        """
        from kcoral.client.functions import RemoteFunction

        def decorate(
            fn: Callable[_Parameters, _ReturnType],
        ) -> RemoteFunction[_Parameters, _ReturnType]:
            return RemoteFunction(
                fn,
                client=self,
                timeout=timeout,
                output_limit_bytes=output_limit_bytes,
                cpu_only=cpu_only,
            )

        return decorate

    def execute(
        self,
        program: Program,
        *,
        timeout_seconds: float | None = None,
        output_limit_bytes: int | None = None,
    ) -> ProgramResult:
        """Submit a program and decode the values it explicitly returns.

        :param program: Program built with the client-side :class:`Program`.
        :param timeout_seconds: Requested execution limit in seconds; ``None``
            uses the server default. The server clamps it to its configured maximum.
        :param output_limit_bytes: Requested captured output limit per stream;
            ``None`` uses the server default, subject to the server maximum.
        :returns: A completed or failed program outcome. Instruction failures do
            not raise an exception; inspect ``status`` and ``error``.
        :raises TypeError: If ``program`` is not a client-side program.
        :raises KCoralError: If the server returns an HTTP request error.
        :raises TransportError: If no HTTP response can be obtained.
        :raises ProtocolError: If the response is malformed or cache recovery fails.

        The first submission omits binary blobs. A cache miss retries with
        the requested blobs; a second miss triggers one final submission with
        every local blob. Returned tensors are CPU NumPy arrays, including
        extended element types provided by ``ml_dtypes``.
        """
        if not isinstance(program, Program):
            raise TypeError("execute expects a Program")
        options: dict[str, Any] = {}
        if timeout_seconds is not None:
            options["timeout_seconds"] = timeout_seconds
        if output_limit_bytes is not None:
            options["output_limit_bytes"] = output_limit_bytes

        body, binary_parts, route = self._post_program(
            program, options, include_blobs=set(), route=None
        )
        if body.get("status") == "CACHE_MISS":
            if binary_parts:
                raise ProtocolError("CACHE_MISS response must not contain binary parts")
            missing = _parse_missing_blobs(body)
            unavailable = missing - set(program._blobs)
            if unavailable:
                raise ProtocolError(
                    f"server misses blobs with no local bytes to send: {sorted(unavailable)}"
                )
            body, binary_parts, route = self._post_program(
                program, options, include_blobs=missing, route=route
            )
            if body.get("status") == "CACHE_MISS":
                body, binary_parts, route = self._post_program(
                    program,
                    options,
                    include_blobs=set(program._blobs),
                    route=route,
                )
            if body.get("status") == "CACHE_MISS":
                raise ProtocolError("server still reports CACHE_MISS after a complete blob resend")
        return _parse_program_result(body, binary_parts)

    def health(self) -> dict[str, Any]:
        """Read endpoint status, load, and compilation environment.

        ``load`` reports capacity (occupied + free), assigned requests (including
        GPU waiting), and requests awaiting assignment.

        :returns: The server's health response with ``status == "ok"``.
        :raises KCoralError: If the server returns an HTTP error.
        :raises TransportError: If no response can be obtained.
        :raises ProtocolError: If readiness or the response format is invalid.
        """
        response = self._request("GET", "/health")
        if response.status_code != 200:
            raise _server_error(response)
        body = _json_body(response)
        if body.get("status") != "ok":
            raise ProtocolError("health status is not ok")
        return body

    def target(self) -> dict[str, str]:
        """Read the GPU architecture an uploaded library must be built for.

        :returns: Target metadata, for example ``{"arch": "sm_100a"}``.
        :raises ProtocolError: If the health response has no compilation target.

        Query the GPU server, not a CPU compilation server. This calls
        :meth:`health` and can raise the same request and transport exceptions.
        """
        target = self.health().get("target")
        if not isinstance(target, dict) or "arch" not in target:
            raise ProtocolError("the server reported no compilation target")
        return target

    def _post_program(
        self,
        program: Program,
        options: dict[str, Any],
        include_blobs: set[str],
        route: str | None,
    ) -> tuple[dict[str, Any], dict[str, bytes], str | None]:
        payload: dict[str, Any] = {"instructions": program.instructions}
        if options:
            payload["options"] = options
        files: list[tuple[str, tuple[None, bytes, str]]] = [
            (
                "program",
                (
                    None,
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8"),
                    "application/json",
                ),
            )
        ]
        files.extend(
            (f"blob:{blob_hash}", (None, data, "application/octet-stream"))
            for blob_hash, data in program._blobs.items()
            if blob_hash in include_blobs
        )
        headers = {"X-KCoral-Node": route} if route else None
        response = self._request("POST", "/execute", files=files, headers=headers)
        if response.status_code != 200:
            raise _server_error(response)
        next_route = response.headers.get("X-KCoral-Node")
        if next_route is not None and not (0 < len(next_route) <= 512):
            next_route = None
        body, binary_parts = _response_body(response)
        return body, binary_parts, next_route

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise TransportError(str(exc)) from exc


def _parse_missing_blobs(body: dict[str, Any]) -> set[str]:
    missing = body.get("missing_blobs")
    if not isinstance(missing, list) or not missing:
        raise ProtocolError("CACHE_MISS response needs a non-empty 'missing_blobs' array")
    if any(not is_blob_hash(blob_hash) for blob_hash in missing):
        raise ProtocolError("CACHE_MISS response contains an invalid blob hash")
    if len(set(missing)) != len(missing):
        raise ProtocolError("CACHE_MISS response contains duplicate blob hashes")
    return set(missing)


def _server_error(response: httpx.Response) -> KCoralError:
    try:
        body = _json_body(response)
    except ProtocolError:
        body = {}
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message", "server error"))
        kind = error.get("kind")
    else:
        message = str(error) if error else f"HTTP {response.status_code}"
        kind = None
    request_id = body.get("request_id") if isinstance(body, dict) else None
    return KCoralError(response.status_code, message, kind=kind, request_id=request_id)
