"""Synchronous Python client for the instruction protocol.

Build a program from the instruction helpers, then execute it:

    from benchmark_server.client import Client, ref, run, upload_function

    with Client("http://127.0.0.1:8000") as client:
        outcome = client.execute(
            [
                upload_function("fn", "def main(a):\\n    return a + 1\\n"),
                run("y", ref("fn"), [41]),
            ]
        )
        print(outcome["y"].value)  # 42

``execute`` follows the protocol's caching flow transparently: it first sends
uploads by key only (the warm-cache fast path) and resends the inline bytes for
exactly the keys the server reports missing.

Error model: a failing *instruction* is data (inspect ``ProgramResult``); a
non-200 response raises :class:`BenchmarkServerError`; connection problems
raise :class:`TransportError`; a malformed server reply raises
:class:`ProtocolError`.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import httpx

from .keys import compute_key


class BenchmarkServerError(Exception):
    """A non-200 response from the server."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        kind: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.kind = kind
        self.request_id = request_id


class TransportError(Exception):
    """The request never produced an HTTP response (connection refused, ...)."""


class ProtocolError(Exception):
    """The server's reply does not follow the protocol."""


# --- instruction builders ----------------------------------------------------


def ref(instruction_id: str) -> dict:
    """A handle reference to an earlier instruction's result."""
    return {"$ref": instruction_id}


def run(instruction_id: str, fn: Any, args: list | None = None) -> dict:
    """A ``run`` instruction; ``fn`` is a builtin name or a :func:`ref`."""
    return {"id": instruction_id, "op": "run", "fn": fn, "args": args or []}


def upload_function(instruction_id: str, source: str) -> dict:
    """Upload Python source defining ``main``; the content key is computed here."""
    inline = {"source": source}
    return {
        "id": instruction_id,
        "op": "upload",
        "kind": "function",
        "key": compute_key("function", inline),
        "inline": inline,
    }


def upload_package(instruction_id: str, files: dict[str, str], entry: str) -> dict:
    """Upload a multi-file function package.

    ``files`` maps relative POSIX paths to source text; ``entry`` is
    ``"path/to/mod.py:attribute"`` naming the callable to bind to the handle.
    """
    inline = {"files": dict(files), "entry": entry}
    return {
        "id": instruction_id,
        "op": "upload",
        "kind": "package",
        "key": compute_key("package", inline),
        "inline": inline,
    }


def upload_tensor(instruction_id: str, array: Any) -> dict:
    """Upload a tensor from a torch tensor or a numpy-like array (anything with
    ``dtype``, ``shape``, and C-order ``tobytes``)."""
    dtype, shape, raw = _tensor_fields(array)
    return upload_tensor_bytes(instruction_id, dtype, shape, raw)


def upload_tensor_bytes(instruction_id: str, dtype: str, shape: list[int], raw: bytes) -> dict:
    """Upload a tensor from raw row-major bytes; needs neither torch nor numpy."""
    inline = {"dtype": dtype, "shape": shape, "data_b64": base64.b64encode(raw).decode("ascii")}
    return {
        "id": instruction_id,
        "op": "upload",
        "kind": "tensor",
        "key": compute_key("tensor", inline),
        "inline": inline,
    }


def _tensor_fields(array: Any) -> tuple[str, list[int], bytes]:
    try:
        import torch

        if isinstance(array, torch.Tensor):
            tensor = array.detach().cpu().contiguous()
            dtype = str(tensor.dtype).removeprefix("torch.")
            shape = [int(d) for d in tensor.shape]
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return dtype, shape, raw
    except ImportError:
        pass
    if hasattr(array, "dtype") and hasattr(array, "shape") and hasattr(array, "tobytes"):
        # numpy (and compatibles); tobytes() copies in C order.
        return str(array.dtype), [int(d) for d in array.shape], array.tobytes()
    raise TypeError(f"cannot upload {type(array).__name__!r} as a tensor")


# --- results -----------------------------------------------------------------


@dataclass
class InstructionResult:
    id: str
    op: str
    status: str  # OK | FAILED | SKIPPED
    value: Any = None
    error: dict | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@dataclass
class ProgramResult:
    status: str  # COMPLETED | FAILED
    request_id: str | None
    queue_ms: float | None
    elapsed_ms: float | None
    results: list[InstructionResult]

    @property
    def completed(self) -> bool:
        return self.status == "COMPLETED"

    def __getitem__(self, instruction_id: str) -> InstructionResult:
        for result in self.results:
            if result.id == instruction_id:
                return result
        raise KeyError(instruction_id)


# --- client ------------------------------------------------------------------


class Client:
    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        connect_timeout_seconds: float = 10.0,
    ) -> None:
        # No read timeout: a benchmark request legitimately runs for minutes.
        timeout = httpx.Timeout(None, connect=connect_timeout_seconds)
        self._http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def execute(
        self,
        instructions: list[dict],
        *,
        timeout_seconds: float | None = None,
        output_limit_bytes: int | None = None,
    ) -> ProgramResult:
        """Run a program; transparently resends inline bytes on ``CACHE_MISS``."""
        options: dict[str, Any] = {}
        if timeout_seconds is not None:
            options["timeout_seconds"] = timeout_seconds
        if output_limit_bytes is not None:
            options["output_limit_bytes"] = output_limit_bytes
        inline_by_key = {
            ins["key"]: ins["inline"]
            for ins in instructions
            if ins.get("op") == "upload" and ins.get("inline") is not None
        }
        body = self._post_program(instructions, options, inline_keys=set())
        if body.get("status") == "CACHE_MISS":
            missing = set(body.get("missing_keys", []))
            unavailable = missing - set(inline_by_key)
            if unavailable:
                raise ProtocolError(
                    f"server misses keys with no local bytes to send: {sorted(unavailable)}"
                )
            body = self._post_program(instructions, options, inline_keys=missing)
            if body.get("status") == "CACHE_MISS":
                raise ProtocolError("server still reports CACHE_MISS after an inline resend")
        return _parse_program_result(body)

    def health(self) -> dict:
        response = self._request("GET", "/health")
        if response.status_code != 200:
            raise _server_error(response)
        body = _json_body(response)
        if body.get("status") != "ok":
            raise ProtocolError("health status is not ok")
        return body

    def _post_program(self, instructions: list[dict], options: dict, inline_keys: set) -> dict:
        wire: list[dict] = []
        for ins in instructions:
            if ins.get("op") == "upload":
                item = {k: v for k, v in ins.items() if k != "inline"}
                if ins.get("inline") is not None and ins["key"] in inline_keys:
                    item["inline"] = ins["inline"]
                wire.append(item)
            else:
                wire.append(ins)
        payload: dict[str, Any] = {"instructions": wire}
        if options:
            payload["options"] = options
        response = self._request("POST", "/benchmark", json=payload)
        if response.status_code != 200:
            raise _server_error(response)
        return _json_body(response)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise TransportError(str(exc)) from exc


def _server_error(response: httpx.Response) -> BenchmarkServerError:
    try:
        body = response.json()
    except Exception:
        body = {}
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = str(error.get("message", "server error"))
        kind = error.get("kind")
    else:
        message = str(error) if error else f"HTTP {response.status_code}"
        kind = None
    request_id = body.get("request_id") if isinstance(body, dict) else None
    return BenchmarkServerError(response.status_code, message, kind=kind, request_id=request_id)


def _json_body(response: httpx.Response) -> dict:
    try:
        body = response.json()
    except Exception as exc:
        raise ProtocolError(f"response is not JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ProtocolError("JSON response must be an object")
    return body


def _parse_program_result(body: dict) -> ProgramResult:
    try:
        status = body["status"]
        if status not in ("COMPLETED", "FAILED"):
            raise ValueError(f"unexpected program status {status!r}")
        results = [
            InstructionResult(
                id=item["id"],
                op=item["op"],
                status=item["status"],
                value=item.get("value"),
                error=item.get("error"),
                stdout=item.get("stdout", ""),
                stderr=item.get("stderr", ""),
                stdout_truncated=item.get("stdout_truncated", False),
                stderr_truncated=item.get("stderr_truncated", False),
            )
            for item in body["results"]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"malformed execution response: {exc}") from exc
    return ProgramResult(
        status=status,
        request_id=body.get("request_id"),
        queue_ms=body.get("queue_ms"),
        elapsed_ms=body.get("elapsed_ms"),
        results=results,
    )
