"""Modal forwarding and failures using a fake HTTP invoker, without a GPU."""

import asyncio
import json
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient
from support.programs import python_instructions

from kcoral.config import ServerConfig
from kcoral.protocol import (
    MultipartPart,
    _encode_response,
    compute_blob_hash,
    encode_multipart,
    parse_program,
    parse_request,
)
from kcoral.runtime import modal_pool
from kcoral.runtime.modal_pool import ModalPool, ModalRejected
from kcoral.runtime.pool import PoolBusy
from kcoral.runtime.worker import WorkerCrashed, WorkerTimeout
from kcoral.server.app import create_app
from kcoral.server.cli import build_parser, config_from_args
from kcoral.support.sandbox import SandboxUnavailable


def test_modal_pool_success_busy_timeout_crash_and_shutdown(monkeypatch):
    data = b"safe binary result"
    digest = compute_blob_hash(data)
    instructions = [
        {"op": "upload", "id": "blob", "kind": "bytes", "blob": digest},
        *python_instructions("call", "def main(x): return x\n", [{"$ref": "blob"}]),
        {"op": "return", "key": "result", "value": {"$ref": "call"}},
    ]
    program = parse_program({"instructions": instructions, "options": {"output_limit_bytes": 64}})
    program.blob_bytes[digest] = data

    class FakeInvoker:
        mode = "success"
        block = False
        calls = 0

        def __init__(self):
            self.entered, self.release = threading.Event(), threading.Event()

        def __call__(self, request):
            assert request.headers["Modal-Key"] == "fake-key"
            assert request.headers["Modal-Secret"] == "fake-secret"
            if request.url.path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "gpu_count": 1,
                        "instance_id": "fake",
                        "load": {"request_capacity": 1},
                        "target": {"arch": "fake"},
                        "versions": {},
                    },
                )
            self.calls += 1
            forwarded, blobs, _ = parse_request(request.headers["content-type"], request.read())
            assert forwarded.instructions == program.instructions
            assert forwarded.options == {"timeout_seconds": 5.0, "output_limit_bytes": 64}
            assert blobs == program.blob_bytes  # every resolved byte crosses the hop
            self.entered.set()
            if self.block:
                assert self.release.wait(5)
            if self.mode == "lost":
                raise httpx.ReadError("container disappeared", request=request)
            if self.mode == "oversized":
                return httpx.Response(200, content=b"x" * 8192)
            if self.mode != "success":
                code, kind, message = {
                    "busy": (503, "busy", "server saturated"),
                    "timeout": (504, "timeout", "execution timed out"),
                    "crash": (500, "engine", "worker crashed"),
                }[self.mode]
                return httpx.Response(
                    code,
                    json={
                        "request_id": request.headers["X-Request-ID"],
                        "error": {"kind": kind, "message": message},
                    },
                )
            body, content_type = _encode_response(
                {
                    "status": "COMPLETED",
                    "request_id": request.headers["X-Request-ID"],
                    "queue_ms": 1.0,
                    "elapsed_ms": 2.0,
                    "lease_wait_ms": 0.0,
                    "lease_held_ms": 1.0,
                    "stdout": "",
                    "stderr": "",
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                    "results": {"result": {"type": "bytes", "part": "return:0", "sha256": digest}},
                },
                {"return:0": data},
            )
            return httpx.Response(200, content=body, headers={"content-type": content_type})

    invoker = FakeInvoker()

    def make_pool():
        client = httpx.Client(
            base_url="https://fake.modal.run",
            headers={"Modal-Key": "fake-key", "Modal-Secret": "fake-secret"},
            transport=httpx.MockTransport(invoker),
        )
        return ModalPool("https://fake.modal.run", 4096, http_client=client)

    pool = make_pool()
    outcome = pool.submit(program, timeout=5)
    assert outcome.execution.binary_parts == {"return:0": data}
    assert pool.target() == {"arch": "fake"} and pool.versions() == {}
    invoker.entered.clear()
    invoker.block = True
    with ThreadPoolExecutor(max_workers=1) as executor:
        active = executor.submit(pool.submit, program, 5, 1)
        try:
            assert invoker.entered.wait(5)
            calls = invoker.calls
            assert pool.active_requests == 1
            assert pool.worker_status() == {"worker_count": 1, "busy_workers": 1}
            with pytest.raises(PoolBusy):
                pool.submit(program, timeout=5)
            assert invoker.calls == calls  # no dispatch without a free admission slot
        finally:
            invoker.release.set()
        assert active.result(timeout=5).execution.status == "COMPLETED"
    invoker.block = False
    for mode, error in (("busy", PoolBusy), ("timeout", WorkerTimeout), ("crash", WorkerCrashed)):
        invoker.mode = mode
        with pytest.raises(error):
            pool.submit(program, timeout=5)
        assert not pool.closing and pool.active_requests == 0
    asyncio.run(pool.shutdown_async())
    assert pool.closing and pool.load()["request_capacity"] == 0

    invoker.mode = "success"
    pool = make_pool()
    monkeypatch.setattr(modal_pool, "ModalPool", lambda **kwargs: pool)
    config = config_from_args(
        build_parser().parse_args(
            [
                "--backend",
                "modal",
                "--modal-url",
                "https://fake.modal.run",
            ]
        )
    )
    assert config.backend == "modal" and config.modal_url == "https://fake.modal.run"
    app = create_app(
        ServerConfig(
            backend="modal",
            modal_url=config.modal_url,
            sandbox="none",
            log_console=False,
            disk_cache_capacity_mbytes=0,
            default_timeout_seconds=5,
            output_limit_bytes=64,
        )
    )
    wire_program = json.dumps({"instructions": instructions}).encode()
    body, content_type = encode_multipart(
        [
            MultipartPart(name="program", content_type="application/json", data=wire_program),
            MultipartPart(
                name=f"blob:{digest}", content_type="application/octet-stream", data=data
            ),
        ]
    )
    with TestClient(app) as client:
        assert client.get("/health").json()["gpu_count"] == 1
        response = client.post("/execute", content=body, headers={"content-type": content_type})
        assert response.status_code == 200
        # The front-end resolves its cache again, and the hop still carries the full bytes.
        cached_body, cached_type = encode_multipart(
            [
                MultipartPart(name="program", content_type="application/json", data=wire_program),
            ]
        )
        response = client.post(
            "/execute", content=cached_body, headers={"content-type": cached_type}
        )
        assert response.status_code == 200
    assert pool.closing and "modal" not in sys.modules

    for mode in ("oversized", "lost"):
        invoker.mode = mode
        pool = make_pool()
        with pytest.raises(WorkerCrashed):
            pool.submit(program, timeout=5, request_id=str(uuid.uuid4()))
        assert pool.closing and pool.active_requests == 0
        with pytest.raises(PoolBusy):
            pool.submit(program, timeout=5)
        pool.shutdown()

    def unavailable(*args):
        raise SandboxUnavailable("fake nested sandbox failure")

    monkeypatch.setattr("kcoral.server.app.sandbox.probe", unavailable)
    required_app = create_app(
        ServerConfig(device="cpu", sandbox_required=True, log_console=False),
        runtime_factory=lambda: None,
    )
    with (
        pytest.raises(SandboxUnavailable, match="fake nested sandbox failure"),
        TestClient(required_app),
    ):
        pytest.fail("a required sandbox must never fall back to none")


def test_modal_request_errors_and_limits_keep_the_backend_usable(monkeypatch):
    data = b"safe binary result"
    digest = compute_blob_hash(data)
    instructions = [
        {"op": "upload", "id": "blob", "kind": "bytes", "blob": digest},
        *python_instructions("call", "def main(x): return x\n", [{"$ref": "blob"}]),
        {"op": "return", "key": "result", "value": {"$ref": "call"}},
    ]
    program = parse_program({"instructions": instructions})
    program.blob_bytes[digest] = data
    replies, executed = [], []

    def invoke(request):
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "gpu_count": 1,
                    "instance_id": "fake",
                    "load": {"request_capacity": 1},
                    "target": {"arch": "fake"},
                    "versions": {},
                },
            )
        executed.append(request)
        request_id = request.headers["X-Request-ID"]
        if replies:
            status, kind = replies.pop(0)
            error = {"kind": kind, "message": f"remote {kind}"}
            return httpx.Response(status, json={"request_id": request_id, "error": error})
        body, content_type = _encode_response(
            {
                "status": "COMPLETED",
                "request_id": request_id,
                "queue_ms": 1.0,
                "elapsed_ms": 2.0,
                "lease_wait_ms": 0.0,
                "lease_held_ms": 1.0,
                "stdout": "",
                "stderr": "",
                "stdout_truncated": False,
                "stderr_truncated": False,
                "results": {"result": {"type": "bytes", "part": "return:0", "sha256": digest}},
            },
            {"return:0": data},
        )
        return httpx.Response(200, content=body, headers={"content-type": content_type})

    client = httpx.Client(base_url="https://fake.modal.run", transport=httpx.MockTransport(invoke))
    pool = ModalPool("https://fake.modal.run", 4096, http_client=client)
    # The remote answers these after finishing cleanly, so they stay with the request.
    for status, kind in ((400, "parse"), (413, "request_too_large"), (500, "response_too_large")):
        replies.append((status, kind))
        with pytest.raises(ModalRejected) as rejected:
            pool.submit(program, timeout=5)
        assert (rejected.value.status, rejected.value.kind) == (status, kind)
        assert not pool.closing and pool.active_requests == 0
    assert pool.submit(program, timeout=5).execution.status == "COMPLETED"

    large = b"x" * 8192
    large_digest = compute_blob_hash(large)
    large_program = parse_program(
        {
            "instructions": [
                {"op": "upload", "id": "blob", "kind": "bytes", "blob": large_digest},
                *instructions[1:],
            ]
        }
    )
    large_program.blob_bytes[large_digest] = large
    with monkeypatch.context() as patch:
        patch.setattr(modal_pool, "REMOTE_MAX_REQUEST_BYTES", 4096)
        # The program fits; only its resolved blob pushes the hop over.
        assert pool.submit(program, timeout=5).execution.status == "COMPLETED"
        sent = len(executed)
        with pytest.raises(ModalRejected, match="cached blobs") as rejected:
            pool.submit(large_program, timeout=5)
        assert rejected.value.status == 413
        assert len(executed) == sent and not pool.closing  # refused before the upload

    created = []

    def reuse_pool(**kwargs):
        created.append(kwargs)
        return pool

    monkeypatch.setattr(modal_pool, "ModalPool", reuse_pool)
    app = create_app(
        ServerConfig(
            backend="modal",
            modal_url="https://fake.modal.run",
            sandbox="none",
            log_console=False,
            disk_cache_capacity_mbytes=0,
            max_response_bytes=64,
        )
    )
    body, content_type = encode_multipart(
        [
            MultipartPart(
                name="program",
                content_type="application/json",
                data=json.dumps({"instructions": instructions}).encode(),
            ),
            MultipartPart(
                name=f"blob:{digest}", content_type="application/octet-stream", data=data
            ),
        ]
    )
    with TestClient(app) as http:
        replies.append((413, "request_too_large"))
        response = http.post("/execute", content=body, headers={"content-type": content_type})
        assert response.status_code == 413
        assert response.json()["error"]["kind"] == "request_too_large"
        # Within the remote's limit but over the local one: only this request fails.
        response = http.post("/execute", content=body, headers={"content-type": content_type})
        assert response.status_code == 500
        assert response.json()["error"]["kind"] == "response_too_large"
        assert not pool.closing
    # The pool accepts what the remote may send; the local limit applies after it.
    assert created == [
        {
            "url": "https://fake.modal.run",
            "max_response_bytes": modal_pool.REMOTE_MAX_RESPONSE_BYTES,
        }
    ]

    for limits in (
        {"max_request_bytes": modal_pool.REMOTE_MAX_REQUEST_BYTES + 1},
        {"max_response_bytes": modal_pool.REMOTE_MAX_RESPONSE_BYTES + 1},
        {"max_timeout_seconds": modal_pool.REMOTE_MAX_TIMEOUT_SECONDS + 1},
    ):
        with pytest.raises(ValueError, match="Modal server's"):
            create_app(
                ServerConfig(
                    backend="modal",
                    modal_url="https://fake.modal.run",
                    sandbox="none",
                    log_console=False,
                    **limits,
                )
            )
