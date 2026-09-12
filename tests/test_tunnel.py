import asyncio
import json

import grpc
import pytest

from kcoral import kcoral_gateway_pb2 as gateway_pb
from kcoral import kcoral_gateway_pb2_grpc as gateway_grpc
from kcoral import tunnel as tunnel_module
from kcoral.app import create_app
from kcoral.config import ServerConfig
from kcoral.keys import compute_blob_hash
from kcoral.multipart import MultipartPart, encode_multipart, parse_multipart
from kcoral.schemas import strict_json_loads
from kcoral.testing import fake_runtime_factory
from kcoral.tunnel import DATA_CHUNK_BYTES, TunnelManager, _create_channel


class FakeCall:
    def __init__(self):
        self.frames = []

    async def write(self, frame):
        self.frames.append(frame)


async def echo_app(scope, receive, send):
    assert scope["path"] == "/execute"
    assert scope["method"] == "POST"
    body = bytearray()
    while True:
        message = await receive()
        assert message["type"] == "http.request"
        body.extend(message["body"])
        if not message["more_body"]:
            break
    await send(
        {
            "type": "http.response.start",
            "status": 201,
            "headers": [(b"set-cookie", b"a=1"), (b"set-cookie", b"b=2")],
        }
    )
    await send({"type": "http.response.body", "body": bytes(body), "more_body": False})


def test_asgi_bridge_preserves_large_binary_body_and_repeated_headers():
    asyncio.run(_test_asgi_bridge_preserves_large_binary_body_and_repeated_headers())


async def _test_asgi_bridge_preserves_large_binary_body_and_repeated_headers():
    manager = TunnelManager(
        echo_app,
        endpoint="http://router:9000",
        node_id="gpu-a",
        node_token=None,
        server_instance_id="server-a",
        slots=1,
    )
    request_id = "request-1"
    first = gateway_pb.SlotFrame(
        request_id=request_id,
        request_head=gateway_pb.RequestHead(
            headers=[
                gateway_pb.HttpHeader(name=b"content-type", value=b"application/octet-stream")
            ],
            content_length=2 * DATA_CHUNK_BYTES + 17,
        ),
    )
    body = bytes(index % 251 for index in range(2 * DATA_CHUNK_BYTES + 17))
    queue = asyncio.Queue(maxsize=2)
    call = FakeCall()
    task = asyncio.create_task(manager._serve_request(call, first, queue))
    for offset in range(0, len(body), DATA_CHUNK_BYTES):
        await queue.put(body[offset : offset + DATA_CHUNK_BYTES])
    await queue.put(tunnel_module._BODY_END)
    await task

    assert call.frames[0].response_head.status == 201
    assert [(header.name, header.value) for header in call.frames[0].response_head.headers] == [
        (b"set-cookie", b"a=1"),
        (b"set-cookie", b"b=2"),
    ]
    assert b"".join(frame.data for frame in call.frames[1:-1]) == body
    assert call.frames[-1].HasField("end")
    assert all(frame.request_id == request_id for frame in call.frames)


def test_channel_rejects_paths_and_supports_insecure_origins():
    asyncio.run(_test_channel_rejects_paths_and_supports_insecure_origins())


async def _test_channel_rejects_paths_and_supports_insecure_origins():
    with pytest.raises(ValueError, match="must not contain"):
        _create_channel("https://router.example.com/path")
    channel = _create_channel("http://127.0.0.1:1/")
    await channel.close()


class FakeRouter(gateway_grpc.RouterGatewayServicer):
    def __init__(self, body, content_type="application/octet-stream"):
        self.body = body
        self.content_type = content_type
        self.response = bytearray()
        self.response_head = None
        self.finished = asyncio.Event()
        self.hello = None

    async def ConnectSlot(self, request_iterator, context):
        first = await anext(request_iterator)
        self.hello = first.hello
        yield gateway_pb.SlotFrame(ack=gateway_pb.SlotAck())
        request_id = "network-request"
        yield gateway_pb.SlotFrame(
            request_id=request_id,
            request_head=gateway_pb.RequestHead(
                headers=[
                    gateway_pb.HttpHeader(
                        name=b"content-type", value=self.content_type.encode("ascii")
                    )
                ],
                content_length=len(self.body),
            ),
        )
        for offset in range(0, len(self.body), DATA_CHUNK_BYTES):
            yield gateway_pb.SlotFrame(
                request_id=request_id,
                data=self.body[offset : offset + DATA_CHUNK_BYTES],
            )
        yield gateway_pb.SlotFrame(request_id=request_id, end=gateway_pb.EndOfBody())
        async for frame in request_iterator:
            payload = frame.WhichOneof("payload")
            if payload == "response_head":
                self.response_head = frame.response_head
            elif payload == "data":
                self.response.extend(frame.data)
            elif payload == "end":
                self.finished.set()
                return


def test_manager_uses_a_real_outbound_grpc_stream():
    asyncio.run(_test_manager_uses_a_real_outbound_grpc_stream())


async def _test_manager_uses_a_real_outbound_grpc_stream():
    body = bytes(index % 251 for index in range(2 * DATA_CHUNK_BYTES + 17))
    servicer = FakeRouter(body)
    server = grpc.aio.server()
    gateway_grpc.add_RouterGatewayServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    manager = TunnelManager(
        echo_app,
        endpoint=f"http://127.0.0.1:{port}/",
        node_id="gpu-a",
        node_token=None,
        server_instance_id="server-a",
        slots=1,
    )
    await manager.start()
    try:
        await asyncio.wait_for(servicer.finished.wait(), timeout=2)
    finally:
        await manager.close()
        await server.stop(grace=None)
    assert servicer.hello.node_id == "gpu-a"
    assert servicer.hello.server_instance_id == "server-a"
    assert servicer.hello.protocol_version == 1
    assert servicer.response == body


def test_real_app_executes_multipart_request_over_outbound_grpc():
    asyncio.run(_test_real_app_executes_multipart_request_over_outbound_grpc())


async def _test_real_app_executes_multipart_request_over_outbound_grpc():
    program = {
        "instructions": [
            {"op": "run", "id": "value", "fn": "builtin.binary"},
            {"op": "return", "key": "value", "value": {"$ref": "value"}},
        ]
    }
    body, content_type = encode_multipart(
        [
            MultipartPart(
                "program",
                "application/json",
                json.dumps(program).encode("utf-8"),
            )
        ]
    )
    servicer = FakeRouter(body, content_type)
    server = grpc.aio.server()
    gateway_grpc.add_RouterGatewayServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    app = create_app(
        ServerConfig(
            device="cpu",
            num_workers=1,
            max_requests_per_worker=0,
            log_console=False,
            router_endpoint=f"http://127.0.0.1:{port}",
            node_id="cpu-a",
        ),
        runtime_factory=fake_runtime_factory,
    )
    try:
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(servicer.finished.wait(), timeout=5)
    finally:
        await server.stop(grace=None)

    assert servicer.hello.node_id == "cpu-a"
    assert servicer.response_head.status == 200
    response_headers = {
        bytes(header.name).decode("ascii").lower(): bytes(header.value).decode("ascii")
        for header in servicer.response_head.headers
    }
    parts = parse_multipart(response_headers["content-type"], bytes(servicer.response))
    assert [part.name for part in parts] == ["result", "return:0"]
    result = strict_json_loads(parts[0].data)
    assert result["status"] == "COMPLETED"
    assert result["results"]["value"]["sha256"] == compute_blob_hash(b"binary-result")
    assert parts[1].data == b"binary-result"


class ControlledCall(FakeCall):
    def __init__(self, frames):
        super().__init__()
        self.incoming = asyncio.Queue()
        for frame in frames:
            self.incoming.put_nowait(frame)
        self.callbacks = []
        self.read_count = 0

    async def read(self):
        frame = await self.incoming.get()
        self.read_count += 1
        return frame

    async def code(self):
        return grpc.StatusCode.CANCELLED

    async def details(self):
        return "client_disconnected"

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def cancel(self):
        callbacks, self.callbacks = self.callbacks, []
        for callback in callbacks:
            callback(self)


def test_app_completion_interrupts_a_put_already_blocked_by_backpressure(monkeypatch):
    asyncio.run(_blocked_put(monkeypatch, disconnect=False))


def test_transport_cancellation_interrupts_a_full_request_queue(monkeypatch):
    asyncio.run(_blocked_put(monkeypatch, disconnect=True))


async def _blocked_put(monkeypatch, *, disconnect):
    full_put = asyncio.Event()
    stopped = asyncio.Event()
    original_queue = asyncio.Queue

    class ObservedQueue(original_queue):
        async def put(self, value):
            if self.full():
                full_put.set()
            await super().put(value)

    request_id = "race"
    call = ControlledCall(
        [
            gateway_pb.SlotFrame(ack=gateway_pb.SlotAck()),
            gateway_pb.SlotFrame(request_id=request_id, request_head=gateway_pb.RequestHead()),
            *[gateway_pb.SlotFrame(request_id=request_id, data=b"body") for _ in range(4)],
            gateway_pb.SlotFrame(request_id=request_id, end=gateway_pb.EndOfBody()),
            grpc.aio.EOF,
        ]
    )

    async def early_app(scope, receive, send):
        try:
            await full_put.wait()
            if disconnect:
                call.cancel()
                await asyncio.Event().wait()
            await send({"type": "http.response.start", "status": 413})
            await send({"type": "http.response.body", "body": b"rejected"})
        finally:
            stopped.set()

    class Stub:
        def ConnectSlot(self, **kwargs):
            return call

    manager = TunnelManager(
        early_app,
        endpoint="http://router:9000",
        node_id="node",
        node_token=None,
        server_instance_id="instance",
        slots=1,
    )
    manager._stub = Stub()
    monkeypatch.setattr(tunnel_module.asyncio, "Queue", ObservedQueue)
    with pytest.raises(ConnectionError):
        await asyncio.wait_for(manager._connected_slot(0), 1)
    assert stopped.is_set()
    assert full_put.is_set()
    if not disconnect:
        assert call.read_count == 8
        assert call.frames[-1].HasField("end")


def test_reconnect_backoff_resets_only_after_acceptance_and_skips_expected_cancellation(
    monkeypatch,
):
    asyncio.run(_backoff(monkeypatch))


async def _backoff(monkeypatch):
    manager = TunnelManager(
        echo_app,
        endpoint="http://router:9000",
        node_id="node",
        node_token=None,
        server_instance_id="instance",
        slots=1,
    )
    waits = []
    events = []
    outcomes = iter(
        [(False, False), (False, False), (True, False), (False, False), (True, True), (True, True)]
    )

    async def connect(index, attempt):
        try:
            accepted, expected = next(outcomes)
        except StopIteration:
            manager.begin_shutdown()
            return
        attempt.update(accepted=accepted, request_seen=expected)
        if expected:
            raise tunnel_module._ExpectedCancellation("client_disconnected")
        raise ConnectionError("offline")

    async def timeout(awaitable, timeout):
        awaitable.close()
        waits.append(timeout)
        raise asyncio.TimeoutError

    monkeypatch.setattr(manager, "_connected_slot", connect)
    monkeypatch.setattr(manager, "_emit", lambda event, **fields: events.append(fields))
    monkeypatch.setattr(tunnel_module.random, "uniform", lambda a, b: 1)
    monkeypatch.setattr(tunnel_module.asyncio, "wait_for", timeout)
    await manager._run_slot(0)
    assert waits == [1, 2, 1, 2]
    assert [event["reconnect_delay_seconds"] for event in events][-2:] == [0, 0]


def test_local_task_cancellation_never_reconnects(monkeypatch):
    async def run():
        manager = TunnelManager(
            echo_app,
            endpoint="http://router:9000",
            node_id="node",
            node_token=None,
            server_instance_id="instance",
            slots=1,
        )
        calls = 0

        async def connect(index, attempt):
            nonlocal calls
            calls += 1
            raise asyncio.CancelledError

        monkeypatch.setattr(manager, "_connected_slot", connect)
        with pytest.raises(asyncio.CancelledError):
            await manager._run_slot(0)
        assert calls == 1

    asyncio.run(run())
