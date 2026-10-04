"""Outbound gRPC execution slots for nodes that cannot accept inbound traffic."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import uuid
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

import grpc

from kcoral.server._generated import kcoral_gateway_pb2 as gateway_pb
from kcoral.server._generated import kcoral_gateway_pb2_grpc as gateway_grpc
from kcoral.server.events import EventLogger

DATA_CHUNK_BYTES = 256 * 1024
GRPC_MESSAGE_BYTES = DATA_CHUNK_BYTES + 64 * 1024
PROTOCOL_VERSION = 1
_BODY_END = object()
_DISCONNECT = object()
_LOG = logging.getLogger(__name__)

ASGIApp = Callable[
    [dict, Callable[[], Awaitable[dict]], Callable[[dict], Awaitable[None]]], Awaitable[None]
]


class _ExpectedCancellation(ConnectionError):
    """The router intentionally discarded a request's connection."""


class TunnelManager:
    """Maintain one bounded, reconnecting stream per execution slot."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        endpoint: str,
        node_id: str,
        node_token: str | None,
        server_instance_id: str,
        slots: int,
        events: EventLogger | None = None,
    ) -> None:
        if slots < 1:
            raise ValueError("at least one tunnel slot is required")
        self._app = app
        self._endpoint = endpoint
        self._node_id = node_id
        self._node_token = node_token
        self._server_instance_id = server_instance_id
        self._slot_count = slots
        self._events = events
        self._channel = None
        self._stub = None
        self._tasks = []
        self._closing = asyncio.Event()

    async def start(self) -> None:
        if self._channel is not None:
            raise RuntimeError("tunnel manager is already started")
        self._channel = _create_channel(self._endpoint)
        self._stub = gateway_grpc.RouterGatewayStub(self._channel)
        self._tasks = [
            asyncio.create_task(self._run_slot(i), name=f"kcoral-tunnel-{i}")
            for i in range(self._slot_count)
        ]

    def begin_shutdown(self) -> None:
        """Stop idle slots and let active requests reach their existing timeout."""
        self._closing.set()

    async def close(self) -> None:
        self.begin_shutdown()
        try:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
            if self._channel is not None:
                await self._channel.close()
                self._channel = None
                self._stub = None

    async def _run_slot(self, index: int) -> None:
        delay = 1.0
        while not self._closing.is_set():
            attempt = {"accepted": False, "request_seen": False}
            try:
                await self._connected_slot(index, attempt)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._closing.is_set():
                    return
                if attempt["accepted"]:
                    delay = 1.0
                expected = isinstance(exc, _ExpectedCancellation) or (
                    isinstance(exc, grpc.aio.AioRpcError)
                    and exc.code() == grpc.StatusCode.CANCELLED
                    and attempt["accepted"]
                    and attempt["request_seen"]
                )
                wait = 0.0 if expected else delay * random.uniform(0.8, 1.2)
                self._emit(
                    "tunnel_disconnected",
                    level="INFO" if expected else "WARNING",
                    slot=index,
                    error=f"{type(exc).__name__}: {exc}",
                    reconnect_delay_seconds=wait,
                )
                if wait:
                    try:
                        await asyncio.wait_for(self._closing.wait(), timeout=wait)
                    except asyncio.TimeoutError:
                        pass
                    delay = min(delay * 2, 30.0)
                else:
                    await asyncio.sleep(0)

    async def _connected_slot(self, index: int, attempt: dict | None = None) -> None:
        assert self._stub is not None
        if attempt is None:
            attempt = {"accepted": False, "request_seen": False}
        metadata = (("authorization", f"Bearer {self._node_token}"),) if self._node_token else None
        call = self._stub.ConnectSlot(metadata=metadata, wait_for_ready=True)
        active = None
        body_queue = None
        request_id = None
        body_ended = False
        disconnected = asyncio.Event()
        call.add_done_callback(lambda _: disconnected.set())
        disconnected_task = asyncio.create_task(disconnected.wait())
        slot_task = asyncio.current_task()

        async def stop_after_active():
            await self._closing.wait()
            if active is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(active)
                # Half-close only after the final response write. Cancelling the
                # RPC here could discard bytes the router has not received yet.
                await call.done_writing()
            else:
                # RPC cancellation alone does not unblock a write waiting for
                # the initial connection when wait_for_ready is enabled.
                slot_task.cancel()

        shutdown_task = asyncio.create_task(stop_after_active())
        slot_id = str(uuid.uuid4())
        try:
            await call.write(
                gateway_pb.SlotFrame(
                    hello=gateway_pb.SlotHello(
                        node_id=self._node_id,
                        server_instance_id=self._server_instance_id,
                        slot_id=slot_id,
                        protocol_version=PROTOCOL_VERSION,
                    )
                )
            )
            ack = await call.read()
            if ack is grpc.aio.EOF or ack.request_id or ack.WhichOneof("payload") != "ack":
                raise ConnectionError("Router did not acknowledge slot registration")
            attempt["accepted"] = True
            self._emit("tunnel_connected", slot=index, slot_id=slot_id)
            while True:
                frame = await call.read()
                if frame is grpc.aio.EOF:
                    raise ConnectionError("Router closed the execution slot")
                payload = frame.WhichOneof("payload")
                if payload == "request_head":
                    if self._closing.is_set():
                        return
                    if active is not None:
                        if not body_ended:
                            raise RuntimeError("Router reused a slot before the request body ended")
                        await active
                    if not frame.request_id:
                        raise RuntimeError("request head has no request identifier")
                    attempt["request_seen"] = True
                    request_id = frame.request_id
                    body_ended = False
                    body_queue = asyncio.Queue(maxsize=2)
                    active = asyncio.create_task(
                        self._serve_request(call, frame, body_queue),
                        name=f"kcoral-tunnel-request-{index}",
                    )
                    active.add_done_callback(lambda task: _cancel_call_on_error(task, call))
                elif payload in {"data", "end", "cancel"}:
                    if active is None or frame.request_id != request_id:
                        raise RuntimeError("slot frame has no matching active request")
                    if payload == "cancel":
                        self._emit(
                            "tunnel_request_cancelled",
                            request_id=request_id,
                            slot=slot_id,
                            reason=frame.cancel.reason,
                        )
                        active.cancel()
                        raise _ExpectedCancellation(frame.cancel.reason)
                    if body_ended:
                        raise RuntimeError("Router sent data after the request body ended")
                    if payload == "data" and len(frame.data) > DATA_CHUNK_BYTES:
                        raise RuntimeError("request data frame exceeds the size limit")
                    if payload == "end":
                        body_ended = True
                    item = _BODY_END if payload == "end" else bytes(frame.data)
                    # Completion must interrupt an already blocked put, not just be
                    # checked beforehand. Only the current request's body is discarded.
                    if active.done():
                        await active
                        continue
                    put = asyncio.create_task(body_queue.put(item))
                    try:
                        done, _ = await asyncio.wait(
                            {put, active, disconnected_task}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if disconnected_task in done:
                            code = await call.code()
                            reason = await call.details()
                            self._emit(
                                "tunnel_request_cancelled",
                                request_id=request_id,
                                slot=slot_id,
                                reason=reason or code.name,
                            )
                            if code == grpc.StatusCode.CANCELLED:
                                raise _ExpectedCancellation(reason)
                            raise ConnectionError("slot closed during request upload")
                        if active in done:
                            await active
                    finally:
                        put.cancel()
                        await asyncio.gather(put, return_exceptions=True)
                else:
                    raise RuntimeError(f"Router sent invalid slot frame {payload!r}")
        except grpc.aio.AioRpcError as exc:
            if request_id is not None and not self._closing.is_set():
                self._emit(
                    "tunnel_request_cancelled",
                    request_id=request_id,
                    slot=slot_id,
                    reason=exc.details() or exc.code().name,
                )
            raise
        except asyncio.CancelledError:
            if not self._closing.is_set():
                raise
        finally:
            if active is not None:
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
            shutdown_task.cancel()
            disconnected_task.cancel()
            await asyncio.gather(shutdown_task, disconnected_task, return_exceptions=True)
            call.cancel()

    async def _serve_request(
        self,
        call,
        first: gateway_pb.SlotFrame,
        body_queue: asyncio.Queue,
    ) -> None:
        request_id = first.request_id
        head = first.request_head
        response_started = False
        response_finished = False

        async def receive() -> dict:
            item = await body_queue.get()
            if item is _DISCONNECT:
                return {"type": "http.disconnect"}
            if item is _BODY_END:
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.request", "body": item, "more_body": True}

        async def send(message: dict) -> None:
            nonlocal response_started, response_finished
            if message["type"] == "http.response.start":
                if response_started:
                    raise RuntimeError("ASGI app started a response twice")
                response_started = True
                await call.write(
                    gateway_pb.SlotFrame(
                        request_id=request_id,
                        response_head=gateway_pb.ResponseHead(
                            status=message["status"],
                            headers=[
                                gateway_pb.HttpHeader(name=name, value=value)
                                for name, value in message.get("headers", [])
                            ],
                        ),
                    )
                )
                return
            if message["type"] != "http.response.body" or not response_started:
                raise RuntimeError("ASGI app sent an invalid response event")
            body = message.get("body", b"")
            for offset in range(0, len(body), DATA_CHUNK_BYTES):
                await call.write(
                    gateway_pb.SlotFrame(
                        request_id=request_id,
                        data=body[offset : offset + DATA_CHUNK_BYTES],
                    )
                )
            if not message.get("more_body", False):
                response_finished = True
                await call.write(
                    gateway_pb.SlotFrame(request_id=request_id, end=gateway_pb.EndOfBody())
                )

        headers = [(bytes(header.name), bytes(header.value)) for header in head.headers]
        if not any(name.lower() == b"host" for name, _ in headers):
            headers.append((b"host", b"kcoral-tunnel"))
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "2",
            "method": "POST",
            "scheme": "https" if self._endpoint.startswith("https://") else "http",
            "path": "/execute",
            "raw_path": b"/execute",
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("outbound-tunnel", 0),
            "server": ("kcoral", 0),
            "state": {},
        }
        await self._app(scope, receive, send)
        if not response_finished:
            raise RuntimeError("ASGI app ended without completing its response")

    def _emit(self, event: str, *, level: str = "INFO", **fields: object) -> None:
        if self._events is not None:
            self._events.emit(event, level=level, node_id=self._node_id, **fields)
        elif level == "WARNING":
            _LOG.warning("%s: %s", event, fields)


def _cancel_call_on_error(task: asyncio.Task, call) -> None:
    if task.cancelled():
        return
    if task.exception() is not None:
        call.cancel()


def _create_channel(endpoint: str) -> grpc.aio.Channel:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Router endpoint must be an HTTP or HTTPS origin")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Router endpoint must not contain a path, query, or fragment")
    options = (
        ("grpc.max_send_message_length", GRPC_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", GRPC_MESSAGE_BYTES),
        ("grpc.keepalive_time_ms", 15_000),
        ("grpc.keepalive_timeout_ms", 5_000),
        ("grpc.keepalive_permit_without_calls", 1),
    )
    if parsed.scheme == "https":
        return grpc.aio.secure_channel(
            parsed.netloc,
            grpc.ssl_channel_credentials(),
            options=options,
        )
    return grpc.aio.insecure_channel(parsed.netloc, options=options)
