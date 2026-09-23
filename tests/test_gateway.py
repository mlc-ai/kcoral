"""The built Rust router against real Python slots, applications and client."""

import asyncio
import json
import os
import socket
import subprocess
import sys
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import grpc
import httpx
import pytest
from support.programs import harness_function

from kcoral import Client, Program
from kcoral import kcoral_gateway_pb2 as pb
from kcoral import kcoral_gateway_pb2_grpc as rpc
from kcoral.app import create_app
from kcoral.config import ServerConfig
from kcoral.health import HealthResponse
from kcoral.testing import fake_runtime_factory
from kcoral.tunnel import TunnelManager


def router_binary():
    configured = os.environ.get("KCORAL_ROUTER_BIN")
    binary = (
        Path(configured)
        if configured
        else Path(__file__).resolve().parents[1] / "target/debug/kcoral-router"
    )
    if not binary.is_file():
        if configured or os.environ.get("KCORAL_REQUIRE_GATEWAY_TESTS") == "1":
            pytest.fail(f"required router binary is missing: {binary}")
        pytest.skip("build kcoral-router to run the cross-language integration tests")
    return binary


async def wait_until(check, timeout=8):
    async def poll():
        while not await check():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout)


@asynccontextmanager
async def running_router(directory, *options):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "router.log").open("w") as log:
        process = subprocess.Popen(
            [str(router_binary()), "--port", str(port), "--recovery-threshold", "1", *options],
            stdout=log,
            stderr=log,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient() as client:

                async def started():
                    assert process.poll() is None, (directory / "router.log").read_text()
                    try:
                        await client.get(url + "/health")
                        return True
                    except httpx.TransportError:
                        return False

                await wait_until(started)
            yield url
        finally:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)


@asynccontextmanager
async def report_node(url, node_id, instance, health):
    async with grpc.aio.insecure_channel(url.removeprefix("http://")) as channel:

        async def reports():
            while True:
                snapshot = health()
                yield pb.SupervisorStatus(
                    node_id=node_id,
                    server_healthy=True,
                    supervisor_instance_id="supervisor-" + node_id,
                    server_instance_id=instance,
                    worker_count=len(snapshot["workers"]),
                    busy_workers=sum(w["status"] == "busy" for w in snapshot["workers"]),
                    target=snapshot["target"],
                    versions=snapshot["versions"],
                )
                await asyncio.sleep(0.05)

        call = rpc.RouterGatewayStub(channel).ConnectSupervisor(reports())
        assert await call.read() is not grpc.aio.EOF
        try:
            yield
        finally:
            call.cancel()


@asynccontextmanager
async def running_app(url, node, directory, **limits):
    limits.setdefault("disk_cache_dir", directory / "cache")
    app = create_app(
        ServerConfig(
            sandbox="none",
            device="cpu",
            num_workers=1,
            max_requests_per_worker=0,
            log_console=False,
            log_dir=directory,
            router_endpoint=url,
            node_id=node,
            **limits,
        ),
        runtime_factory=fake_runtime_factory,
    )
    async with app.router.lifespan_context(app):
        async with report_node(url, node, app.state.instance_id, app.state.pool.health):
            yield app


async def ready_nodes(url, count):
    async with httpx.AsyncClient() as client:

        async def ready():
            state = (await client.get(url + "/health")).json()
            return state["status"] == "ok" and state["load"]["request_capacity"] == count

        await wait_until(ready)


@pytest.mark.parametrize("upload_kind", ["bytes", "folder"])
def test_real_router_client_cache_affinity_binary_and_request_logs(tmp_path, upload_kind):
    asyncio.run(_cache_affinity(tmp_path, upload_kind))


async def _cache_affinity(tmp_path, upload_kind):
    async with running_router(tmp_path) as url, AsyncExitStack() as stack:
        apps = [
            await stack.enter_async_context(running_app(url, f"node-{i}", tmp_path / f"node-{i}"))
            for i in range(2)
        ]
        await ready_nodes(url, 2)
        observations = []
        data = b"shared-binary-data" * 65536
        program = Program()
        if upload_kind == "bytes":
            value = program.upload(id="value", kind="bytes", value=data)
        else:
            source = tmp_path / "upload-source"
            source.mkdir()
            (source / "value.bin").write_bytes(data)
            program.upload_folder(source, path="input")
            module = program.upload(
                id="reader",
                kind="module",
                source=(
                    "from pathlib import Path\n"
                    "def read():\n"
                    "    return Path('input/value.bin').read_bytes()\n"
                ),
            )
            reader = program.get_function(id="read", module=module, name="read")
            value = program.run(id="value", fn=reader)
            program.return_file(key="file", path="input/value.bin")
            program.return_folder(key="folder", path="input")
        program.return_(key="value", value=value)

        def execute():
            with Client(url) as client:

                def response_hook(response):
                    response.read()
                    observations.append(
                        (
                            response.headers["x-kcoral-node"],
                            response.headers["x-request-id"],
                            response.json()["status"]
                            if "application/json" in response.headers["content-type"]
                            else "MULTIPART",
                        )
                    )

                client._http.event_hooks["response"] = [response_hook]
                return client.execute(program)

        result = await asyncio.wait_for(asyncio.to_thread(execute), 10)
        assert result.results["value"] == data
        if upload_kind == "folder":
            assert result["file"].read_bytes() == data
            assert result["folder"].files["value.bin"].read_bytes() == data
        assert [item[2] for item in observations] == ["CACHE_MISS", "MULTIPART"]
        assert observations[0][0] == observations[1][0]
        assert observations[0][1] != observations[1][1]
        assert result.request_id == observations[-1][1]
        assert all(str(uuid.UUID(item[1])) == item[1] for item in observations)
        events = [
            json.loads(line)
            for app in apps
            for line in (app.state.events.run_dir / "events.jsonl").read_text().splitlines()
        ]
        assert sum(e["event"] == "request_accepted" for e in events) == 1
        for _, request_id, _ in observations:
            assert any(
                e.get("request_id") == request_id and e["event"] == "request_received"
                for e in events
            )
        await asyncio.sleep(0.05)
        router_log = (tmp_path / "router.log").read_text()
        for _, request_id, _ in observations:
            assert request_id in router_log


def test_real_router_early_413_and_large_early_response_do_not_hang(tmp_path):
    asyncio.run(_early_responses(tmp_path))


async def _early_responses(tmp_path):
    async with running_router(tmp_path) as url:
        async with running_app(url, "small", tmp_path / "small", max_request_mbytes=1 / 1024):
            await ready_nodes(url, 1)
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(url + "/execute", content=b"x" * (4 * 1024**2))
                assert response.status_code == 413
                assert response.json()["request_id"] == response.headers["x-request-id"]
        response_body = b"rejected" * (256 * 1024)

        async def reject(scope, receive, send):
            await send({"type": "http.response.start", "status": 413, "headers": []})
            await send({"type": "http.response.body", "body": response_body})

        manager = TunnelManager(
            reject,
            endpoint=url,
            node_id="large-response",
            node_token=None,
            server_instance_id="large-instance",
            slots=1,
        )

        def snapshot():
            return {"workers": [{"status": "idle"}], "target": {"arch": "fake"}, "versions": {}}

        async with report_node(url, "large-response", "large-instance", snapshot):
            await manager.start()
            try:
                await ready_nodes(url, 1)
                async with httpx.AsyncClient(timeout=5) as client:
                    response = await client.post(url + "/execute", content=b"x" * (4 * 1024**2))
                    assert response.status_code == 413
                    assert response.content == response_body
            finally:
                await manager.close()


def test_router_errors_share_ids_and_client_cancel_reconnects(tmp_path):
    asyncio.run(_errors_and_cancel(tmp_path))


async def _errors_and_cancel(tmp_path):
    async with running_router(
        tmp_path, "--queue-wait-timeout-seconds", "0.05", "--max-request-mbytes", str(1 / 1024)
    ) as url:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.post(url + "/execute", content=b"x" * 2048)
            assert response.status_code == 413
            assert response.json()["request_id"] == response.headers["x-request-id"]
            response = await client.post(url + "/execute", content=b"x")
            assert response.status_code == 503
            assert response.json()["request_id"] == response.headers["x-request-id"]
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def endless(scope, receive, send):
            while (await receive()).get("more_body"):
                pass
            try:
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b"first", "more_body": True})
                started.set()
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        manager = TunnelManager(
            endless,
            endpoint=url,
            node_id="cancel",
            node_token=None,
            server_instance_id="cancel-instance",
            slots=1,
        )

        def snapshot():
            return {"workers": [{"status": "idle"}], "target": {"arch": "fake"}, "versions": {}}

        async with report_node(url, "cancel", "cancel-instance", snapshot):
            await manager.start()
            try:
                await ready_nodes(url, 1)
                async with httpx.AsyncClient(timeout=3) as client:
                    async with client.stream("POST", url + "/execute", content=b"x") as response:
                        assert await anext(response.aiter_bytes()) == b"first"
                    await asyncio.wait_for(cancelled.wait(), 2)
                    await ready_nodes(url, 1)
                    health = (await client.get(url + "/health")).json()
                    assert health["load"]["requests_in_progress"] == 0
            finally:
                await manager.close()


def test_cache_retry_recovers_when_same_node_restarts_between_attempts(tmp_path):
    asyncio.run(_restart_during_retry(tmp_path))


async def _restart_during_retry(tmp_path):
    from kcoral.keys import compute_blob_hash

    async with running_router(tmp_path) as url, AsyncExitStack() as resources:
        original = await resources.enter_async_context(AsyncExitStack())
        app = await original.enter_async_context(running_app(url, "same-node", tmp_path / "first"))
        await ready_nodes(url, 1)
        first, second = b"previously-cached", b"new-blob"
        app.state.cache.put(compute_blob_hash(first), first)
        program = Program()
        for name, data in [("first", first), ("second", second)]:
            value = program.upload(id=name, kind="bytes", value=data)
            program.return_(key=name, value=value)
        attempts = []
        loop = asyncio.get_running_loop()

        async def restart():
            await original.aclose()
            replacement = await resources.enter_async_context(
                running_app(url, "same-node", tmp_path / "replacement")
            )
            assert replacement.state.instance_id != app.state.instance_id
            await ready_nodes(url, 1)

        def execute():
            with Client(url) as client:

                def observe(response):
                    response.read()
                    payload = (
                        response.json()
                        if "application/json" in response.headers["content-type"]
                        else {}
                    )
                    attempts.append(payload)
                    assert response.headers["x-kcoral-node"] == "same-node"
                    if len(attempts) == 1:
                        asyncio.run_coroutine_threadsafe(restart(), loop).result(timeout=8)

                client._http.event_hooks["response"] = [observe]
                return client.execute(program)

        result = await asyncio.wait_for(asyncio.to_thread(execute), 12)
        assert result.results == {"first": first, "second": second}
        assert attempts[0]["missing_blobs"] == [compute_blob_hash(second)]
        assert attempts[1]["missing_blobs"] == [compute_blob_hash(first)]
        assert len(attempts) == 3


def test_tunnel_shutdown_delivers_the_active_response_before_closing(tmp_path):
    asyncio.run(_graceful_tunnel_shutdown(tmp_path))


async def _graceful_tunnel_shutdown(tmp_path):
    async with running_router(tmp_path) as url:
        app_context = running_app(url, "graceful", tmp_path / "node")
        app = await app_context.__aenter__()
        try:
            await ready_nodes(url, 1)
            program = Program()
            program.run(id="sleep", fn=harness_function(program, "sleep", "sleep"), args=[0.3])

            def execute():
                with Client(url) as client:
                    return client.execute(program)

            request = asyncio.create_task(asyncio.to_thread(execute))

            async def active():
                return app.state.pool.active_requests == 1

            await wait_until(active)
            closing = asyncio.create_task(app.state.tunnel.close())
            await asyncio.sleep(0)
            assert not closing.done()
            assert (await asyncio.wait_for(request, 5)).completed
            await asyncio.wait_for(closing, 5)
        finally:
            await app_context.__aexit__(None, None, None)


@pytest.mark.parametrize("failure", ["request_rejected", "invalid_response"])
def test_router_cancellation_reason_reaches_node_logs(tmp_path, failure):
    asyncio.run(_cancel_reason(tmp_path, failure))


async def _cancel_reason(tmp_path, failure):
    events = []

    class Recorder:
        def emit(self, event, **fields):
            events.append((event, fields))

    async def application(scope, receive, send):
        if failure == "invalid_response":
            await send({"type": "http.response.start", "status": 99})
            await send({"type": "http.response.body", "body": b""})
        else:
            while (await receive()).get("more_body"):
                pass
            await asyncio.Event().wait()

    async with running_router(tmp_path, "--max-request-mbytes", str(1 / 1024)) as url:
        manager = TunnelManager(
            application,
            endpoint=url,
            node_id="reason",
            node_token=None,
            server_instance_id="reason-instance",
            slots=1,
            events=Recorder(),
        )

        def snapshot():
            return {"workers": [{"status": "idle"}], "target": {}, "versions": {}}

        async with report_node(url, "reason", "reason-instance", snapshot):
            await manager.start()
            try:
                await ready_nodes(url, 1)

                async def body():
                    yield b"x" * 2048

                async with httpx.AsyncClient(timeout=3) as client:
                    response = await client.post(
                        url + "/execute", content=body() if failure == "request_rejected" else b"x"
                    )
                    assert response.status_code == (413 if failure == "request_rejected" else 502)
                    request_id = response.headers["x-request-id"]

                async def logged():
                    return any(
                        event == "tunnel_request_cancelled"
                        and fields.get("reason") == failure
                        and fields.get("request_id") == request_id
                        for event, fields in events
                    )

                await wait_until(logged)
                await ready_nodes(url, 1)
                lines = [
                    line
                    for line in (tmp_path / "router.log").read_text().splitlines()
                    if "request_finished" in line and request_id in line
                ]
                assert len(lines) == 1
                assert f'finish_reason="{failure}"' in lines[0]
            finally:
                await manager.close()


def test_supervisor_reports_python_worker_status(tmp_path):
    asyncio.run(_supervisor_reports_python_worker_status(tmp_path))


async def _supervisor_reports_python_worker_status(tmp_path):
    node_binary = router_binary().with_name("kcoral-node")
    if not node_binary.is_file():
        pytest.fail(f"required supervisor binary is missing: {node_binary}")
    child_script = tmp_path / "server.py"
    child_script.write_text(
        "import os\n"
        "import uvicorn\n"
        "from kcoral.app import create_app\n"
        "from kcoral.config import ServerConfig\n"
        "from kcoral.testing import fake_runtime_factory\n"
        "if __name__ == '__main__':\n"
        "    config = ServerConfig(sandbox='none', device='cpu', num_workers=2,\n"
        "        max_requests_per_worker=0,\n"
        "        log_console=False, router_endpoint=os.environ['KCORAL_ROUTER_ENDPOINT'],\n"
        "        node_id=os.environ['KCORAL_NODE_ID'])\n"
        "    uvicorn.run(create_app(config, runtime_factory=fake_runtime_factory),\n"
        "        host=os.environ['KCORAL_SERVER_HOST'],\n"
        "        port=int(os.environ['KCORAL_SERVER_PORT']))\n"
    )
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        server_port = reservation.getsockname()[1]
    server_url = f"http://127.0.0.1:{server_port}"
    async with running_router(tmp_path / "router") as router_url:
        async with httpx.AsyncClient() as client:
            initial = await client.get(router_url + "/health")
            assert initial.status_code == 503
            HealthResponse.model_validate(initial.json())
            assert initial.json()["load"]["request_capacity"] == 0
        with (tmp_path / "node.log").open("w") as log:
            process = subprocess.Popen(
                [
                    str(node_binary),
                    "--router-endpoint",
                    router_url,
                    "--node-id",
                    "cpu-node",
                    "--server-url",
                    server_url,
                    "--health-interval-seconds",
                    "0.1",
                    "--",
                    sys.executable,
                    str(child_script),
                ],
                env={
                    **os.environ,
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "python"),
                },
                stdout=log,
                stderr=log,
            )
            try:
                await ready_nodes(router_url, 2)
                async with httpx.AsyncClient() as client:
                    direct = (await client.get(server_url + "/health")).json()
                    routed = (await client.get(router_url + "/health")).json()
                    HealthResponse.model_validate(direct)
                    HealthResponse.model_validate(routed)
                    assert set(direct) == set(routed)
                    assert direct["gpu_count"] == 0
                    assert routed["gpu_count"] is None
                    assert (
                        direct["load"]
                        == routed["load"]
                        == {"request_capacity": 2, "requests_in_progress": 0, "requests_waiting": 0}
                    )
            finally:
                process.terminate()
                try:
                    await asyncio.to_thread(process.wait, timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait)
