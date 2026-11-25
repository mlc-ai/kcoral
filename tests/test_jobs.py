"""Durable job API against the real Rust router and Python worker tunnel."""

import asyncio
import json
import sqlite3
import subprocess
import sys
import uuid

import httpx
import pytest
from test_gateway import ready_nodes, running_app, running_router, wait_until

from kcoral import Client, KCoralError, Program, TransportError


def program_with_result(data=b"binary\x00\xff" * 1000, *, delay=0, fail=False):
    program = Program()
    module = program.upload(
        kind="module",
        source=(
            "def run(data, delay, fail):\n"
            "    import time\n"
            "    time.sleep(delay)\n"
            "    if fail:\n"
            "        raise ValueError('job failed')\n"
            "    return data\n"
        ),
    )
    function = program.get_function(module=module, name="run", cpu_only=True)
    blob = program.upload(kind="bytes", value=data)
    result = program.run(fn=function, args=[blob, delay, fail])
    program.return_(key="data", value=result)
    return program


def test_submit_without_nodes_process_exit_and_restart(tmp_path):
    asyncio.run(_submit_without_nodes_process_exit_and_restart(tmp_path))


async def _submit_without_nodes_process_exit_and_restart(tmp_path):
    directory = tmp_path / "jobs"
    options = ("--job-dir", str(directory))
    async with running_router(tmp_path / "first", *options) as url:
        # A submitting process exits before any worker is available.
        script = """
import sys
from kcoral import Client, Program
p = Program()
v = p.upload(kind="bytes", value=b"survived client exit")
p.return_(key="data", value=v)
with Client(sys.argv[1]) as client:
    print(client.submit(p, idempotency_key="restart-test").id)
"""
        completed = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", script, url],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stderr
        job_id = completed.stdout.strip()
        with Client(url) as client:
            handle = client.get_job(job_id)
            assert (await asyncio.to_thread(handle.status)).state == "QUEUED"
            with pytest.raises(TimeoutError):
                await asyncio.to_thread(handle.result, timeout=0.05, poll_interval=0.01)
            assert (await asyncio.to_thread(handle.status)).state == "QUEUED"

    async with running_router(tmp_path / "second", *options) as url:
        async with running_app(url, "worker", tmp_path / "worker"):
            await ready_nodes(url, 1)
            with Client(url) as client:
                result = await asyncio.to_thread(client.get_job(job_id).result, timeout=10)
                assert result["data"] == b"survived client exit"
    # Terminal results also survive restart without any connected nodes.
    async with running_router(tmp_path / "third", *options) as url:
        with Client(url) as client:
            handle = client.get_job(job_id)
            assert (await asyncio.to_thread(handle.status)).state == "COMPLETED"
            result = await asyncio.to_thread(handle.result, timeout=2)
            assert result["data"] == b"survived client exit"


def test_busy_worker_idempotency_result_and_errors(tmp_path):
    asyncio.run(_busy_worker_idempotency_result_and_errors(tmp_path))


async def _busy_worker_idempotency_result_and_errors(tmp_path):
    directory = tmp_path / "jobs"
    async with running_router(tmp_path, "--job-dir", str(directory)) as url:
        async with running_app(url, "worker", tmp_path / "worker", default_timeout_seconds=5):
            await ready_nodes(url, 1)
            with Client(url, client="job-client") as client:
                first = await asyncio.to_thread(client.submit, program_with_result(delay=1))

                async def dispatched():
                    return (await asyncio.to_thread(first.status)).state == "DISPATCHED"

                await wait_until(dispatched)
                payload = bytes(range(256)) * 10000
                program = program_with_result(payload)
                # Concurrent same-key submissions with independently generated multipart boundaries.
                handles = await asyncio.gather(
                    *[
                        asyncio.to_thread(client.submit, program, idempotency_key="same-key")
                        for _ in range(3)
                    ]
                )
                assert len({handle.id for handle in handles}) == 1
                second = handles[0]
                assert (await asyncio.to_thread(second.status)).state == "QUEUED"
                with pytest.raises(KCoralError) as conflict:
                    await asyncio.to_thread(
                        client.submit, program_with_result(b"different"), idempotency_key="same-key"
                    )
                assert conflict.value.status_code == 409
                assert (await asyncio.to_thread(first.result, timeout=10)).completed
                for _ in range(2):
                    result = await asyncio.to_thread(second.result, timeout=10, poll_interval=0.02)
                    assert result["data"] == payload
                assert (await asyncio.to_thread(second.status)).node_id == "worker"
                with sqlite3.connect(directory / "jobs.sqlite3") as db:
                    headers = json.loads(
                        db.execute(
                            "SELECT request_headers FROM jobs WHERE id = ?", (second.id,)
                        ).fetchone()[0]
                    )
                    assert ["x-kcoral-client", list(b"job-client")] in headers
                    assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 2

                failed = await asyncio.to_thread(client.submit, program_with_result(fail=True))
                assert (await asyncio.to_thread(failed.result, timeout=10)).status == "FAILED"
                assert (await asyncio.to_thread(failed.status)).state == "FAILED"
                timeout = await asyncio.to_thread(
                    client.submit, program_with_result(delay=2), timeout_seconds=0.1
                )
                with pytest.raises(KCoralError) as expired:
                    await asyncio.to_thread(timeout.result, timeout=10)
                assert expired.value.status_code == 504
                assert (await asyncio.to_thread(timeout.status)).state == "ERROR"


def test_killed_router_marks_dispatched_unknown(tmp_path):
    asyncio.run(_killed_router_marks_dispatched_unknown(tmp_path))


async def _killed_router_marks_dispatched_unknown(tmp_path):
    options = ("--job-dir", str(tmp_path / "jobs"))
    async with running_router(tmp_path / "first", *options, include_process=True) as (url, process):
        async with running_app(url, "worker", tmp_path / "worker"):
            await ready_nodes(url, 1)
            with Client(url) as client:
                handle = await asyncio.to_thread(client.submit, program_with_result(delay=1))

                async def dispatched():
                    return (await asyncio.to_thread(handle.status)).state == "DISPATCHED"

                await wait_until(dispatched)
                process.kill()
                await asyncio.to_thread(process.wait, timeout=5)
    async with running_router(tmp_path / "second", *options) as url:
        with Client(url) as client:
            restored = client.get_job(handle.id)
            state = await asyncio.to_thread(restored.status)
            assert state.state == "ERROR" and state.error["kind"] == "execution_unknown"
            with pytest.raises(KCoralError, match="execution outcome is unknown"):
                await asyncio.to_thread(restored.result, timeout=2)


def test_limits_missing_blobs_and_expiration(tmp_path):
    asyncio.run(_limits_missing_blobs_and_expiration(tmp_path))


async def _limits_missing_blobs_and_expiration(tmp_path):
    directory = tmp_path / "jobs"
    async with running_router(
        tmp_path,
        "--job-dir",
        str(directory),
        "--max-jobs",
        "1",
        "--job-retention-seconds",
        "1",
        "--max-request-bytes",
        "20000",
        "--max-job-result-bytes",
        "5000",
    ) as url:
        async with httpx.AsyncClient(base_url=url) as http:
            program = program_with_result()
            response = await http.post(
                "/jobs",
                files={
                    "program": (
                        None,
                        json.dumps({"instructions": program.instructions}),
                        "application/json",
                    )
                },
            )
            assert response.status_code == 400 and "referenced blobs" in response.text
        with Client(url) as client:
            with pytest.raises(KCoralError) as oversized:
                await asyncio.to_thread(client.submit, program_with_result(b"x" * 30000))
            assert oversized.value.status_code == 413
            first = await asyncio.to_thread(
                client.submit, program_with_result(b"x" * 10000), idempotency_key="full"
            )
            with pytest.raises(KCoralError) as full:
                await asyncio.to_thread(client.submit, program_with_result(b"other"))
            assert full.value.status_code == 503
            duplicate = await asyncio.to_thread(
                client.submit, program_with_result(b"x" * 10000), idempotency_key="full"
            )
            assert duplicate.id == first.id
            async with running_app(url, "worker", tmp_path / "worker"):
                with pytest.raises(KCoralError) as too_large:
                    await asyncio.to_thread(first.result, timeout=10, poll_interval=0.02)
                assert too_large.value.kind == "response_too_large"

                async def expired():
                    with sqlite3.connect(directory / "jobs.sqlite3") as db:
                        return db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0

                await wait_until(expired)
                with pytest.raises(KCoralError) as missing:
                    await asyncio.to_thread(first.status)
                assert missing.value.status_code == 404
                assert not list(directory.glob("*.response"))
                second = await asyncio.to_thread(client.submit, program_with_result(b"ok"))
                assert (await asyncio.to_thread(second.result, timeout=10, poll_interval=0.02))[
                    "data"
                ] == b"ok"


def test_submission_transport_retry_keeps_key():
    attempts = []
    job_id = str(uuid.uuid4())

    def respond(request):
        attempts.append(request.headers["Idempotency-Key"])
        if len(attempts) == 1:
            raise httpx.ReadError("response lost")
        return httpx.Response(202, json={"job_id": job_id})

    with Client("http://router") as client:
        client._http.close()
        client._http = httpx.Client(
            base_url="http://router", transport=httpx.MockTransport(respond)
        )
        assert client.submit(program_with_result()).id == job_id
    assert len(attempts) == 2 and attempts[0] == attempts[1]


def test_failed_submission_exposes_recovery_key():
    def disconnected(request):
        raise httpx.ConnectError("offline")

    with Client("http://router") as client:
        client._http.close()
        client._http = httpx.Client(
            base_url="http://router", transport=httpx.MockTransport(disconnected)
        )
        with pytest.raises(TransportError) as failure:
            client.submit(program_with_result())
    assert str(uuid.UUID(failure.value.idempotency_key)) == failure.value.idempotency_key


def test_lost_submission_response_executes_once(tmp_path):
    asyncio.run(_lost_submission_response_executes_once(tmp_path))


async def _lost_submission_response_executes_once(tmp_path):
    class LoseAcknowledgment(httpx.HTTPTransport):
        def __init__(self):
            super().__init__()
            self.lost = False

        def handle_request(self, request):
            response = super().handle_request(request)
            if request.url.path == "/jobs" and not self.lost:
                self.lost = True
                assert response.status_code == 202
                response.close()
                raise httpx.ReadError("admission acknowledgment lost")
            return response

    async with running_router(tmp_path, "--job-dir", str(tmp_path / "jobs")) as url:
        async with running_app(url, "worker", tmp_path / "worker"):
            await ready_nodes(url, 1)
            marker = tmp_path / "executions"
            program = Program()
            module = program.upload(
                kind="module",
                source=(
                    "def run():\n"
                    "    from pathlib import Path\n"
                    f"    path = Path({str(marker)!r})\n"
                    "    with path.open('a') as stream:\n"
                    "        stream.write('execution\\n')\n"
                    "    return 42\n"
                ),
            )
            function = program.get_function(module=module, name="run", cpu_only=True)
            program.return_(key="answer", value=program.run(fn=function))
            with Client(url) as client:
                client._http.close()
                client._http = httpx.Client(base_url=url, transport=LoseAcknowledgment())
                handle = await asyncio.to_thread(client.submit, program)
                assert (await asyncio.to_thread(handle.result, timeout=10))["answer"] == 42
            assert marker.read_text() == "execution\n"


def test_large_integer_fingerprints_and_exclusive_store(tmp_path):
    asyncio.run(_large_integer_fingerprints_and_exclusive_store(tmp_path))


async def _large_integer_fingerprints_and_exclusive_store(tmp_path):
    from test_gateway import router_binary

    directory = tmp_path / "jobs"
    async with running_router(tmp_path, "--job-dir", str(directory)) as url:
        process = await asyncio.to_thread(
            subprocess.run,
            [str(router_binary()), "--port", "0", "--job-dir", str(directory)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert process.returncode != 0 and "already in use" in process.stderr

        def integer_program(value):
            program = Program()
            module = program.upload(kind="module", source="def identity(x): return x")
            function = program.get_function(module=module, name="identity", cpu_only=True)
            program.return_(key="integer", value=program.run(fn=function, args=[value]))
            return program

        first = integer_program(10**400)
        second = integer_program(10**400 + 1)
        with Client(url) as client:
            await asyncio.to_thread(client.submit, first, idempotency_key="integer")
            with pytest.raises(KCoralError) as conflict:
                await asyncio.to_thread(client.submit, second, idempotency_key="integer")
            assert conflict.value.status_code == 409


def test_node_disconnect_is_a_terminal_unknown_outcome(tmp_path):
    asyncio.run(_node_disconnect_is_a_terminal_unknown_outcome(tmp_path))


async def _node_disconnect_is_a_terminal_unknown_outcome(tmp_path):
    async with running_router(tmp_path, "--job-dir", str(tmp_path / "jobs")) as url:
        async with running_app(url, "worker", tmp_path / "worker") as app:
            await ready_nodes(url, 1)
            with Client(url) as client:
                handle = await asyncio.to_thread(client.submit, program_with_result(delay=1))

                async def dispatched():
                    return (await asyncio.to_thread(handle.status)).state == "DISPATCHED"

                await wait_until(dispatched)
                await app.state.tunnel._channel.close()
                with pytest.raises(KCoralError) as failure:
                    await asyncio.to_thread(handle.result, timeout=5, poll_interval=0.02)
                assert failure.value.kind == "execution_unknown"
                assert (await asyncio.to_thread(handle.status)).state == "ERROR"


def test_graceful_shutdown_saves_dispatched_result(tmp_path):
    asyncio.run(_graceful_shutdown_saves_dispatched_result(tmp_path))


async def _graceful_shutdown_saves_dispatched_result(tmp_path):
    directory = tmp_path / "jobs"
    options = ("--job-dir", str(directory))
    async with running_router(tmp_path / "first", *options, include_process=True) as (url, process):
        async with running_app(url, "worker", tmp_path / "worker"):
            await ready_nodes(url, 1)
            with Client(url) as client:
                handle = await asyncio.to_thread(client.submit, program_with_result(delay=0.5))

                async def dispatched():
                    return (await asyncio.to_thread(handle.status)).state == "DISPATCHED"

                await wait_until(dispatched)
                process.terminate()

                async def saved():
                    with sqlite3.connect(directory / "jobs.sqlite3") as db:
                        record = json.loads(
                            db.execute(
                                "SELECT record FROM jobs WHERE id = ?", (handle.id,)
                            ).fetchone()[0]
                        )
                    return record["state"] == "COMPLETED"

                await wait_until(saved)
    async with running_router(tmp_path / "second", *options) as url:
        with Client(url) as client:
            assert (await asyncio.to_thread(client.get_job(handle.id).result, timeout=2)).completed


@pytest.mark.parametrize("job_id", ["../health", "", None, "A" * 32])
def test_invalid_handle_id(job_id):
    with Client("http://router") as client, pytest.raises(ValueError):
        client.get_job(job_id)
