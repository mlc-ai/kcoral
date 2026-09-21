"""Service launchers and real supervised execution, without a GPU."""

import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from kcoral import Client, Program, service_cli
from kcoral.__main__ import main
from kcoral._server import build_parser, config_from_args

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(monkeypatch):
    for name in ("KCORAL_ROUTER_ENDPOINT", "KCORAL_NODE_ID", "KCORAL_NODE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    monkeypatch.setattr(service_cli, "exec_service", lambda *a, **kw: calls.append((a, kw)))
    return calls


def test_root_help_and_no_implicit_server(capsys):
    main([])
    assert "{server,router}" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        main(["--gpus", "0"])
    assert exc.value.code == 2


def test_router_forwards_native_options(launch):
    main(["router", "--port", "9001", "--max-queued-requests=20"])
    assert launch == [(("kcoral-router", ["--port", "9001", "--max-queued-requests=20"]), {})]


def test_standalone_preserves_worker_options_and_python_environment(launch):
    main(
        [
            "server",
            "--device=cpu",
            "--num-workers",
            "3",
            "--host",
            "0.0.0.0",
            "--port",
            "8123",
            "--log-dir=logs with spaces",
            "--no-log-console",
            "--no-log-programs",
            "--sandbox-readonly-path=/opt/compiler one",
            "--sandbox-readonly-path=/opt/compiler-two",
            "--health-interval-seconds=0.25",
            "--restart-jitter=0",
        ]
    )
    (name, argv), kwargs = launch[0]
    assert name == "kcoral-node"
    assert argv[:2] == ["--server-url", "http://127.0.0.1:8123/"]
    assert "--router-endpoint" not in argv
    boundary = argv.index("--")
    assert argv[boundary + 1 : boundary + 4] == [sys.executable, "-m", "kcoral._server"]
    worker_args = argv[boundary + 4 :]
    config = config_from_args(build_parser().parse_args(worker_args))
    assert config.device == "cpu" and config.num_workers == 3
    assert config.log_dir == Path("logs with spaces")
    assert config.log_console is False and config.log_programs is False
    assert config.sandbox_readonly_paths == [Path("/opt/compiler one"), Path("/opt/compiler-two")]
    assert "--host=0.0.0.0" in worker_args
    assert argv[argv.index("--health-interval-seconds") + 1] == "0.25"
    assert "KCORAL_ROUTER_ENDPOINT" not in kwargs["env"]


def test_routing_flags_override_environment_without_exposing_token(launch, monkeypatch):
    monkeypatch.setenv("KCORAL_ROUTER_ENDPOINT", "http://old:9000")
    monkeypatch.setenv("KCORAL_NODE_ID", "old")
    monkeypatch.setenv("KCORAL_NODE_TOKEN", "old-token")
    main(
        [
            "server",
            "--router",
            "http://new:9000",
            "--node-id",
            "new",
            "--node-token",
            "new-token",
        ]
    )
    (_, argv), kwargs = launch[0]
    assert argv[argv.index("--router-endpoint") + 1] == "http://new:9000"
    assert argv[argv.index("--node-id") + 1] == "new"
    assert "new-token" not in " ".join(argv)
    assert kwargs["env"]["KCORAL_NODE_TOKEN"] == "new-token"
    assert "KCORAL_ROUTER_ENDPOINT" not in kwargs["env"]
    assert "KCORAL_NODE_ID" not in kwargs["env"]


def test_routing_environment_alone_is_supported(launch, monkeypatch):
    monkeypatch.setenv("KCORAL_ROUTER_ENDPOINT", "http://router:9000")
    monkeypatch.setenv("KCORAL_NODE_ID", "gpu-a")
    main(["server"])
    argv = launch[0][0][1]
    assert argv[argv.index("--router-endpoint") + 1] == "http://router:9000"
    assert argv[argv.index("--node-id") + 1] == "gpu-a"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--router", "http://router:9000"],
        ["--node-id", "gpu-a"],
        ["--health-interval-seconds", "0"],
        ["--failure-threshold", "0"],
        ["--restart-jitter", "0.6"],
        ["--health-timeout-seconds", "nan"],
        ["--restart-min-delay-seconds", "20", "--restart-max-delay-seconds", "10"],
        ["--port", "0"],
        ["--num-workers", "0"],
    ],
)
def test_invalid_settings_never_start_a_process(launch, arguments):
    with pytest.raises(SystemExit):
        main(["server", *arguments])
    assert not launch


def test_service_helper_prefers_current_environment_and_replaces_process(monkeypatch):
    searched, executed = [], []
    monkeypatch.setattr(
        service_cli.shutil, "which", lambda name, path: searched.append(path) or "/native/helper"
    )
    monkeypatch.setattr(service_cli.os, "execve", lambda *args: executed.append(args))
    service_cli.exec_service("kcoral-router", ["--port", "9000"], env={"EXPLICIT": "1"})
    assert searched[0].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    assert executed == [("/native/helper", ["/native/helper", "--port", "9000"], {"EXPLICIT": "1"})]


def test_missing_helper_has_installation_instructions(monkeypatch):
    monkeypatch.setattr(service_cli.shutil, "which", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit, match="cargo install --locked"):
        service_cli.exec_service("kcoral-node", [])


def _binary_dir():
    binary = Path(os.environ.get("KCORAL_NODE_BIN", ROOT / "target/debug/kcoral-node"))
    if not binary.is_file() or not binary.with_name("kcoral-router").is_file():
        if os.environ.get("KCORAL_REQUIRE_GATEWAY_TESTS") == "1":
            pytest.fail("build both native service binaries before running these tests")
        pytest.skip("build native services to test the public launchers")
    return binary.parent


def _port(host="127.0.0.1", family=socket.AF_INET):
    with socket.socket(family) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _wait(check, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.1)
    pytest.fail("service condition timed out")


@contextmanager
def _running(tmp_path, name, *args, extra_env=None):
    log_path = tmp_path / (name + ".log")
    env = {k: v for k, v in os.environ.items() if not k.startswith("KCORAL_")}
    env.update(
        {
            "PYTHONPATH": str(ROOT / "python"),
            "PATH": str(_binary_dir()) + os.pathsep + os.environ.get("PATH", ""),
            **(extra_env or {}),
        }
    )
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "kcoral", *args], env=env, stdout=log, stderr=log
        )
        try:
            yield proc, log_path
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            assert proc.returncode == 0, log_path.read_text()


def _healthy(url, process, log_path, previous=None):
    assert process.poll() is None, log_path.read_text()
    try:
        with httpx.Client(trust_env=False) as client:
            response = client.get(url + "/health", timeout=1)
        if response.status_code != 200:
            return None
        value = response.json()
        return value if value.get("instance_id") != previous or previous is None else None
    except httpx.TransportError:
        return None


def _program():
    program = Program()
    module = program.upload(id="module", kind="module", source="def answer():\n    return 42\n")
    fn = program.get_function(id="fn", module=module, name="answer")
    result = program.run(id="result", fn=fn)
    program.return_(key="answer", value=result)
    return program


def _server_options(tmp_path, port):
    return [
        "server",
        "--device",
        "cpu",
        "--num-workers",
        "1",
        "--sandbox",
        "none",
        "--port",
        str(port),
        "--log-dir",
        str(tmp_path / "events"),
        "--disk-cache-dir",
        "",
        "--health-interval-seconds",
        "0.1",
        "--restart-min-delay-seconds",
        "0.1",
        "--restart-max-delay-seconds",
        "0.1",
        "--startup-grace-seconds",
        "10",
        "--termination-grace-seconds",
        "0.1",
    ]


@pytest.mark.parametrize("routed", [False, True])
def test_public_server_executes_restarts_and_shuts_down(tmp_path, routed):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack

    server_port = _port()
    direct_url = f"http://127.0.0.1:{server_port}"
    with ExitStack() as stack:
        options = _server_options(tmp_path, server_port)
        if routed:
            router_port = _port()
            url = f"http://127.0.0.1:{router_port}"
            router, router_log = stack.enter_context(
                _running(
                    tmp_path,
                    "router",
                    "router",
                    "--port",
                    str(router_port),
                    "--recovery-threshold",
                    "1",
                    extra_env={"KCORAL_NODE_TOKEN": "test-token"},
                )
            )
            options += ["--router", url, "--node-id", "cpu-a", "--node-token", "test-token"]
        else:
            url = direct_url
        server, server_log = stack.enter_context(_running(tmp_path, "server", *options))
        health = _wait(lambda: _healthy(direct_url, server, server_log))
        if routed:
            _wait(lambda: _healthy(url, router, router_log))
        with Client(url) as client:
            assert client.execute(_program()).results == {"answer": 42}
        # The public launcher execs the supervisor, whose direct child is the
        # Python server. Kill it, then prove a new generation can execute work.
        children = {
            int(pid)
            for thread in Path(f"/proc/{server.pid}/task").iterdir()
            for pid in (thread / "children").read_text().split()
        }
        assert len(children) == 1
        child = children.pop()
        os.kill(child, signal.SIGKILL)
        _wait(lambda: _healthy(direct_url, server, server_log, health["instance_id"]))
        if routed:
            # A fresh slot and a fresh status report must agree before routing.
            def recovered():
                try:
                    with Client(url) as client:
                        return client.execute(_program()).results == {"answer": 42}
                except Exception:
                    return False

            _wait(recovered)
        else:
            with Client(url) as client:
                assert client.execute(_program()).results == {"answer": 42}
        assert "restarting KCoral Server" in server_log.read_text()
        # A normal stop must finish accepted work even when it takes longer
        # than the grace period used for unhealthy-service restarts.
        started = tmp_path / "request-started"
        pending = Program()
        module = pending.upload(
            id="pending_module",
            kind="module",
            source=(
                "import time\nfrom pathlib import Path\n"
                f"def finish():\n    Path({str(started)!r}).touch()\n"
                "    time.sleep(0.8)\n    return 'finished'\n"
            ),
        )
        fn = pending.get_function(id="finish", module=module, name="finish")
        result = pending.run(id="result", fn=fn)
        pending.return_(key="value", value=result)
        with ThreadPoolExecutor(max_workers=1) as pool, Client(url) as client:
            future = pool.submit(client.execute, pending, timeout_seconds=10)
            _wait(started.exists)
            server.send_signal(signal.SIGTERM if routed else signal.SIGINT)
            assert future.result(timeout=15).results == {"value": "finished"}
            assert server.wait(timeout=15) == 0

    assert not Path(f"/proc/{child}").exists()


@pytest.mark.parametrize("host", ["0.0.0.0", "::1", "::", "interface"])
def test_supervisor_health_with_bind_addresses(tmp_path, host):
    if host == "interface":
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 9))
            host = sock.getsockname()[0]
    ipv6 = ":" in host
    target = "::1" if ipv6 else ("127.0.0.1" if host == "0.0.0.0" else host)
    try:
        port = _port(target, socket.AF_INET6 if ipv6 else socket.AF_INET)
    except OSError:
        pytest.skip("bind address is unavailable on this host")
    url = f"http://[{target}]:{port}" if ipv6 else f"http://{target}:{port}"
    with _running(tmp_path, "bind", *_server_options(tmp_path, port), "--host", host) as (
        proc,
        log,
    ):
        _wait(lambda: _healthy(url, proc, log))
        # Observe multiple real probes after startup; a 403 or failed bind must
        # not be hidden by the startup grace period.
        time.sleep(0.5)
        assert (
            "health check failed" not in log.read_text().split("Application startup complete.")[-1]
        )
