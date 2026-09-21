import os
import subprocess
import sys
from pathlib import Path

import pytest

from kcoral._server import build_parser, config_from_args
from kcoral.app import _describe


def parse(argv):
    return config_from_args(build_parser().parse_args(argv))


def test_defaults():
    config = parse([])
    assert config.device == "gpu"
    assert config.gpus == [0]
    assert config.num_workers == 1
    assert config.log_dir == Path("logs")
    assert config.max_requests_per_worker == 1
    assert config.disk_cache_capacity_mbytes == 16 * 1024
    assert config.sandbox == "bubblewrap"
    assert config.sandbox_readonly_paths == []


def test_filesystem_sandbox_flags():
    config = parse(
        [
            "--sandbox",
            "bubblewrap",
            "--sandbox-readonly-path",
            "/opt/compiler",
            "--sandbox-readonly-path",
            "/opt/dependencies",
            "--max-requests-per-worker",
            "0",
        ]
    )
    assert config.sandbox == "bubblewrap"
    assert config.sandbox_readonly_paths == [Path("/opt/compiler"), Path("/opt/dependencies")]
    assert config.max_requests_per_worker == 0
    assert _describe(config)["sandbox_readonly_paths"] == ["/opt/compiler", "/opt/dependencies"]
    with pytest.raises(SystemExit, match="requires --sandbox"):
        parse(["--sandbox", "none", "--sandbox-readonly-path", "/opt/compiler"])
    assert parse(["--sandbox", "none"]).sandbox == "none"
    assert parse(["--sandbox-readonly-path", "/opt/compiler"]).sandbox == "bubblewrap"


def test_host_default_and_flag():
    assert build_parser().parse_args([]).host == "127.0.0.1"
    assert build_parser().parse_args(["--host", "0.0.0.0"]).host == "0.0.0.0"


def test_host_env():
    code = "from kcoral._server import build_parser; print(build_parser().parse_args([]).host)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "KCORAL_SERVER_HOST": "192.0.2.1",
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "python"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "192.0.2.1"


def test_outbound_tunnel_environment_is_used_and_token_is_redacted(monkeypatch):
    monkeypatch.setenv("KCORAL_ROUTER_ENDPOINT", "https://router.example.com/")
    monkeypatch.setenv("KCORAL_NODE_ID", "gpu-a")
    monkeypatch.setenv("KCORAL_NODE_TOKEN", "sensitive-node-token")
    config = parse([])
    assert config.router_endpoint == "https://router.example.com/"
    assert config.node_id == "gpu-a"
    assert config.node_token == "sensitive-node-token"
    assert _describe(config)["node_token"] == "<redacted>"


def test_outbound_tunnel_endpoint_and_node_id_are_configured_together():
    with pytest.raises(SystemExit, match="configured together"):
        parse(["--router", "http://router:9000/"])


def test_all_flags_reach_config():
    config = parse(
        [
            "--device",
            "gpu",
            "--gpus",
            "1,3",
            "--num-workers",
            "5",
            "--cache-capacity-bytes",
            "1234",
            "--disk-cache-dir",
            "/tmp/kcoral-files",
            "--disk-cache-capacity-mbytes",
            "5678",
            "--log-dir",
            "",
            "--default-timeout-seconds",
            "12",
            "--max-timeout-seconds",
            "34",
            "--worker-wait-timeout-seconds",
            "5",
            "--workers-per-gpu",
            "3",
            "--max-requests-per-worker",
            "7",
            "--worker-termination-grace-seconds",
            "2",
            "--max-request-bytes",
            "1000",
            "--max-response-bytes",
            "2000",
            "--output-limit-bytes",
            "300",
            "--max-output-limit-bytes",
            "400",
        ]
    )
    assert config.device == "gpu"
    assert config.gpus == [1, 3]
    assert config.num_workers == 5
    assert config.log_dir is None  # empty string disables logging
    assert config.cache_capacity_bytes == 1234
    assert config.disk_cache_dir == Path("/tmp/kcoral-files")
    assert config.disk_cache_capacity_mbytes == 5678
    assert config.default_timeout_seconds == 12
    assert config.max_timeout_seconds == 34
    assert config.worker_wait_timeout_seconds == 5
    assert config.workers_per_gpu == 3
    assert config.max_requests_per_worker == 7
    assert config.worker_termination_grace_seconds == 2
    assert config.max_request_bytes == 1000
    assert config.max_response_bytes == 2000
    assert config.output_limit_bytes == 300
    assert config.max_output_limit_bytes == 400


def test_bad_port_rejected():
    with pytest.raises(SystemExit):
        config_from_args(build_parser().parse_args(["--port", "70000"]))


def test_empty_gpus_rejected():
    with pytest.raises(SystemExit):
        parse(["--gpus", ","])


def test_cpu_mode_uses_workers_and_needs_no_gpus():
    config = parse(["--device", "cpu", "--num-workers", "4", "--gpus", ","])
    assert config.device == "cpu"
    assert config.gpus == []
    assert config.num_workers == 4


@pytest.mark.parametrize("flag", ["--num-workers", "--workers-per-gpu"])
def test_worker_count_must_be_positive(flag):
    with pytest.raises(SystemExit):
        parse([flag, "0"])


def test_negative_max_requests_per_worker_rejected():
    with pytest.raises(SystemExit):
        parse(["--max-requests-per-worker", "-1"])


def test_disk_cache_can_be_disabled_and_rejects_negative_budget():
    assert parse(["--disk-cache-dir", ""]).disk_cache_dir is None
    assert parse(["--disk-cache-capacity-mbytes", "0"]).disk_cache_capacity_mbytes == 0
    with pytest.raises(SystemExit):
        parse(["--disk-cache-capacity-mbytes", "-1"])


@pytest.mark.parametrize("xdg_cache_home", ["absolute", "relative", "", None])
def test_disk_cache_default_uses_only_absolute_xdg_cache_home(
    tmp_path, monkeypatch, xdg_cache_home
):
    from kcoral.config import ServerConfig

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    if xdg_cache_home == "absolute":
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        expected = tmp_path
    else:
        if xdg_cache_home is None:
            monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        else:
            monkeypatch.setenv("XDG_CACHE_HOME", xdg_cache_home)
        expected = tmp_path / "home" / ".cache"
    assert ServerConfig().disk_cache_dir == expected / "kcoral" / "files"
