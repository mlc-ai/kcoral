import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark_server.__main__ import build_parser, config_from_args


def parse(argv):
    return config_from_args(build_parser().parse_args(argv))


def test_defaults():
    config = parse([])
    assert config.device == "gpu"
    assert config.gpus == [0]
    assert config.num_workers == 1
    assert config.log_dir == Path("logs")
    assert config.max_requests_per_worker == 1


def test_host_default_and_flag():
    assert build_parser().parse_args([]).host == "127.0.0.1"
    assert build_parser().parse_args(["--host", "0.0.0.0"]).host == "0.0.0.0"


def test_host_env():
    code = (
        "from benchmark_server.__main__ import build_parser; "
        "print(build_parser().parse_args([]).host)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "BENCH_HOST": "192.0.2.1",
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "python"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "192.0.2.1"


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
