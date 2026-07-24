from pathlib import Path

import pytest

from benchmark_server.__main__ import build_parser, config_from_args


def parse(argv):
    return config_from_args(build_parser().parse_args(argv))


def test_defaults():
    config = parse([])
    assert config.gpus == [0]
    assert config.cache_dir == Path("cache")
    assert config.log_dir == Path("logs")


def test_all_flags_reach_config():
    config = parse(
        [
            "--gpus",
            "1,3",
            "--cache-dir",
            "/data/cache",
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
    assert config.gpus == [1, 3]
    assert config.cache_dir == Path("/data/cache")
    assert config.log_dir is None  # empty string disables logging
    assert config.cache_capacity_bytes == 1234
    assert config.default_timeout_seconds == 12
    assert config.max_timeout_seconds == 34
    assert config.worker_wait_timeout_seconds == 5
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
