import json

from fastapi.testclient import TestClient

from benchmark_server.app import create_app
from benchmark_server.config import ServerConfig
from benchmark_server.testing import fake_runtime_factory

STRUCTURAL_PROGRAM = {
    "instructions": [{"id": "x", "op": "run", "fn": "builtin.structural", "args": []}]
}


def _make_client(tmp_path):
    config = ServerConfig(gpus=[0], log_dir=tmp_path / "logs")
    return TestClient(create_app(config, runtime_factory=fake_runtime_factory))


def _read_events(tmp_path):
    run_dirs = list((tmp_path / "logs" / "runs").iterdir())
    assert len(run_dirs) == 1
    lines = (run_dirs[0] / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_request_lifecycle_events(tmp_path):
    with _make_client(tmp_path) as c:
        response = c.post("/benchmark", json=STRUCTURAL_PROGRAM)
        request_id = response.headers["x-request-id"]
    events = _read_events(tmp_path)
    assert [e["event"] for e in events] == [
        "server_started",
        "request_started",
        "request_finished",
        "server_stopped",
    ]
    assert events[0]["gpus"] == [0]
    finished = events[2]
    assert finished["request_id"] == request_id
    assert finished["http_status"] == 200 and finished["status"] == "COMPLETED"
    assert finished["gpu_id"] == 0
    assert finished["queue_ms"] >= 0 and finished["elapsed_ms"] >= 0


def test_worker_restart_event_on_timeout(tmp_path):
    program = {
        "instructions": [{"id": "s", "op": "run", "fn": "builtin.sleep", "args": [5.0]}],
        "options": {"timeout_seconds": 0.5},
    }
    with _make_client(tmp_path) as c:
        assert c.post("/benchmark", json=program).status_code == 504
    events = _read_events(tmp_path)
    restarts = [e for e in events if e["event"] == "worker_restarted"]
    assert len(restarts) == 1
    assert restarts[0]["reason"] == "timeout" and restarts[0]["gpu_id"] == 0
    finished = next(e for e in events if e["event"] == "request_finished")
    assert finished["http_status"] == 504 and finished["error"] == "timeout"


def test_invalid_request_is_logged(tmp_path):
    with _make_client(tmp_path) as c:
        assert c.post("/benchmark", json={"instructions": []}).status_code == 400
    finished = next(e for e in _read_events(tmp_path) if e["event"] == "request_finished")
    assert finished["http_status"] == 400 and finished["level"] == "WARNING"


def test_logging_disabled_without_log_dir(tmp_path):
    config = ServerConfig(gpus=[0])  # log_dir=None
    with TestClient(create_app(config, runtime_factory=fake_runtime_factory)) as c:
        assert c.post("/benchmark", json=STRUCTURAL_PROGRAM).status_code == 200
    assert not (tmp_path / "logs").exists()
