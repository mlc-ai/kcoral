import json

from fastapi.testclient import TestClient

from benchmark_server.app import create_app
from benchmark_server.config import ServerConfig
from benchmark_server.testing import fake_runtime_factory

STRUCTURAL_PROGRAM = {
    "instructions": [
        {"op": "run", "id": "value", "fn": "builtin.structural", "gpu": "auto"},
        {"op": "return", "key": "value", "value": {"$ref": "value"}},
    ]
}


def _make_client(tmp_path):
    config = ServerConfig(gpus=[0], log_dir=tmp_path / "logs", max_requests_per_worker=0)
    return TestClient(create_app(config, runtime_factory=fake_runtime_factory))


def _post(client, program):
    return client.post(
        "/execute",
        files={"program": (None, json.dumps(program), "application/json")},
    )


def _read_events(tmp_path):
    run_dirs = list((tmp_path / "logs" / "runs").iterdir())
    assert len(run_dirs) == 1
    lines = (run_dirs[0] / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_request_lifecycle_events(tmp_path):
    with _make_client(tmp_path) as client:
        response = _post(client, STRUCTURAL_PROGRAM)
        request_id = response.headers["x-request-id"]
    events = _read_events(tmp_path)
    assert [event["event"] for event in events] == [
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
        "instructions": [
            {
                "op": "run",
                "id": "sleep",
                "fn": "builtin.sleep",
                "args": [5],
                "gpu": "auto",
            }
        ],
        "options": {"timeout_seconds": 0.5},
    }
    with _make_client(tmp_path) as client:
        assert _post(client, program).status_code == 504
    events = _read_events(tmp_path)
    restarts = [event for event in events if event["event"] == "worker_restarted"]
    assert len(restarts) == 1
    assert restarts[0]["reason"] == "timeout" and restarts[0]["gpu_id"] == 0
    finished = next(event for event in events if event["event"] == "request_finished")
    assert finished["http_status"] == 504 and finished["error"] == "timeout"


def test_worker_restart_event_on_poisoned_context(tmp_path):
    program = {"instructions": [{"op": "run", "id": "bad", "fn": "builtin.poison", "gpu": "auto"}]}
    with _make_client(tmp_path) as client:
        response = _post(client, program)
    assert response.status_code == 200
    assert response.json()["error"]["kind"] == "runtime"
    events = _read_events(tmp_path)
    restart = next(event for event in events if event["event"] == "worker_restarted")
    assert restart["reason"] == "poisoned_context" and restart["gpu_id"] == 0
    finished = next(event for event in events if event["event"] == "request_finished")
    assert finished["http_status"] == 200 and finished["status"] == "FAILED"


def test_worker_restart_event_on_request_limit(tmp_path):
    config = ServerConfig(gpus=[0], log_dir=tmp_path / "logs")
    with TestClient(create_app(config, runtime_factory=fake_runtime_factory)) as client:
        assert _post(client, STRUCTURAL_PROGRAM).status_code == 200
    restart = next(
        event for event in _read_events(tmp_path) if event["event"] == "worker_restarted"
    )
    assert restart["reason"] == "request_limit" and restart["gpu_id"] == 0


def test_invalid_request_is_logged(tmp_path):
    with _make_client(tmp_path) as client:
        assert _post(client, {"instructions": []}).status_code == 400
    finished = next(
        event for event in _read_events(tmp_path) if event["event"] == "request_finished"
    )
    assert finished["http_status"] == 400 and finished["level"] == "WARNING"


def test_logging_disabled_without_log_dir(tmp_path):
    with TestClient(
        create_app(ServerConfig(gpus=[0]), runtime_factory=fake_runtime_factory)
    ) as client:
        assert _post(client, STRUCTURAL_PROGRAM).status_code == 200
    assert not (tmp_path / "logs").exists()
