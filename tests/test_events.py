import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from support.programs import harness_instructions

from kcoral.app import create_app
from kcoral.config import ServerConfig
from kcoral.events import EventLogger
from kcoral.pool import PoolBusy
from kcoral.testing import fake_runtime_factory

STRUCTURAL_PROGRAM = {
    "instructions": [
        *harness_instructions("value", "structural"),
        {"op": "return", "key": "value", "value": {"$ref": "value"}},
    ]
}


def failing_runtime_factory():
    """Module level, so ``spawn`` can pickle it into the worker it fails in."""
    raise RuntimeError("simulated runtime init failure")


def _make_client(tmp_path, **overrides):
    settings = dict(
        gpus=[0],
        log_dir=tmp_path / "logs",
        max_requests_per_worker=0,
        workers_per_gpu=1,  # so a test can name the worker it expects
        log_console=False,  # the log file is what these assert on
    )
    config = ServerConfig(**{**settings, **overrides})
    return TestClient(create_app(config, runtime_factory=fake_runtime_factory))


def _post(client, program):
    return client.post(
        "/execute",
        files={"program": (None, json.dumps(program), "application/json")},
    )


def _run_dir(tmp_path):
    run_dirs = list((tmp_path / "logs" / "runs").iterdir())
    assert len(run_dirs) == 1
    return run_dirs[0]


def _read_events(tmp_path):
    lines = (_run_dir(tmp_path) / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _named(events, name):
    return [event for event in events if event["event"] == name]


def _one(events, name):
    matches = _named(events, name)
    assert len(matches) == 1, f"expected one {name}, got {len(matches)}"
    return matches[0]


def _finished(tmp_path):
    return _one(_read_events(tmp_path), "request_finished")


# --- the trace a request leaves ---------------------------------------------


def test_request_trace_runs_arrival_to_worker_to_outcome(tmp_path):
    with _make_client(tmp_path) as client:
        response = _post(client, STRUCTURAL_PROGRAM)
        request_id = response.headers["x-request-id"]
    events = _read_events(tmp_path)

    assert [event["event"] for event in events if event["event"].startswith("request_")] == [
        "request_received",
        "request_accepted",
        "request_routed",
        "request_finished",
    ]
    assert {event["request_id"] for event in events if "request_id" in event} == {request_id}
    routed, finished = _one(events, "request_routed"), _one(events, "request_finished")
    # The worker named on the way in is the one credited on the way out.
    assert routed["worker_id"] == finished["worker_id"] == "gpu0/w0"
    assert routed["pid"] and routed["generation"] == 1
    assert finished["http_status"] == 200 and finished["status"] == "COMPLETED"
    assert finished["finish_reason"] == "completed" and finished["gpu_id"] == 0
    assert finished["queue_ms"] >= 0 and finished["elapsed_ms"] >= 0


def test_accepted_record_describes_the_workload(tmp_path):
    program = {
        "instructions": [
            {"op": "upload", "id": "m", "kind": "module", "language": "python", "source": "x = 1"},
            *harness_instructions("value", "structural"),
            {"op": "return", "key": "value", "value": {"$ref": "value"}},
        ],
        "options": {"timeout_seconds": 12.0},
    }
    with _make_client(tmp_path) as client:
        assert _post(client, program).status_code == 200
    accepted = _one(_read_events(tmp_path), "request_accepted")
    assert accepted["instructions"] == 5
    assert accepted["ops"] == {"upload": 2, "get_function": 1, "run": 1, "return": 1}
    assert accepted["uploads"] == {"module:python": 2}
    assert "builtins" not in accepted
    assert accepted["timeout_seconds"] == 12.0
    assert accepted["request_bytes"] > 0


def test_arrival_is_recorded_before_a_request_can_be_rejected(tmp_path):
    with _make_client(tmp_path) as client:
        assert _post(client, {"instructions": []}).status_code == 400
    events = _read_events(tmp_path)
    assert _one(events, "request_received")["request_id"]
    assert not _named(events, "request_accepted")  # it never parsed
    finished = _one(events, "request_finished")
    assert finished["http_status"] == 400 and finished["level"] == "WARNING"
    assert finished["finish_reason"] == "rejected" and finished["error_kind"] == "invalid_request"


# --- why the worker answered -------------------------------------------------


def test_program_failure_records_the_failing_instruction(tmp_path):
    program = {"instructions": [*harness_instructions("bad", "stale_cuda_error")]}
    with _make_client(tmp_path) as client:
        assert _post(client, program).status_code == 200
    finished = _finished(tmp_path)
    assert finished["finish_reason"] == "program_failed" and finished["status"] == "FAILED"
    assert finished["error_kind"] == "runtime"
    assert "cudaErrorInvalidValue" in finished["error_message"]
    assert finished["instruction_index"] == 2 and finished["instruction_op"] == "run"
    assert finished["instruction_id"] == "bad"
    # The client's kernel misbehaved, not the server: not an ERROR.
    assert finished["level"] == "INFO"


def test_request_limit_is_the_reason_a_healthy_worker_retires(tmp_path):
    config = ServerConfig(  # the default limit of one request per worker
        gpus=[0], log_dir=tmp_path / "logs", workers_per_gpu=1
    )
    with TestClient(create_app(config, runtime_factory=fake_runtime_factory)) as client:
        assert _post(client, STRUCTURAL_PROGRAM).status_code == 200
        # Recycling is allowed while serving, but is interrupted by shutdown.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if sum(e["event"] == "worker_ready" for e in _read_events(tmp_path)) == 2:
                break
            time.sleep(0.01)
        else:
            pytest.fail("replacement never became ready")
    events = _read_events(tmp_path)
    finished = _one(events, "request_finished")
    assert finished["finish_reason"] == "request_limit" and finished["status"] == "COMPLETED"
    retired = _one(events, "worker_retired")
    assert retired["reason"] == "request_limit" and retired["worker_id"] == "gpu0/w0"
    assert [event["event"] for event in events].count("worker_ready") == 2  # first, replacement


def test_poisoned_context_is_the_reason_a_failed_cleanup_retires_a_worker(tmp_path):
    program = {"instructions": [*harness_instructions("bad", "poison")]}
    with _make_client(tmp_path) as client:
        assert _post(client, program).status_code == 200
    events = _read_events(tmp_path)
    finished = _one(events, "request_finished")
    assert finished["finish_reason"] == "poisoned_context" and finished["status"] == "FAILED"
    assert finished["level"] == "WARNING"  # the server's problem, not the program's
    assert _one(events, "worker_retired")["reason"] == "poisoned_context"


def test_timeout_is_the_reason_a_worker_never_answered(tmp_path):
    program = {
        "instructions": [*harness_instructions("sleep", "sleep", [5])],
        "options": {"timeout_seconds": 0.5},
    }
    with _make_client(tmp_path) as client:
        assert _post(client, program).status_code == 504
    events = _read_events(tmp_path)
    finished = _one(events, "request_finished")
    assert finished["finish_reason"] == "timeout" and finished["http_status"] == 504
    assert finished["worker_id"] == "gpu0/w0" and finished["timeout_seconds"] == 0.5
    assert _one(events, "worker_retired")["reason"] == "timeout"


def test_crash_is_the_reason_a_worker_exited_mid_request(tmp_path):
    program = {"instructions": [*harness_instructions("boom", "crash")]}
    with _make_client(tmp_path) as client:
        assert _post(client, program).status_code == 200
    events = _read_events(tmp_path)
    finished = _one(events, "request_finished")
    assert finished["finish_reason"] == "crashed" and finished["status"] == "FAILED"
    assert finished["level"] == "ERROR" and finished["exitcode"] == 1
    assert finished["instruction_op"] == "run" and finished["instruction_id"] == "boom"
    assert _one(events, "worker_retired")["reason"] == "crashed"


def test_saturation_is_the_reason_no_worker_ran_the_request(tmp_path, monkeypatch):
    with _make_client(tmp_path) as client:
        monkeypatch.setattr(
            client.app.state.pool,
            "submit",
            lambda *args, **kwargs: (_ for _ in ()).throw(PoolBusy("busy", queue_ms=30.0)),
        )
        assert _post(client, STRUCTURAL_PROGRAM).status_code == 503
    finished = _finished(tmp_path)
    assert finished["finish_reason"] == "no_worker" and finished["http_status"] == 503
    assert finished["queue_ms"] == 30.0


def test_a_replacement_that_fails_is_not_silent(tmp_path):
    """The pool carries on with a dead worker, which the next request revives.
    Unlogged, that reads as the pool being mysteriously slow."""
    config = ServerConfig(  # the default limit of one, so answering retires it
        gpus=[0], log_dir=tmp_path / "logs", workers_per_gpu=1, log_console=False
    )
    app = create_app(config, runtime_factory=fake_runtime_factory)
    with TestClient(app) as client:
        worker = app.state.pool._workers[0]
        worker._start_process = _raise_respawn_failure
        assert _post(client, STRUCTURAL_PROGRAM).status_code == 200
        _settled(app.state.pool)
    events = _read_events(tmp_path)
    assert _one(events, "worker_retired")["reason"] == "request_limit"
    respawn = _one(events, "worker_failed")
    assert respawn["phase"] == "respawn" and "no process to be had" in respawn["error"]
    failed = _one(events, "worker_replace_failed")
    assert failed["level"] == "ERROR" and failed["reason"] == "request_limit"
    assert failed["worker_id"] == "gpu0/w0"


def _raise_respawn_failure():
    raise RuntimeError("no process to be had")


def _settled(pool, timeout=30.0):
    """Wait out the background replacement, which happens after the answer."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with pool._replacing_lock:
            if not pool._replacing:
                return
        time.sleep(0.01)
    raise AssertionError("a worker replacement never finished")


# Its cpu_only call sleeps first, so a neighbour can take the GPU before it is touched.
CPU_ONLY_TOUCHING_GPU = (
    "import time\n"
    "from kcoral.testing import simulate_cuda_call\n\n"
    "def main():\n"
    "    time.sleep(0.5)\n"
    "    simulate_cuda_call('cudaMalloc')\n"
)
VIOLATING_PROGRAM = {
    "instructions": [
        {"op": "upload", "id": "m", "kind": "module", "source": CPU_ONLY_TOUCHING_GPU},
        {
            "op": "get_function",
            "id": "fn",
            "module": {"$ref": "m"},
            "name": "main",
            "cpu_only": True,
        },
        {"op": "run", "id": "call", "fn": {"$ref": "fn"}},
    ]
}


def test_a_cpu_only_function_touching_the_gpu_is_a_warning_naming_both_requests(tmp_path):
    holding = {"instructions": [*harness_instructions("hold", "sleep", [1.0])]}
    with _make_client(tmp_path, workers_per_gpu=2) as client:
        alone = _post(client, VIOLATING_PROGRAM).json()
        with ThreadPoolExecutor(max_workers=2) as pool:
            violator = pool.submit(_post, client, VIOLATING_PROGRAM)
            time.sleep(0.2)  # into its cpu_only call, with the GPU released
            holder = pool.submit(_post, client, holding)
            violator, holder = violator.result().json(), holder.result().json()

    assert alone["status"] == "FAILED" and alone["error"]["interfered_request_id"] is None
    assert holder["status"] == "COMPLETED"
    assert violator["status"] == "FAILED"
    error = violator["error"]
    assert error["kind"] == "gpu_access" and error["cuda_call"] == "cudaMalloc"
    assert error["interfered_request_id"] == holder["request_id"]

    warnings = _named(_read_events(tmp_path), "gpu_access_violation")
    assert [warning["interfered_request_id"] for warning in warnings] == [
        None,
        holder["request_id"],
    ]
    assert warnings[1]["level"] == "WARNING"
    assert warnings[1]["request_id"] == violator["request_id"]
    assert warnings[1]["instruction_id"] == "call" and warnings[1]["cuda_call"] == "cudaMalloc"


# --- failures outside a request ----------------------------------------------


def test_a_server_that_cannot_start_says_so_on_disk(tmp_path):
    config = ServerConfig(gpus=[0], log_dir=tmp_path / "logs", workers_per_gpu=1)
    with pytest.raises(Exception):
        with TestClient(create_app(config, runtime_factory=failing_runtime_factory)):
            pass
    events = _read_events(tmp_path)
    assert _one(events, "server_started")["config"]["gpus"] == [0]
    failed = _one(events, "server_start_failed")
    assert failed["level"] == "ERROR"
    assert "simulated runtime init failure" in failed["error"] + failed["traceback"]
    assert _one(events, "worker_failed")["phase"] == "startup"


def test_an_unhandled_front_end_error_still_finishes_the_request(tmp_path, monkeypatch):
    import kcoral.app as app_module

    def boom(payload, parts):
        raise RuntimeError("simulated front-end bug")

    monkeypatch.setattr(app_module, "_encode_response", boom)
    config = ServerConfig(
        gpus=[0], log_dir=tmp_path / "logs", max_requests_per_worker=0, workers_per_gpu=1
    )
    client = TestClient(
        create_app(config, runtime_factory=fake_runtime_factory), raise_server_exceptions=False
    )
    with client:
        assert _post(client, STRUCTURAL_PROGRAM).status_code == 500
    finished = _finished(tmp_path)
    assert finished["finish_reason"] == "server_error" and finished["error_kind"] == "unhandled"
    assert "simulated front-end bug" in finished["error_message"]
    assert "simulated front-end bug" in finished["traceback"]
    assert finished["request_id"] == _one(_read_events(tmp_path), "request_received")["request_id"]


# --- the log itself ----------------------------------------------------------


def test_console_mirror_follows_its_setting(tmp_path, capfd):
    with _make_client(tmp_path, log_console=True) as client:
        _post(client, STRUCTURAL_PROGRAM)
    console = capfd.readouterr().err
    assert "request_routed" in console and "worker_id=gpu0/w0" in console
    assert "finish_reason=completed" in console

    with _make_client(tmp_path / "b", log_console=False) as client:
        _post(client, STRUCTURAL_PROGRAM)
    assert "request_finished" not in capfd.readouterr().err


def test_log_programs_keeps_the_source_beside_the_log(tmp_path):
    with _make_client(tmp_path) as client:
        request_id = _post(client, STRUCTURAL_PROGRAM).headers["x-request-id"]
    accepted = _one(_read_events(tmp_path), "request_accepted")
    assert accepted["program"] == f"{request_id}.json"
    kept = json.loads((_run_dir(tmp_path) / "programs" / accepted["program"]).read_text())
    assert kept == STRUCTURAL_PROGRAM


def test_one_unencodable_field_does_not_cost_the_event(tmp_path):
    log = EventLogger(tmp_path / "logs")
    log.emit("odd", request_id="r1", value=object())
    log.emit("nan", value=float("nan"))
    log.close()
    events = _read_events(tmp_path)
    assert events[0]["event"] == "odd" and events[0]["request_id"] == "r1"
    assert "object object at" in events[0]["value"]
    assert "nan" in events[1]["unencodable_fields"]


def test_no_log_dir_writes_nothing_to_disk(tmp_path):
    config = ServerConfig(gpus=[0], log_dir=None)  # the console mirror stays on
    with TestClient(create_app(config, runtime_factory=fake_runtime_factory)) as client:
        assert _post(client, STRUCTURAL_PROGRAM).status_code == 200
    assert not (tmp_path / "logs").exists()


def test_a_record_stays_cheap_enough_to_leave_on(tmp_path):
    """A guard against the one regression that would matter: an ``fsync`` per
    record, or anything else that turns a buffered write into a disk round trip.
    The bound is ~20x the measured cost, so only a change of that kind trips it."""
    log = EventLogger(tmp_path / "logs")
    fields = dict(
        request_id="r" * 36, worker_id="gpu0/w3", finish_reason="completed", elapsed_ms=1.0
    )
    for _ in range(200):  # warm the page cache and the JSON encoder
        log.emit("request_finished", **fields)
    started = time.perf_counter()
    for _ in range(2000):
        log.emit("request_finished", **fields)
    per_record_us = (time.perf_counter() - started) / 2000 * 1e6
    log.close()
    assert per_record_us < 150, f"{per_record_us:.1f} us/record"


def test_a_killed_worker_leaves_its_output_behind(tmp_path):
    program = {"instructions": [*harness_instructions("boom", "crash_after_output")]}
    with _make_client(tmp_path) as client:
        assert _post(client, program).status_code == 200
    finished = _finished(tmp_path)
    assert finished["finish_reason"] == "crashed"
    assert "simulated device-side assert" in finished["output_tail"]
    # Consumed with it: the capture belongs to the request that produced it.
    assert list((_run_dir(tmp_path) / "output").iterdir()) == []
