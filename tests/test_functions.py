"""Remote decorators exercised through the real HTTP and worker protocol."""

import numpy as np
import pytest
from starlette.routing import compile_path
from test_client import _start_server
from test_client import server_url as server_url

from kcoral import Client, RemoteExecutionError, ServerConfig, create_app
from kcoral.testing import fake_runtime_factory


def echo(value):
    return value


def test_local_and_remote_argument_binding(server_url):
    with Client(server_url) as client:

        @client.function()
        def combine(a, /, b=2, *extra, scale=1, **labels):
            return {"total": (a + b + sum(extra)) * scale, "labels": labels}

        expected = {"total": 40, "labels": {"timeout": 3}}
        assert combine(1, 3, 6, scale=4, timeout=3) == expected
        assert combine.remote(1, 3, 6, scale=4, timeout=3) == expected
        assert combine.remote(1) == {"total": 3, "labels": {}}
        with pytest.raises(TypeError, match="missing"):
            combine.build_program()


@pytest.fixture(scope="module")
def proxy_server(tmp_path_factory):
    app = create_app(
        ServerConfig(
            gpus=[0], max_requests_per_worker=0, disk_cache_dir=tmp_path_factory.mktemp("proxy")
        ),
        runtime_factory=fake_runtime_factory,
    )
    paths = {"/execute": "/prefix/tasks/run", "/health": "/prefix/status"}
    for route in app.routes:
        if route.path in paths:
            route.path = paths[route.path]
            route.path_regex, route.path_format, route.param_convertors = compile_path(route.path)
    requests = []

    @app.middleware("http")
    async def record(request, call_next):
        requests.append((request.method, request.url.path, request.headers.get("authorization")))
        return await call_next(request)

    server, thread, url = _start_server(app)
    yield url, requests
    server.should_exit = True
    thread.join(timeout=10)


def test_custom_paths_headers_and_binary_transfers(proxy_server):
    url, requests = proxy_server
    requests.clear()
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    with Client(
        url + "/prefix",
        execute_path="tasks/run",
        health_path="/status",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        remote = client.function()(echo)
        assert client.health()["status"] == "ok"
        np.testing.assert_array_equal(remote.remote(value), value)
        assert remote.remote(b"binary\x00input") == b"binary\x00input"
        # The bound decorator leaves the borrowed client open.
        assert client.health()["status"] == "ok"
    assert all(auth == "Bearer test-token" for _, _, auth in requests)
    posts = [path for method, path, _ in requests if method == "POST"]
    assert len(posts) >= 4  # Tensor and bytes both negotiate cold uploads.
    assert set(posts) == {"/prefix/tasks/run"}


def test_literals_are_snapshotted_and_reference_shaped_dicts_stay_data(server_url):
    original = {"$ref": "not-a-register"}
    with Client(server_url) as client:
        remote = client.function()(echo)
        program = remote.build_program(original)
        original["$ref"] = "changed"
        assert client.execute(program).results["output"] == {"$ref": "not-a-register"}
        nested = {"items": [{"$ref": "also-literal"}], "value": None, "enabled": True}
        assert remote.remote(nested) == nested


def test_defaults_annotations_imports_and_nested_helpers(server_url):
    class LocalOnlyType:
        pass

    default = 9

    def compute(value: "LocalOnlyType" = default, *, offset=1) -> "LocalOnlyType":
        import math

        def inner(x):
            return math.sqrt(x) + offset

        return inner(value)

    with Client(server_url) as client:
        remote = client.function()(compute)
        assert remote.remote() == 4
        assert remote.remote(16, offset=2) == 6


def test_function_names_do_not_collide_with_generated_dispatcher(server_url):
    def values(value):
        return value

    def dict(value):
        return value

    with Client(server_url) as client:
        for fn in (values, dict):
            assert client.function()(fn).remote(42) == 42


def test_execution_results_and_failures(server_url):
    with Client(server_url) as client:

        @client.function(timeout=12, output_limit_bytes=1024, cpu_only=True)
        def report(value):
            import sys

            print("stdout marker")
            print("stderr marker", file=sys.stderr)
            if value < 0:
                raise ValueError("negative input")
            return value + 1

        result = report.execute(2)
        assert result.completed and result.results["output"] == 3
        assert "stdout marker" in result.stdout and "stderr marker" in result.stderr
        with pytest.raises(RemoteExecutionError, match="negative input") as caught:
            report.remote(-1)
        failed = caught.value.result
        assert failed.status == "FAILED" and failed.request_id
        assert "ValueError" in failed.error["traceback"]
        assert "stdout marker" in failed.stdout and "stderr marker" in failed.stderr
        assert report.execute(-1).status == "FAILED"


def test_rejects_external_state_and_nested_binary_arguments():
    captured = 4

    def closure(x):
        return x + captured

    def nested_dependency(x):
        return [np.asarray(item) for item in x]

    with Client("http://127.0.0.1:1") as client:
        for fn, match in [(closure, "capture"), (nested_dependency, "np")]:
            with pytest.raises(ValueError, match=match):
                client.function()(fn)
        with pytest.raises(TypeError, match="unsupported argument"):
            client.function()(echo).build_program({"nested": b"bytes"})
