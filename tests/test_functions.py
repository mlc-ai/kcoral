"""Remote decorators exercised through the real HTTP and worker protocol."""

import inspect
import json

import httpx
import numpy as np
import pytest
from starlette.routing import compile_path
from test_client import _start_server
from test_client import server_url as server_url

from kcoral import (
    Client,
    KCoralError,
    RemoteExecutionError,
    RemoteFunction,
    ServerConfig,
    create_app,
    function,
)
from kcoral.multipart import parse_multipart
from kcoral.testing import fake_runtime_factory


@function()
def combine(a, /, b=2, *extra, scale=1, **labels):
    """Exercise all Python argument kinds."""
    return {"total": (a + b + sum(extra)) * scale, "labels": labels}


def echo(value):
    return value


@function()
def factorial(n):
    return 1 if n <= 1 else n * factorial(n - 1)


def test_local_and_remote_calls_preserve_binding_and_metadata(server_url, monkeypatch):
    monkeypatch.setenv("KCORAL_URL", server_url)
    assert isinstance(combine, RemoteFunction)
    assert combine.__name__ == "combine"
    assert combine.__doc__ == "Exercise all Python argument kinds."
    assert inspect.signature(combine) == inspect.signature(combine.__wrapped__)
    expected = {"total": 40, "labels": {"timeout": 3, "name": "sample"}}
    assert combine(1, 3, 6, scale=4, timeout=3, name="sample") == expected
    assert combine.remote(1, 3, 6, scale=4, timeout=3, name="sample") == expected
    assert combine.remote(1) == {"total": 3, "labels": {}}
    assert factorial.remote(5) == factorial(5) == 120


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


def test_endpoint_prefix_custom_paths_headers_and_binary_cache_retry(proxy_server, monkeypatch):
    url, requests = proxy_server
    requests.clear()
    monkeypatch.setenv("KCORAL_URL", "http://127.0.0.1:1")
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    with Client(
        url + "/prefix",
        execute_path="tasks/run",
        health_path="/status",
        headers={"Authorization": "Bearer test-token"},
    ) as client:
        remote = client.function()(echo)
        assert client.health()["status"] == "ok"
        assert client.target()["arch"] == "fake"
        np.testing.assert_array_equal(remote.remote(value), value)
        assert remote.remote(b"binary\x00input") == b"binary\x00input"
        # The bound decorator leaves the borrowed client open.
        assert client.health()["status"] == "ok"
    assert all(auth == "Bearer test-token" for _, _, auth in requests)
    posts = [path for method, path, _ in requests if method == "POST"]
    assert len(posts) >= 4  # Tensor and bytes both negotiate cold uploads.
    assert set(posts) == {"/prefix/tasks/run"}

    # A decorator can select a base URL and custom route directly in Python.
    remote = function(endpoint=url + "/prefix/", execute_path="/tasks/run")(echo)
    assert remote.remote("configured in code") == "configured in code"


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

    def zip(value):
        return value

    def _kcoral_builtins(value):
        return value

    with Client(server_url) as client:
        for fn in (values, dict, zip, _kcoral_builtins):
            assert client.function()(fn).remote(42) == 42


def test_execution_options_metadata_failures_and_recovery(server_url):
    with Client(server_url) as client:
        requests = []

        def capture(request):
            parts = parse_multipart(request.headers["content-type"], request.read())
            requests.append(json.loads(next(p.data for p in parts if p.name == "program")))

        client._http.event_hooks["request"] = [capture]

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
        assert result.request_id and result.elapsed_ms >= 0
        assert "stdout marker" in result.stdout and "stderr marker" in result.stderr
        assert requests[0]["options"] == {"timeout_seconds": 12, "output_limit_bytes": 1024}
        assert requests[0]["instructions"][1]["cpu_only"] is True
        with pytest.raises(RemoteExecutionError, match="negative input") as caught:
            report.remote(-1)
        failed = caught.value.result
        assert failed.status == "FAILED" and failed.request_id
        assert "ValueError" in failed.error["traceback"]
        assert "stdout marker" in failed.stdout and "stderr marker" in failed.stderr
        assert report.execute(-1).status == "FAILED"
        assert report.remote(4) == 5


def test_remote_preserves_http_errors():
    def forbidden(request):
        return httpx.Response(403, json={"error": "denied"})

    with Client("http://server") as client:
        client._http.close()
        client._http = httpx.Client(
            base_url="http://server", transport=httpx.MockTransport(forbidden)
        )
        with pytest.raises(KCoralError) as caught:
            client.function()(echo).remote(1)
        assert caught.value.status_code == 403


def test_bad_arguments_fail_before_network_access():
    remote = function(endpoint="http://127.0.0.1:1")(echo)
    with pytest.raises(TypeError, match="missing"):
        remote.remote()
    with pytest.raises(TypeError, match="unexpected"):
        remote.remote(1, unknown=2)
    for value in [(1, 2), object(), {"nested": b"bytes"}, [np.zeros(2)]]:
        with pytest.raises(TypeError, match="unsupported argument"):
            remote.remote(value)
    with pytest.raises(TypeError, match="string keys"):
        remote.remote({1: "non-string key"})
    with pytest.raises(ValueError):
        remote.remote(float("nan"))
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="circular"):
        remote.remote(cycle)


def test_rejects_external_state_including_nested_scopes():
    captured = 4

    def closure(x):
        return x + captured

    def module_dependency(x):
        return np.asarray(x)

    def nested_dependency(x):
        return [np.asarray(item) for item in x]

    for fn, match in [(closure, "capture"), (module_dependency, "np"), (nested_dependency, "np")]:
        with pytest.raises(ValueError, match=match):
            function()(fn)


def test_rejects_async_generators_wrappers_and_missing_source():
    async def coroutine():
        return 1

    def generator():
        yield 1

    for fn in (coroutine, generator, len):
        with pytest.raises(TypeError, match="synchronous Python"):
            function()(fn)

    namespace = {}
    exec("def generated(): return 1", namespace)
    with pytest.raises(ValueError, match="source"):
        function()(namespace["generated"])

    def identity(fn):
        return fn

    with pytest.raises(ValueError, match="other decorators"):

        @function()
        @identity
        def wrapped():
            return 1


@pytest.mark.parametrize(
    "path", ["", "/", "//other/execute", "https://other/run", "/run?q=1", "/run#x"]
)
def test_rejects_non_path_endpoints(path):
    with pytest.raises(ValueError, match="endpoint path"):
        Client("http://server", execute_path=path)
    with pytest.raises(ValueError, match="endpoint path"):
        Client("http://server", health_path=path)
    with pytest.raises(ValueError, match="endpoint path"):
        function(execute_path=path)(echo)


@pytest.mark.parametrize("endpoint", ["", " ", 42])
def test_explicit_invalid_endpoint_does_not_fall_back_to_environment(endpoint):
    with pytest.raises(ValueError, match="server base URL"):
        function(endpoint=endpoint)(echo)
