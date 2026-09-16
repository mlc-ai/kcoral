"""Remote decorators exercised through the real HTTP and worker protocol."""

import numpy as np
import pytest
from test_client import server_url as server_url

from kcoral import Client, RemoteExecutionError


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


def test_binary_transfers_reuse_client(server_url):
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    with Client(server_url) as client:
        remote = client.function()(echo)
        np.testing.assert_array_equal(remote.remote(value), value)
        assert remote.remote(b"binary\x00input") == b"binary\x00input"
        # The bound decorator leaves the borrowed client open.
        assert client.health()["status"] == "ok"


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
