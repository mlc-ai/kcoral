"""Opt-in GPU tests. The caller must reserve the physical devices first."""

import json
import os
import runpy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kcoral import Client
from kcoral.config import ServerConfig
from kcoral.server.app import create_app
from kcoral.support import vendor

pytestmark = pytest.mark.skipif(
    os.environ.get("KCORAL_MULTI_GPU_TEST") != "1",
    reason="requires reserved GPUs and KCORAL_MULTI_GPU_TEST=1",
)


@pytest.fixture(scope="module", params=["none", "bubblewrap"])
def gpu_client(request):
    devices = [int(value) for value in os.environ["KCORAL_TEST_GPUS"].split(",")]
    if request.param == "bubblewrap" and vendor.backend() == "rocm":
        pytest.skip("bubblewrap does not yet support ROCm GPU workers")
    with TestClient(
        create_app(
            ServerConfig(
                gpus=devices,
                sandbox=request.param,
                workers_per_gpu=1,
                max_requests_per_worker=0,
                worker_termination_grace_seconds=1,
                log_console=False,
            )
        )
    ) as server:
        if request.param == "bubblewrap" and server.app.state.sandbox != "bubblewrap":
            pytest.skip("bubblewrap is unavailable on this GPU server")
        client = Client("http://testserver")
        client._http.close()
        client._http = server
        yield client, len(devices)


@pytest.mark.parametrize("multiprocess", [False, True])
def test_execute_library_multi_gpu_kernel(gpu_client, multiprocess):
    client, capacity = gpu_client
    count = 2
    if count > capacity:
        pytest.skip("not enough reserved GPUs")
    name = "multiprocess" if multiprocess else "single_process"
    example = Path(__file__).parents[1] / "examples" / "multi_gpu" / f"{name}.py"
    program = runpy.run_path(str(example))["build_program"]()
    result = client.execute(program, gpu_count=count, timeout_seconds=120)
    assert result.completed, (result.error, result.stdout, result.stderr)
    assert len(result.gpu_ids) == count
    report = json.loads(result["report"].read_bytes())
    assert report["ok"] and report["kernel"] == "nccl_all_reduce"
    assert report["gpu_count"] == count
    assert report["gpu_processes"] == (count if multiprocess else 1)
    assert len(set(report["pids"])) == report["gpu_processes"]
    if client._http.app.state.sandbox == "none":
        assert all(not Path(f"/proc/{pid}").exists() for pid in report["pids"])
    if multiprocess:
        assert [(rank["rank"], rank["device"]) for rank in report["ranks"]] == [
            (i, i) for i in range(count)
        ]
    assert report["checked_elements"] == count * report["elements"] * report["iterations"]


IDENTITY_PROGRAM = """
import torch
from kcoral.support import vendor

def main():
    return {"count": torch.cuda.device_count(), "identity": vendor.describe_device_identity()}
"""


def test_each_worker_sees_exactly_its_physical_gpu():
    """A worker pinned to the last reserved GPU must see one device, and it must be
    that physical card (by UUID on CUDA, PCI address on ROCm), not merely be given
    the expected device files."""
    from kcoral import Program
    from kcoral.client.result import _decode_value
    from kcoral.protocol import parse_program
    from kcoral.runtime.gpu import gpu_runtime_factory
    from kcoral.runtime.lease import NoopLeases
    from kcoral.runtime.worker import Worker

    gpu = [int(value) for value in os.environ["KCORAL_TEST_GPUS"].split(",")][-1]
    program = Program()
    module = program.upload(kind="module", source=IDENTITY_PROGRAM)
    program.return_(
        key="seen", value=program.run(fn=program.get_function(module=module, name="main"))
    )
    parsed = parse_program({"instructions": program.instructions})
    parsed.blob_bytes = program._blobs
    worker = Worker(gpu, gpu_runtime_factory, sandbox="none", max_requests=0)
    try:
        result = worker.run(parsed, 120, NoopLeases())
    finally:
        worker._kill()
    assert result.execution.status == "COMPLETED", result.execution.error
    outcome = result.execution
    seen = _decode_value({"type": "object", "value": outcome.results}, outcome.binary_parts, set())
    assert seen["seen"]["count"] == 1
    reported = vendor.identity_key(seen["seen"]["identity"])
    assert reported == vendor.identity_key(vendor.device_identity(gpu))
