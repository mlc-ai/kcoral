"""Opt-in GPU tests. The caller must reserve the physical devices first."""

import json
import os
import runpy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kcoral import Client, Program
from kcoral.app import create_app
from kcoral.client import KCoralError
from kcoral.config import ServerConfig

pytestmark = pytest.mark.skipif(
    os.environ.get("KCORAL_MULTI_GPU_TEST") != "1",
    reason="requires reserved GPUs and KCORAL_MULTI_GPU_TEST=1",
)


@pytest.fixture(scope="module")
def gpu_client():
    devices = [int(value) for value in os.environ["KCORAL_TEST_GPUS"].split(",")]
    with TestClient(
        create_app(
            ServerConfig(
                gpus=devices,
                workers_per_gpu=1,
                max_requests_per_worker=0,
                worker_termination_grace_seconds=1,
                log_console=False,
            )
        )
    ) as server:
        client = Client("http://testserver")
        client._http.close()
        client._http = server
        yield client, len(devices)


@pytest.mark.parametrize("count", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("torchrun", [False, True])
def test_complete_inference_script(gpu_client, count, torchrun):
    client, capacity = gpu_client
    if count > capacity:
        pytest.skip("not enough reserved GPUs")
    example = Path(__file__).parents[1] / "examples/multi_gpu_inference/client.py"
    program = runpy.run_path(str(example))["build_program"](count, torchrun=torchrun)
    result = client.execute(program, gpu_count=count, timeout_seconds=120)
    assert result.completed, (result.error, result.stdout, result.stderr)
    assert len(result.gpu_ids) == count
    report = json.loads(result["report"].read_bytes())
    assert len(report["ranks"]) == (count if torchrun else 1)
    assert all(rank["ok"] and rank["gpu_count"] == count for rank in report["ranks"])


def test_interpreter_local_objects_survive_multiple_runs(gpu_client):
    client, capacity = gpu_client
    count = min(capacity, 3)
    p = Program()
    module = p.upload(
        id="module",
        kind="module",
        source="""
def allocate():
    import torch
    return [torch.ones(16, device=f"cuda:{i}") for i in range(torch.cuda.device_count())]

def update(values):
    for i, value in enumerate(values):
        value.add_(i)
    return values
""",
    )
    allocate = p.get_function(id="allocate", module=module, name="allocate")
    update = p.get_function(id="update", module=module, name="update")
    values = p.run(id="values", fn=allocate)
    updated = p.run(id="updated", fn=update, args=[values])
    p.return_(key="values", value=updated)
    result = client.execute(p, gpu_count=count, timeout_seconds=30)
    assert result.completed, result.error
    for i, value in enumerate(result["values"]):
        assert (value == 1 + i).all()


def test_torchrun_timeout_reclaims_detached_gpu_ranks(gpu_client, tmp_path):
    client, capacity = gpu_client
    if capacity < 2:
        pytest.skip("needs two GPUs")
    script = f"""
import os
import signal
import time
from pathlib import Path
import torch
import torch.distributed as dist
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
value = torch.ones(1024, device="cuda")
dist.all_reduce(value)
torch.cuda.synchronize()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path({str(tmp_path)!r}, f"rank-{{rank}}.pid").write_text(str(os.getpid()))
while True:
    time.sleep(1)
"""
    p = Program()
    p.upload_file(path="hang.py", blob=script.encode())
    module = p.upload(
        id="launcher",
        kind="module",
        source="""
def main():
    import subprocess, sys
    subprocess.run([
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nnodes=1", "--nproc-per-node=2", "--max-restarts=0", "hang.py",
    ], check=True)
""",
    )
    fn = p.get_function(id="main", module=module, name="main")
    p.run(id="hang", fn=fn)
    with pytest.raises(KCoralError) as error:
        client.execute(p, gpu_count=2, timeout_seconds=20)
    assert error.value.status_code == 504
    for rank in range(2):
        pid = (tmp_path / f"rank-{rank}.pid").read_text()
        assert not Path(f"/proc/{pid}").exists()
    example = Path(__file__).parents[1] / "examples/multi_gpu_inference/client.py"
    program = runpy.run_path(str(example))["build_program"](2, steps=1)
    result = client.execute(program, gpu_count=2, timeout_seconds=90)
    assert result.completed, (result.error, result.stderr)
