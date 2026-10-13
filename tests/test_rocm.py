"""ROCm device handling and GPU-vendor selection, without AMD hardware."""

import os
from pathlib import Path

import pytest

from kcoral.runtime.worker import Worker, WorkerCrashed
from kcoral.support import rocm, vendor


def _node(root: Path, number: int, gpu_id: int, **properties: int) -> None:
    node = root / str(number)
    node.mkdir(parents=True)
    (node / "gpu_id").write_text(f"{gpu_id}\n")
    (node / "properties").write_text(
        "".join(f"{key} {value}\n" for key, value in properties.items())
    )


@pytest.fixture
def topology(tmp_path, monkeypatch):
    """A CPU node, then two GPUs, then one GPU this process cannot open."""
    root = tmp_path / "nodes"
    _node(root, 0, 0, cpu_cores_count=64, simd_count=0)
    gpu = {"domain": 0, "unique_id": 0x4B6AD5C27B2E4BD1, "drm_render_minor": 128}
    _node(root, 1, 1234, location_id=(0xC1 << 8) | (0 << 3), **gpu)
    _node(
        root,
        2,
        5678,
        domain=1,
        unique_id=0,
        location_id=(0x05 << 8) | (2 << 3),
        drm_render_minor=136,
    )
    _node(root, 10, 9012, domain=0, unique_id=7, location_id=0x0900, drm_render_minor=144)
    (root / "not-a-node").mkdir()
    readable = {"/dev/dri/renderD128", "/dev/dri/renderD136"}
    monkeypatch.setattr(rocm.os, "access", lambda path, mode: str(path) in readable)
    scan = rocm.agents.__wrapped__
    monkeypatch.setattr(rocm, "agents", lambda: scan(root))
    return root


def test_agents_follow_rocr_order_and_skip_cpus_and_inaccessible_gpus(topology):
    agents = rocm.agents()
    assert [agent.node for agent in agents] == [1, 2]
    assert [agent.pci for agent in agents] == ["0000:c1:00", "0001:05:02"]


def test_selector_prefers_the_rocminfo_uuid(topology):
    assert rocm.device_selector(0) == "GPU-4b6ad5c27b2e4bd1"
    assert rocm.device_selector(1) == "1"  # no unique id reported
    assert rocm.device_selector(5) == "5"


def test_identity_is_the_pci_address(topology):
    assert rocm.device_identity(1) == "0001:05:02"
    assert rocm.device_identity(9) is None
    assert rocm.identity_key(" 0000:C1:00 ") == "0000:c1:00"


def test_device_nodes_expose_only_assigned_render_nodes(topology):
    assert rocm.device_nodes((1,)) == [Path("/dev/kfd"), Path("/dev/dri/renderD136")]


def test_missing_topology_has_no_agents(tmp_path):
    assert rocm.agents.__wrapped__(tmp_path / "absent") == ()


def test_visibility_clears_hip_level_variables():
    assert rocm.visible_devices_env("GPU-1,GPU-2") == {
        "ROCR_VISIBLE_DEVICES": "GPU-1,GPU-2",
        "HIP_VISIBLE_DEVICES": None,
        "CUDA_VISIBLE_DEVICES": None,
        "GPU_DEVICE_ORDINAL": None,
    }


class _Function:
    def __init__(self, fn):
        self._fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self._fn(*args)


class _HIPLibrary:
    def __init__(self, errors):
        self.hipGetLastError = _Function(lambda: errors.pop(0))
        self.hipGetErrorName = _Function(lambda code: b"hipErrorInvalidValue")
        self.hipGetErrorString = _Function(lambda code: b"invalid argument")


def test_hip_error_api_consumes_and_formats_last_error():
    api = rocm._HIPErrorAPI(_HIPLibrary([1, 0]))

    assert api.take_last_error() == "HIP error hipErrorInvalidValue (1): invalid argument"
    assert api.take_last_error() is None


@pytest.fixture
def backend_env(monkeypatch):
    # setenv first, so the variable select() writes is restored afterwards.
    monkeypatch.setenv(vendor.ENVIRONMENT, "auto")
    monkeypatch.delenv(vendor.ENVIRONMENT)
    vendor.backend.cache_clear()
    yield
    vendor.backend.cache_clear()


def test_select_exports_the_backend_for_workers(backend_env, monkeypatch):
    monkeypatch.setattr(vendor, "detect", lambda: "rocm")
    assert vendor.select("auto") == "rocm"
    assert os.environ[vendor.ENVIRONMENT] == "rocm"
    assert vendor.backend() == "rocm"
    assert vendor.select("cuda") == "cuda"
    assert vendor.backend() == "cuda"
    with pytest.raises(ValueError):
        vendor.select("metal")


def test_detect_reads_the_torch_local_version(backend_env, monkeypatch):
    monkeypatch.setattr(vendor.importlib.metadata, "version", lambda name: "2.14.0+rocm7.0")
    assert vendor.detect() == "rocm"
    monkeypatch.setattr(vendor.importlib.metadata, "version", lambda name: "2.14.0+cu130")
    monkeypatch.setattr(vendor.Path, "exists", lambda path: False)
    assert vendor.detect() == "cuda"


def _worker(expected, reported):
    worker = object.__new__(Worker)
    worker.gpu_id = 0
    worker._expected_uuid = expected
    worker.device_uuid = reported
    worker.killed = False
    worker._kill = lambda: setattr(worker, "killed", True)
    return worker


def test_rocm_refuses_a_worker_on_a_card_it_cannot_place(monkeypatch):
    monkeypatch.setattr(vendor, "backend", lambda: "rocm")
    worker = _worker(None, "0000:c1:00")
    with pytest.raises(WorkerCrashed, match="cannot confirm"):
        worker._require_expected_device()
    assert worker.killed


def test_a_runtime_reporting_no_card_is_left_to_the_runtime(monkeypatch):
    """The real ROCm runtime refuses to start unidentified; fakes report nothing."""
    monkeypatch.setattr(vendor, "backend", lambda: "rocm")
    worker = _worker("0000:c1:00", None)
    worker._require_expected_device()
    assert not worker.killed


def test_rocm_accepts_a_worker_on_the_requested_card(monkeypatch):
    monkeypatch.setattr(vendor, "backend", lambda: "rocm")
    worker = _worker("0000:c1:00", "0000:C1:00")
    worker._require_expected_device()
    assert not worker.killed


def test_rocm_refuses_a_worker_on_another_card(monkeypatch):
    monkeypatch.setattr(vendor, "backend", lambda: "rocm")
    worker = _worker("0000:c1:00", "0000:05:00")
    with pytest.raises(WorkerCrashed, match="not the requested"):
        worker._require_expected_device()


def test_cuda_keeps_identity_best_effort(monkeypatch):
    monkeypatch.setattr(vendor, "backend", lambda: "cuda")
    worker = _worker(None, "GPU-abc")
    worker._require_expected_device()
    assert not worker.killed


def test_capabilities_describe_the_backend(monkeypatch):
    monkeypatch.setattr(vendor.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(vendor, "backend", lambda: "rocm")
    assert vendor.capabilities("gpu") == {
        "backend": "rocm",
        "timing": "trace",
        "cpu_only_check": "torch_at_entry",
        "profilers": ["rocprofv3", "rocprof-compute"],
    }
    monkeypatch.setattr(vendor, "backend", lambda: "cuda")
    assert vendor.capabilities("gpu")["profilers"] == ["ncu", "compute-sanitizer", "run-iket"]
    assert vendor.capabilities("cpu") == {"backend": "cpu"}


def test_rocm_gpu_workers_refuse_bubblewrap_when_the_probe_fails(monkeypatch):
    probed = []

    def problem(gpu, readonly):
        probed.append(gpu)
        return "HIP found 0 GPUs inside the sandbox, not 1" if gpu == 5 else None

    monkeypatch.setattr(vendor, "backend", lambda: "rocm")
    monkeypatch.setattr(rocm, "sandbox_problem", problem)
    vendor.require_gpu_sandbox_support([0, 1, 1])
    assert probed == [0, 1]
    with pytest.raises(RuntimeError, match=r"GPU 5: HIP found 0 GPUs.*--sandbox none"):
        vendor.require_gpu_sandbox_support([0, 5])


def test_cuda_never_probes_the_sandbox(monkeypatch):
    monkeypatch.setattr(vendor, "backend", lambda: "cuda")
    monkeypatch.setattr(rocm, "sandbox_problem", lambda *_: pytest.fail("probed"))
    vendor.require_gpu_sandbox_support([0])


class _Completed:
    def __init__(self, stdout, returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


@pytest.mark.parametrize(
    "stdout, expected",
    [
        ('{"count": 1, "pci": "0001:05:02"}', None),
        ('{"count": 0, "pci": null}', "HIP found 0 GPUs"),
        ('{"count": 1, "pci": "0000:c1:00"}', "HIP saw PCI 0000:c1:00"),
        ("", "the HIP probe failed"),
    ],
)
def test_sandbox_probe_requires_exactly_the_assigned_card(topology, monkeypatch, stdout, expected):
    import subprocess

    from kcoral.support import sandbox as sandboxing

    class FakeSandbox:
        def __init__(self, readonly):
            pass

        def command(self, gpu, program):
            return ["bwrap", "--", *program]

        def close(self):
            pass

    environments = []
    monkeypatch.setattr(sandboxing, "Sandbox", FakeSandbox)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kw: environments.append(kw["env"]) or _Completed(stdout),
    )
    problem = rocm.sandbox_problem(1)
    assert problem is None if expected is None else expected in problem
    assert environments[0]["ROCR_VISIBLE_DEVICES"] == "1"  # GPU 1 has no unique id
