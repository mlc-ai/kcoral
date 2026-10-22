"""Allocation-relative CUDA ordinals must resolve to physical GPU identities."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from kcoral.support import cuda


@pytest.fixture
def probe(monkeypatch):
    calls = []

    def install(identities, physical=None):
        physical = physical if physical is not None else {"GPU-first": 5, "GPU-second": 2}

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(stdout=json.dumps(identities))

        monkeypatch.setattr(cuda.subprocess, "run", run)
        monkeypatch.setattr(
            cuda,
            "_nvml_api",
            lambda: SimpleNamespace(
                device_index=physical.get,
                device_uuid={index: identity for identity, index in physical.items()}.get,
            ),
        )

    return install, calls


def test_resolves_cuda_visible_identities_in_order_without_using_ordinals(probe, monkeypatch):
    install, calls = probe
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    install(["FIRST", "GPU-second"])
    assert cuda.visible_device_ids() == [5, 2]
    command, options = calls[0]
    assert command[0] == cuda.sys.executable
    assert command[1] == "-c"
    assert "torch.cuda.init()" in command[2]
    assert options["timeout"] == 60
    assert options["check"] is True
    assert "env" not in options


@pytest.mark.parametrize(
    ("identities", "message"),
    [
        ([], "no CUDA devices"),
        (["missing"], "cannot map"),
        (["first", "first"], "duplicate"),
        ([None], "invalid GPU UUID"),
        ([""], "invalid GPU UUID"),
        ({"gpu": "first"}, "no CUDA devices"),
    ],
)
def test_fails_closed_for_missing_or_ambiguous_devices(probe, identities, message):
    install, _ = probe
    install(identities)
    with pytest.raises(RuntimeError, match=message):
        cuda.visible_device_ids()


def test_requires_nvml(probe, monkeypatch):
    install, _ = probe
    install(["first"])
    monkeypatch.setattr(cuda, "_nvml_api", lambda: None)
    with pytest.raises(RuntimeError, match="NVML is required"):
        cuda.visible_device_ids()


def test_rejects_nvml_identity_mismatch(probe, monkeypatch):
    install, _ = probe
    install(["first"])
    monkeypatch.setattr(
        cuda,
        "_nvml_api",
        lambda: SimpleNamespace(
            device_index=lambda identity: 0, device_uuid=lambda index: "GPU-other"
        ),
    )
    with pytest.raises(RuntimeError, match="cannot map"):
        cuda.visible_device_ids()


@pytest.mark.parametrize(
    "error",
    [
        OSError("no python"),
        subprocess.TimeoutExpired("probe", 60),
        subprocess.CalledProcessError(1, "probe"),
    ],
)
def test_probe_failure_never_falls_back_to_gpu_zero(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(cuda.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="cannot probe"):
        cuda.visible_device_ids()


def test_nvml_uuid_lookup_uses_the_returned_handle():
    def handle(identity, result):
        assert identity == b"GPU-first"
        result._obj.value = 123
        return 0

    def index(handle, result):
        assert handle.value == 123
        result._obj.value = 5
        return 0

    api = cuda._NVMLAPI(
        SimpleNamespace(
            nvmlInit_v2=lambda: 0,
            nvmlDeviceGetHandleByIndex_v2=None,
            nvmlDeviceGetUUID=None,
            nvmlDeviceGetMinorNumber=None,
            nvmlDeviceGetHandleByUUID=handle,
            nvmlDeviceGetIndex=index,
        )
    )
    assert api.device_index("GPU-first") == 5
    api._handle_by_uuid = lambda *args: 1
    assert api.device_index("GPU-first") is None
