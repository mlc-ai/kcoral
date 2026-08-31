"""Settings a program can leave changed behind it, and the reset that undoes them."""

import ctypes
import os
import sys
import types

import pytest

from kcoral import gpu_runtime, process_state

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def settings_restored():
    """Undo what these tests change even when one aborts mid-way. Written out by
    hand rather than through :func:`process_state.restore`, which is under test."""
    environ = dict(os.environ)
    fp32 = torch.backends.cuda.matmul.fp32_precision
    precision = torch.get_float32_matmul_precision()
    grad = torch.is_grad_enabled()
    yield
    torch.backends.cuda.matmul.fp32_precision = fp32  # before the getter below is read
    torch.set_float32_matmul_precision(precision)
    torch.set_grad_enabled(grad)
    for name in set(os.environ) - set(environ):
        del os.environ[name]
    for name in ("KCORAL_LEAKED", "KCORAL_CPP_LEAKED"):
        os.unsetenv(name)  # these tests can set one where only libc can see it
        os.environ.pop(name, None)
    os.environ.update(environ)  # each write putenv()s, resyncing libc as well


@pytest.fixture
def runtime(monkeypatch):
    """A GPURuntime with the GPU-touching warm-up stubbed out."""
    monkeypatch.setitem(sys.modules, "tvm_ffi", types.SimpleNamespace())
    monkeypatch.setattr(gpu_runtime, "_warm_up", lambda: None)
    return gpu_runtime.GPURuntime()


POLLUTING = """
import os, torch

torch.backends.cuda.matmul.allow_tf32 = not torch.backends.cuda.matmul.allow_tf32
torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)
os.environ["KCORAL_LEAKED"] = "1"
os.environ["PATH"] = "/nowhere"


def main():
    pass
"""


def test_reset_restores_settings_an_upload_changed(runtime):
    before = process_state.snapshot()

    runtime.load_module(POLLUTING)
    assert process_state.snapshot() != before  # the upload took
    runtime.reset()

    assert process_state.snapshot()["torch"] == before["torch"]
    assert "KCORAL_LEAKED" not in os.environ
    assert os.environ["PATH"] == before["environ"]["PATH"]


def test_restore_undoes_a_setenv_made_behind_python(runtime):
    """A compiled library changes libc's environment directly, which leaves
    ``os.environ`` stale - so the restore has to diff against libc, not against it."""
    libc = ctypes.CDLL(None)
    before = os.environ["PATH"]

    libc.setenv(b"KCORAL_CPP_LEAKED", b"1", 1)  # what an uploaded .so would call
    libc.setenv(b"PATH", b"/nowhere-cpp", 1)
    assert "KCORAL_CPP_LEAKED" not in os.environ  # Python never saw either change
    assert os.environ["PATH"] == before

    runtime.reset()

    assert process_state._environ().get("KCORAL_CPP_LEAKED") is None
    assert process_state._environ()["PATH"] == before


def test_snapshot_survives_a_torch_without_the_knobs(monkeypatch):
    """Which knobs exist varies by torch build, so a missing one is skipped."""
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    snapshot = process_state.snapshot()

    assert snapshot["torch"] == {}
    process_state.restore(snapshot)
