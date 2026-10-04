"""CUPTI cleanup without requiring a GPU."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from kcoral.runtime import gpu as gpu_runtime


@pytest.mark.parametrize("guard_used", [False, True])
@pytest.mark.parametrize("device_count", [1, 2])
def test_reset_finalizes_guard_after_cuda_sync(monkeypatch, guard_used, device_count):
    events = []

    @contextmanager
    def device(index):
        events.append(("device", index))
        yield

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                device_count=lambda: device_count,
                device=device,
                synchronize=lambda index: events.append(("sync", index)),
                empty_cache=lambda: events.append("empty_cache"),
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "cupti",
        SimpleNamespace(
            cupti=SimpleNamespace(
                finalize=lambda: events.append("finalize"),
            )
        ),
    )
    monkeypatch.setattr(gpu_runtime.process_state, "restore", lambda state: None)
    monkeypatch.setattr(gpu_runtime.sandbox, "active", lambda: False)
    runtime = object.__new__(gpu_runtime.GPURuntime)
    runtime._process_state = None
    runtime._seeded_fnames = []
    runtime._request_libraries = []
    runtime._cupti_guard_used = guard_used
    runtime.reset()
    assert events == (
        [("sync", index) for index in range(device_count)]
        + [event for index in range(device_count) for event in (("device", index), "empty_cache")]
        + (["finalize"] if guard_used else [])
    )
    assert not runtime._cupti_guard_used


def test_release_clears_unused_cache_on_every_device(monkeypatch):
    events = []
    current = 0

    @contextmanager
    def device(index):
        nonlocal current
        previous, current = current, index
        try:
            yield
        finally:
            current = previous

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                device_count=lambda: 2,
                device=device,
                synchronize=lambda index: events.append(("sync", index)),
                memory_reserved=lambda: 8 if current == 1 else 4,
                memory_allocated=lambda: 4,
                empty_cache=lambda: events.append(("empty_cache", current)),
            )
        ),
    )
    runtime = object.__new__(gpu_runtime.GPURuntime)
    runtime.prepare_to_release_gpu()
    assert events == [("sync", 0), ("sync", 1), ("empty_cache", 1), ("sync", 1)]
