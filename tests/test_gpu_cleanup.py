"""CUPTI cleanup without requiring a GPU."""
import sys
from types import SimpleNamespace

import pytest

from kcoral import gpu_runtime


@pytest.mark.parametrize('guard_used', [False, True])
def test_reset_finalizes_guard_after_cuda_sync(monkeypatch, guard_used):
    events = []
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: events.append('sync'),
        empty_cache=lambda: events.append('empty_cache'),
    )))
    monkeypatch.setitem(sys.modules, 'cupti', SimpleNamespace(cupti=SimpleNamespace(
        finalize=lambda: events.append('finalize'),
    )))
    monkeypatch.setattr(gpu_runtime.process_state, 'restore', lambda state: None)
    monkeypatch.setattr(gpu_runtime.sandbox, 'active', lambda: False)
    runtime = object.__new__(gpu_runtime.GPURuntime)
    runtime._process_state = None
    runtime._seeded_fnames = []
    runtime._request_libraries = []
    runtime._cupti_guard_used = guard_used
    runtime.reset()
    assert events == ['sync', 'empty_cache'] + (['finalize'] if guard_used else [])
    assert not runtime._cupti_guard_used
