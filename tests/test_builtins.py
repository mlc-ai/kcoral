"""Builtin behaviour that needs neither a GPU nor tvm."""

import sys

import pytest

from benchmark_server.builtin_ops import compile_tirx
from benchmark_server.errors import ExecutionError


def test_compile_tirx_unavailable_without_tvm(monkeypatch):
    # tvm is an optional server dependency; simulate its absence.
    monkeypatch.setitem(sys.modules, "tvm", None)  # makes `import tvm` raise ImportError
    with pytest.raises(ExecutionError) as exc:
        compile_tirx(object())
    assert exc.value.kind == "unavailable"
