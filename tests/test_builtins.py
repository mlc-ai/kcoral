"""Importable GPU utilities that can be checked without a GPU."""

import sys

import pytest

from kcoral.builtins import _iteration_counts, compile_tirx
from kcoral.errors import ExecutionError


def test_compile_tirx_unavailable_without_tvm(monkeypatch):
    monkeypatch.setitem(sys.modules, "tvm", None)  # makes `import tvm` raise ImportError
    with pytest.raises(ExecutionError) as exc:
        compile_tirx(object())
    assert exc.value.kind == "unavailable"


@pytest.mark.parametrize("warmup", [0, 3])
def test_explicit_iteration_counts_preserve_zero_warmup(monkeypatch, warmup):
    monkeypatch.setitem(sys.modules, "torch", object())
    assert _iteration_counts(lambda: None, {"warmup": warmup, "repeat": 2}, None) == (warmup, 2)
