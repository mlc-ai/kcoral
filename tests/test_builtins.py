"""Importable GPU utilities that can be checked without a GPU."""

import sys

import pytest

from kcoral.builtins import compile_tirx
from kcoral.errors import ExecutionError


def test_compile_tirx_unavailable_without_tvm(monkeypatch):
    monkeypatch.setitem(sys.modules, "tvm", None)  # makes `import tvm` raise ImportError
    with pytest.raises(ExecutionError) as exc:
        compile_tirx(object())
    assert exc.value.kind == "unavailable"
