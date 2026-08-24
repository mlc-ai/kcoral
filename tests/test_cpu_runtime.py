import sys

import pytest

from benchmark_server.builtin_ops.cuda import CUDASource, compile_cuda_binary
from benchmark_server.cpu_runtime import CPURuntime
from benchmark_server.errors import ExecutionError


def test_cpu_runtime_binds_cuda_source_without_importing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    runtime = CPURuntime()

    source = runtime.load_module("void add() {}", entry="add", language="cuda")

    assert source == CUDASource(source="void add() {}", entry="add")
    assert runtime.builtin("builtin.compile_cuda_binary") is compile_cuda_binary
    assert runtime.cpu_only_builtins() == frozenset({"builtin.compile_cuda_binary"})
    assert runtime.target() == {}
    assert runtime.device_uuid() is None
    assert runtime.take_last_error() is None
    runtime.synchronize()


@pytest.mark.parametrize(
    "operation",
    [
        lambda runtime: runtime.load_module("def main(): pass"),
        lambda runtime: runtime.load_library(b"library", "entry"),
        lambda runtime: runtime.load_tensor(b"", "float32", [0]),
        lambda runtime: runtime.builtin("builtin.empty"),
    ],
)
def test_cpu_runtime_rejects_gpu_operations(operation):
    with pytest.raises(ExecutionError) as exc:
        operation(CPURuntime())
    assert exc.value.kind == "unavailable"
