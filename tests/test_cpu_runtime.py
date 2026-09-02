import sys

import pytest

from kcoral.builtin_ops.cuda import CUDAModule, compile_cuda_binary
from kcoral.cpu_runtime import CPURuntime
from kcoral.errors import ExecutionError


def test_cpu_runtime_binds_cuda_source_without_importing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    runtime = CPURuntime()

    module = runtime.load_module("void add() {}", language="cuda")
    source = runtime.get_function(module, "add")

    assert module == CUDAModule(source="void add() {}")
    assert source == CUDAModule(source="void add() {}", name="add")
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
        lambda runtime: runtime.load_library(b"library"),
        lambda runtime: runtime.load_tensor(b"", "float32", [0]),
        lambda runtime: runtime.builtin("builtin.empty"),
    ],
)
def test_cpu_runtime_rejects_gpu_operations(operation):
    with pytest.raises(ExecutionError) as exc:
        operation(CPURuntime())
    assert exc.value.kind == "unavailable"


@pytest.mark.parametrize("name", ["main", "not::an::identifier"])
def test_cpu_runtime_rejects_invalid_cuda_function_names(name):
    module = CPURuntime().load_module("void add() {}", language="cuda")
    with pytest.raises(ExecutionError) as exc:
        CPURuntime().get_function(module, name)
    assert exc.value.kind == "parse"
