import sys

import pytest

from kcoral.cpu_runtime import CPURuntime
from kcoral.cuda_source import CUDAModule
from kcoral.errors import ExecutionError


def test_cpu_runtime_binds_cuda_source_without_importing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    runtime = CPURuntime()

    module = runtime.load_module("void add() {}", language="cuda")
    source = runtime.get_function(module, "add")

    assert module == CUDAModule(source="void add() {}")
    assert source == CUDAModule(source="void add() {}", name="add")
    assert runtime.target() == {}
    assert runtime.device_uuid() is None
    assert runtime.take_last_error() is None
    runtime.synchronize()


@pytest.mark.parametrize(
    "operation",
    [
        lambda runtime: runtime.load_library(b"library"),
        lambda runtime: runtime.load_tensor(b"", "float32", [0]),
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


def test_cpu_runtime_restores_environment_and_releases_module_cycles(monkeypatch):
    import os
    import weakref

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("KCORAL_TEST_CPU_STATE", "original")
    runtime = CPURuntime()
    module = runtime.load_module(
        "import os\nos.environ['KCORAL_TEST_CPU_STATE'] = 'changed'\n"
        "class Value: pass\nvalue = Value()\ndef main(): return value\n"
    )
    value = weakref.ref(module.namespace["value"])
    del module
    runtime.reset()
    assert value() is None
    assert os.environ["KCORAL_TEST_CPU_STATE"] == "original"
