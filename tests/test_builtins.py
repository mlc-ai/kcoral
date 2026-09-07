"""Builtin behaviour that needs neither a GPU nor a kernel-language toolchain."""

import os
import shutil
import sys
import types

import pytest

from kcoral import gpu_runtime
from kcoral.builtin_ops import _common, _registry, cuda
from kcoral.builtin_ops.cuda import CUDAModule, compile_cuda, compile_cuda_binary
from kcoral.builtin_ops.tirx import compile_tirx
from kcoral.deferred import DeferredGPUResult
from kcoral.errors import ExecutionError
from kcoral.gpu_runtime import LoadedFunction, LoadedLibrary


def test_compile_tirx_unavailable_without_tvm(monkeypatch):
    monkeypatch.setitem(sys.modules, "tvm", None)  # makes `import tvm` raise ImportError
    with pytest.raises(ExecutionError) as exc:
        compile_tirx(object())
    assert exc.value.kind == "unavailable"


def test_loaded_function_keeps_its_module():
    owner = LoadedLibrary(module=object())
    function = LoadedFunction(owner=owner, function=lambda value: value + 1)

    assert function(41) == 42
    assert function.owner is owner


def test_compile_cuda_unavailable_without_a_build_toolchain(monkeypatch, tmp_path):
    pytest.importorskip("tvm_ffi")  # else the missing piece is tvm_ffi, not the tools
    monkeypatch.setattr(shutil, "which", lambda tool: None)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))  # holds no bin/nvcc
    with pytest.raises(ExecutionError) as exc:
        compile_cuda(CUDAModule(source="", name="add"))
    assert exc.value.kind == "unavailable"
    assert "nvcc" in exc.value.message and "ninja" in exc.value.message


def test_compile_cuda_finds_nvcc_under_cuda_home(monkeypatch, tmp_path):
    # nvcc off PATH but under CUDA_HOME is a working install, not an unavailable
    # one — tvm_ffi resolves it the same way.
    pytest.importorskip("tvm_ffi")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "nvcc").touch()
    monkeypatch.setattr(shutil, "which", lambda tool: None)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    with pytest.raises(ExecutionError) as exc:
        compile_cuda(CUDAModule(source="", name="add"))
    assert exc.value.kind == "unavailable"
    assert "nvcc" not in exc.value.message  # only ninja and the host compiler are missing


@pytest.mark.parametrize(
    "args",
    [
        (object(),),  # not a cuda upload
        (CUDAModule(source=""),),  # no function has been selected
        (CUDAModule(source="", name="add"), []),  # cfg is not a dict
        (CUDAModule(source="", name="add"), {"extra_cuda_cflags": "-O3"}),  # not a list
    ],
)
def test_compile_cuda_rejects_bad_arguments(args):
    with pytest.raises(ExecutionError) as exc:
        compile_cuda(*args)
    assert exc.value.kind == "compile"


def test_compile_cuda_builds_off_lease_and_defers_module_loading(monkeypatch):
    tvm_ffi = pytest.importorskip("tvm_ffi")
    tvm_ffi_cpp = pytest.importorskip("tvm_ffi.cpp")

    calls = []
    monkeypatch.setattr(cuda, "_require_cuda_toolchain", lambda: None)
    monkeypatch.setattr(cuda, "_cuda_arch_list", lambda: "10.0a")
    monkeypatch.setattr(
        tvm_ffi_cpp,
        "build_inline",
        lambda **kwargs: calls.append(("build", kwargs)) or "/cache/upload_run.so",
    )

    def compiled_fn(value):
        return value + 1

    monkeypatch.setattr(
        tvm_ffi,
        "load_module",
        lambda path: calls.append(("load", path)) or types.SimpleNamespace(run=compiled_fn),
    )

    deferred = compile_cuda(CUDAModule(source="void run() {}", name="run"))

    assert isinstance(deferred, DeferredGPUResult)
    assert [call[0] for call in calls] == ["build"]
    assert calls[0][1]["backend"] == "cuda"
    resolved = deferred.resolve()
    assert [call[0] for call in calls] == ["build", "load"]
    assert resolved(41) == 42


@pytest.mark.parametrize(
    "args",
    [
        (object(), {"arch": "sm_90a"}),
        (CUDAModule(source=""), {"arch": "sm_90a"}),
        (CUDAModule(source="", name="add"), []),
        (CUDAModule(source="", name="add"), {}),
        (CUDAModule(source="", name="add"), {"arch": "90a"}),
        (
            CUDAModule(source="", name="add"),
            {"arch": "sm_90a", "extra_cuda_cflags": "-O3"},
        ),
    ],
)
def test_compile_cuda_binary_rejects_bad_arguments(args):
    with pytest.raises(ExecutionError) as exc:
        compile_cuda_binary(*args)
    assert exc.value.kind == "compile"


def test_compile_cuda_binary_returns_built_library(monkeypatch, tmp_path):
    library = tmp_path / "add.so"
    library.write_bytes(b"compiled-library")
    calls = []

    def build(src, flags, arch_list=None):
        calls.append((src, flags, arch_list))
        return str(library)

    monkeypatch.setattr(cuda, "_build_cuda", build)
    source = CUDAModule(source="void add() {}", name="add")

    result = compile_cuda_binary(source, {"arch": "sm_100a", "extra_cuda_cflags": ["-O3"]})

    assert result == b"compiled-library"
    assert calls == [(source, ["-O3"], "10.0a")]


@pytest.mark.parametrize(
    "arch,expected",
    [("sm_89", "8.9"), ("sm_90a", "9.0a"), ("sm_100a", "10.0a")],
)
def test_tvm_ffi_arch_conversion(arch, expected):
    assert cuda._tvm_ffi_arch(arch) == expected


def test_cuda_arch_override_is_temporary(monkeypatch):
    monkeypatch.setenv("TVM_FFI_CUDA_ARCH_LIST", "8.9")
    with cuda._cuda_arch_override("10.0a"):
        assert os.environ["TVM_FFI_CUDA_ARCH_LIST"] == "10.0a"
    assert os.environ["TVM_FFI_CUDA_ARCH_LIST"] == "8.9"


@pytest.mark.parametrize(
    "capability,expected", [((8, 9), "8.9"), ((9, 0), "9.0a"), ((10, 0), "10.0a")]
)
def test_cuda_arch_list_suffixes_hopper_and_newer(monkeypatch, capability, expected):
    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(get_device_capability=lambda: capability)
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    cuda._cuda_arch_list.cache_clear()
    try:
        assert cuda._cuda_arch_list() == expected
    finally:
        cuda._cuda_arch_list.cache_clear()


def test_diagnostics_drops_the_ninja_invocation():
    raw = (
        "ninja exited with status 2\nstdout:\n"
        "[1/2] /usr/local/cuda/bin/nvcc " + "-I/a/very/long/include " * 20 + "-c cuda.cu\n"
        "FAILED: [code=2] cuda_0.o\n"
        'cuda.cu(9): error: identifier "alpha" is undefined\n'
    )
    assert cuda._diagnostics(raw).startswith("cuda.cu(9): error:")
    # ptxas spells it differently, and output with no diagnostic is left alone
    assert cuda._diagnostics("a\nptxas x.ptx, line 3; error   : bad\n").startswith("ptxas")
    assert cuda._diagnostics("no diagnostic here") == "no diagnostic here"


def test_cuda_source_can_manage_its_own_ffi_export():
    source = """
void run() {}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""
    assert cuda._declares_tvm_ffi_macro(source)
    assert not cuda._declares_tvm_ffi_macro("// TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);")


def test_short_keeps_both_ends_of_a_compiler_error():
    text = "nvcc " + "-I/long/include/path " * 200 + "error: 'x' was not declared"
    short = _common.short(RuntimeError(text))
    assert short.startswith("nvcc -I/long")
    assert short.endswith("error: 'x' was not declared")


def test_builtin_module_exposes_the_registry():
    from kcoral import builtin
    from kcoral.builtin_ops import resolve

    assert builtin.check_close is resolve("builtin.check_close")
    assert builtin.compile_cuda is resolve("builtin.compile_cuda")
    assert builtin.compile_cuda_binary is resolve("builtin.compile_cuda_binary")
    assert "randn" in dir(builtin) and "benchmark" in dir(builtin)
    with pytest.raises(AttributeError) as exc:
        builtin.no_such_builtin
    assert "no_such_builtin" in str(exc.value)


TAMPERING = """
from kcoral.builtin_ops import _registry

_registry._REGISTRY["builtin.check_close"] = lambda *args: {"passed": True}
_registry._CPU_ONLY.add("builtin.check_close")


def main():
    pass
"""


def test_reset_restores_builtins_an_upload_rewired(monkeypatch):
    # The registry is process-global: swap in copies first, so an assertion that
    # aborts before the reset cannot leave the tampering for later tests to hit.
    monkeypatch.setattr(_registry, "_REGISTRY", dict(_registry._REGISTRY))
    monkeypatch.setattr(_registry, "_CPU_ONLY", set(_registry._CPU_ONLY))
    cuda = types.SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "tvm_ffi", types.SimpleNamespace())
    monkeypatch.setattr(gpu_runtime, "_warm_up", lambda: None)  # a warm-up needs a GPU
    runtime = gpu_runtime.GPURuntime()
    real = runtime.builtin("builtin.check_close")

    runtime.load_module(TAMPERING)
    assert runtime.builtin("builtin.check_close") is not real  # both edits took
    assert "builtin.check_close" in runtime.cpu_only_builtins()
    runtime.reset()

    assert runtime.builtin("builtin.check_close") is real
    assert "builtin.check_close" not in runtime.cpu_only_builtins()
