"""Builtin behaviour that needs neither a GPU nor a kernel-language toolchain."""

import shutil
import sys
import types

import pytest

from benchmark_server.builtin_ops import _common, cuda
from benchmark_server.builtin_ops.cuda import CUDASource, compile_cuda
from benchmark_server.builtin_ops.tirx import compile_tirx
from benchmark_server.errors import ExecutionError


def test_compile_tirx_unavailable_without_tvm(monkeypatch):
    monkeypatch.setitem(sys.modules, "tvm", None)  # makes `import tvm` raise ImportError
    with pytest.raises(ExecutionError) as exc:
        compile_tirx(object())
    assert exc.value.kind == "unavailable"


def test_compile_cuda_unavailable_without_a_build_toolchain(monkeypatch, tmp_path):
    pytest.importorskip("tvm_ffi")  # else the missing piece is tvm_ffi, not the tools
    monkeypatch.setattr(shutil, "which", lambda tool: None)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))  # holds no bin/nvcc
    with pytest.raises(ExecutionError) as exc:
        compile_cuda(CUDASource(source="", entry="add"))
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
        compile_cuda(CUDASource(source="", entry="add"))
    assert exc.value.kind == "unavailable"
    assert "nvcc" not in exc.value.message  # only ninja and the host compiler are missing


@pytest.mark.parametrize(
    "args",
    [
        (object(),),  # not a cuda upload
        (CUDASource(source="", entry="add"), []),  # cfg is not a dict
        (CUDASource(source="", entry="add"), {"extra_cuda_cflags": "-O3"}),  # not a list
    ],
)
def test_compile_cuda_rejects_bad_arguments(args):
    with pytest.raises(ExecutionError) as exc:
        compile_cuda(*args)
    assert exc.value.kind == "compile"


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


def test_short_keeps_both_ends_of_a_compiler_error():
    text = "nvcc " + "-I/long/include/path " * 200 + "error: 'x' was not declared"
    short = _common.short(RuntimeError(text))
    assert short.startswith("nvcc -I/long")
    assert short.endswith("error: 'x' was not declared")


def test_builtin_module_exposes_the_registry():
    from benchmark_server import builtin
    from benchmark_server.builtin_ops import resolve

    assert builtin.check_close is resolve("builtin.check_close")
    assert builtin.compile_cuda is resolve("builtin.compile_cuda")
    assert "randn" in dir(builtin) and "benchmark" in dir(builtin)
    with pytest.raises(AttributeError) as exc:
        builtin.no_such_builtin
    assert "no_such_builtin" in str(exc.value)
