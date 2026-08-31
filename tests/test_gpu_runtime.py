"""GPU integration tests for the runtime and the TIRx and CUDA C builtins."""

import importlib.util
import os
import pathlib

import pytest

from kcoral.engine import execute
from kcoral.keys import compute_blob_hash
from kcoral.schemas import Program, Ref, Return, Run, Upload
from kcoral.testing import UNSHARED_GPU

pytestmark = pytest.mark.skipif(
    os.environ.get("KCORAL_GPU_TEST") != "1",
    reason="GPU integration test requires KCORAL_GPU_TEST=1",
)

KERNEL = """from __future__ import annotations
from tvm.script import tirx as T

@T.jit
def main(A: T.Buffer((N,), "float32"), B: T.Buffer((N,), "float32"), *, N: T.constexpr):
    T.device_entry()
    i = T.cta_id([N])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0
"""
REF = "def main(a):\n    return a + 1.0\n"

PRIM_KERNEL = """from tvm.script import tirx as T

@T.prim_func
def main(A: T.Buffer((256,), "float32"), B: T.Buffer((256,), "float32")):
    T.device_entry()
    i = T.cta_id([256])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0
"""

CUDA_KERNEL = """
__global__ void add_one_kernel(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + 1.0f;
}

void add_one(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  int n = static_cast<int>(x.numel());
  add_one_kernel<<<(n + 255) / 256, 256>>>(static_cast<const float*>(x.data_ptr()),
                                           static_cast<float*>(y.data_ptr()), n);
}
"""

# One warp round-trips a value per lane through tensor memory. tcgen05 is gated
# behind the sm_100a target. The tcgen05 ops are warp-synchronous, so no
# __syncthreads is needed, but the allocation must be freed or the launch reports
# cudaErrorTensorMemoryLeak.
TMEM_KERNEL = """
__global__ void tmem_kernel(int* out) {
  __shared__ unsigned taddr_smem;
  unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(&taddr_smem));
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :: "r"(smem), "r"(32));
  unsigned taddr = taddr_smem;
  unsigned value = 100u + threadIdx.x;
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" :: "r"(taddr), "r"(value));
  asm volatile("tcgen05.wait::st.sync.aligned;");
  unsigned got;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(got) : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
  out[threadIdx.x] = static_cast<int>(got);
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(32));
}

void tmem_roundtrip(tvm::ffi::TensorView out) {
  tmem_kernel<<<1, 32>>>(static_cast<int*>(out.data_ptr()));
}
"""


def build_library(source, entry, tmp_path):
    """Compile CUDA C the way a client would, off the server, and return the bytes."""
    import tvm_ffi.cpp
    from tvm_ffi.cpp import extension

    from kcoral.builtin_ops.cuda import _cuda_arch_list

    os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", _cuda_arch_list())
    cu = tmp_path / f"{entry}.cu"
    cu.write_text(extension._decorate_with_tvm_ffi(source, {entry: ""}))
    return pathlib.Path(tvm_ffi.cpp.build(name=entry, cuda_files=[str(cu)])).read_bytes()


def build_tirx_library(tmp_path):
    """Compile TIRx the way a client would, with an explicit target so no GPU is
    consulted. The exported name comes from the function, not the IRModule key."""
    import tvm

    from kcoral.gpu_runtime import GPURuntime, describe_target

    prim_func = GPURuntime().load_module(PRIM_KERNEL.replace("def main(", "def add_one("))
    target = tvm.target.Target({"kind": "cuda", "arch": describe_target()["arch"]})
    with target:  # the tirx pipeline reads the arch from Target.current()
        executable = tvm.compile(
            tvm.IRModule({"add_one": prim_func}), target=target, tir_pipeline="tirx"
        )
    path = tmp_path / "tirx_add_one.so"
    executable.export_library(str(path))
    return path.read_bytes()


CUTEDSL_KERNEL = """import cutlass.cute as cute

@cute.kernel
def add_one_kernel(src: cute.Tensor, dst: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 256 + tidx
    if i < cute.size(src):
        dst[i] = src[i] + 1.0

@cute.jit
def add_one(src: cute.Tensor, dst: cute.Tensor):
    n = cute.size(src)
    add_one_kernel(src, dst).launch(grid=((n + 255) // 256, 1, 1), block=(256, 1, 1))
"""


CUTEDSL_BUILD = """import pathlib, subprocess, sys
import torch, tvm_ffi.libinfo
from cutlass.cute import compile as cute_compile
from cutlass.cute.export.aot_config import get_libdir
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cute_add_one import add_one

operand = from_dlpack(torch.zeros(256, dtype=torch.float32, device="cuda"))
compiled = cute_compile(add_one, operand, operand, options="--enable-tvm-ffi")
compiled.export_to_c("cute_add_one.o", "add_one", export_only_tvm_ffi_symbols=True)
subprocess.run(
    # --no-undefined: the static runtime archive links clean but fails at load.
    ["g++", "-shared", "-o", "cute_add_one.so", "cute_add_one.o"]
    + [f"-L{get_libdir()}", "-lcute_dsl_runtime"]
    + [f"-L{pathlib.Path(tvm_ffi.libinfo.find_libtvm_ffi()).parent}", "-ltvm_ffi"]
    + ["-Wl,--no-undefined"],
    check=True,
)
"""


def build_cutedsl_library(tmp_path):
    """Compile CuTeDSL the way a client would. In a subprocess, because importing
    cutlass here would load the runtime and mask the worker's own preload. The
    kernel needs a real file: the DSL re-reads its source with ``inspect``."""
    import subprocess
    import sys

    (tmp_path / "cute_add_one.py").write_text(CUTEDSL_KERNEL)
    (tmp_path / "build.py").write_text(CUTEDSL_BUILD)
    subprocess.run([sys.executable, "build.py"], cwd=tmp_path, check=True)
    return (tmp_path / "cute_add_one.so").read_bytes()


TRITON_KERNEL = """import triton
import triton.language as tl

@triton.jit
def add_one(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1.0, mask=mask)
"""


def ref(handle):
    return Ref(handle)


def runtime():
    from kcoral.gpu_runtime import GPURuntime

    return GPURuntime()


def decode_structural(encoded):
    value_type = encoded["type"]
    if value_type == "object":
        return {key: decode_structural(value) for key, value in encoded["value"].items()}
    if value_type == "array":
        return [decode_structural(value) for value in encoded["value"]]
    if value_type == "null":
        return None
    return encoded["value"]


def test_compile_correctness_and_benchmark():
    program = Program(
        [
            Upload("kernel", "module", source=KERNEL),
            Upload("reference", "module", source=REF),
            Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32", "seed": 0}]),
            Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
            Run("compiled", "builtin.compile_tirx", [ref("kernel"), {"N": 256}]),
            Run("invoke", ref("compiled"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            Run("check", "builtin.check_close", [ref("output"), ref("expected")]),
            Run(
                "timing",
                "builtin.benchmark",
                [ref("compiled"), ref("input"), ref("output"), {"warmup": 5, "repeat": 20}],
            ),
            Return("check", ref("check")),
            Return("timing", ref("timing")),
        ]
    )
    outcome = execute(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED"
    check = decode_structural(outcome.results["check"])
    timing = decode_structural(outcome.results["timing"])
    assert check["passed"] and check["max_abs_err"] == 0
    assert timing["latency_ms_median"] > 0 and timing["repeat"] == 20


def test_cuda_c_compile_correctness_and_benchmark():
    program = Program(
        [
            Upload("kernel", "module", source=CUDA_KERNEL, entry="add_one", language="cuda"),
            Upload("reference", "module", source=REF),
            Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32", "seed": 0}]),
            Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
            Run("compiled", "builtin.compile_cuda", [ref("kernel")]),
            Run("invoke", ref("compiled"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            Run("check", "builtin.check_close", [ref("output"), ref("expected")]),
            Run(
                "timing",
                "builtin.benchmark",
                [ref("compiled"), ref("input"), ref("output"), {"warmup": 5, "repeat": 20}],
            ),
            Return("check", ref("check")),
            Return("timing", ref("timing")),
        ]
    )
    outcome = execute(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    check = decode_structural(outcome.results["check"])
    timing = decode_structural(outcome.results["timing"])
    assert check["passed"] and check["max_abs_err"] == 0
    assert timing["latency_ms_median"] > 0 and timing["repeat"] == 20


def test_cuda_c_nvcc_error_is_compile_failure_and_names_the_mistake():
    bad = CUDA_KERNEL.replace("x.data_ptr()", "undeclared_symbol")
    outcome = execute(
        Program(
            [
                Upload("kernel", "module", source=bad, entry="add_one", language="cuda"),
                Run("compiled", "builtin.compile_cuda", [ref("kernel")]),
            ]
        ),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "compile"
    assert "undeclared_symbol" in outcome.error["message"]


def test_cuda_c_runs_on_the_arch_specific_target():
    """A plain sm_100 target cannot even assemble tcgen05, so building and running
    this kernel is what proves the worker selects sm_100a."""
    import numpy as np
    import torch

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("tcgen05 needs a Blackwell device")
    outcome = execute(
        Program(
            [
                Upload(
                    "kernel", "module", source=TMEM_KERNEL, entry="tmem_roundtrip", language="cuda"
                ),
                Run("out", "builtin.zeros", [{"shape": [32], "dtype": "int32"}]),
                Run("compiled", "builtin.compile_cuda", [ref("kernel")]),
                Run("invoke", ref("compiled"), [ref("out")]),
                Return("out", ref("out")),
            ]
        ),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "COMPLETED", outcome.error
    np.testing.assert_array_equal(
        np.frombuffer(outcome.binary_parts["return:0"], dtype=np.int32), np.arange(100, 132)
    )


def test_prebuilt_library_runs_and_benchmarks(tmp_path):
    """A client-compiled .so uploads as bytes and needs no compile instruction."""
    data = build_library(CUDA_KERNEL, "add_one", tmp_path)
    digest = compute_blob_hash(data)
    program = Program(
        [
            Upload("kernel", "library", blob=digest, entry="add_one"),
            Upload("reference", "module", source=REF),
            Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32", "seed": 0}]),
            Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
            Run("invoke", ref("kernel"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            Run("check", "builtin.check_close", [ref("output"), ref("expected")]),
            Run(
                "timing",
                "builtin.benchmark",
                [ref("kernel"), ref("input"), ref("output"), {"warmup": 5, "repeat": 20}],
            ),
            Return("check", ref("check")),
            Return("timing", ref("timing")),
        ],
        blob_bytes={digest: data},
    )
    outcome = execute(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    assert decode_structural(outcome.results["check"])["max_abs_err"] == 0
    assert decode_structural(outcome.results["timing"])["latency_ms_median"] > 0


def test_prebuilt_tirx_library_runs(tmp_path):
    """An export_library artifact carries its device code as an embedded blob, so
    this is the path that needs the TVM CUDA runtime loader registered."""
    data = build_tirx_library(tmp_path)
    digest = compute_blob_hash(data)
    program = Program(
        [
            Upload("kernel", "library", blob=digest, entry="add_one"),
            Upload("reference", "module", source=REF),
            Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32", "seed": 0}]),
            Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
            Run("invoke", ref("kernel"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            Run("check", "builtin.check_close", [ref("output"), ref("expected")]),
            Return("check", ref("check")),
        ],
        blob_bytes={digest: data},
    )
    outcome = execute(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    assert decode_structural(outcome.results["check"])["max_abs_err"] == 0


def test_prebuilt_cutedsl_library_runs(tmp_path):
    """CuTeDSL exports the same __tvm_ffi_<entry> symbol as CUDA C, but its object
    is not self-contained: it loads only because the worker preloads the runtime."""
    if importlib.util.find_spec("cutlass") is None:  # not importorskip: see the helper
        pytest.skip("CuTeDSL export requires cutlass")
    data = build_cutedsl_library(tmp_path)
    digest = compute_blob_hash(data)
    program = Program(
        [
            Upload("kernel", "library", blob=digest, entry="add_one"),
            Upload("reference", "module", source=REF),
            Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32", "seed": 0}]),
            Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
            Run("invoke", ref("kernel"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            Run("check", "builtin.check_close", [ref("output"), ref("expected")]),
            Return("check", ref("check")),
        ],
        blob_bytes={digest: data},
    )
    outcome = execute(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    assert decode_structural(outcome.results["check"])["max_abs_err"] == 0


def test_cutedsl_source_compiles_on_the_server():
    """The other CuTeDSL route: upload the kernel as text, so the client needs no
    CUDA toolchain of its own."""
    if importlib.util.find_spec("cutlass") is None:
        pytest.skip("CuTeDSL compilation requires cutlass")
    program = Program(
        [
            Upload("kernel", "module", source=CUTEDSL_KERNEL, entry="add_one"),
            Upload("reference", "module", source=REF),
            Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32", "seed": 0}]),
            Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
            # Compiling specializes on these tensors; the result takes torch ones.
            Run(
                "compiled",
                "builtin.compile_cutedsl",
                [ref("kernel"), ref("input"), ref("output")],
            ),
            Run("invoke", ref("compiled"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            Run("check", "builtin.check_close", [ref("output"), ref("expected")]),
            Return("check", ref("check")),
        ]
    )
    outcome = execute(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    assert decode_structural(outcome.results["check"])["max_abs_err"] == 0


def test_compile_cutedsl_reports_an_undecorated_kernel():
    """CuTeDSL writes colour into its diagnostics; the client gets it stripped."""
    if importlib.util.find_spec("cutlass") is None:
        pytest.skip("CuTeDSL compilation requires cutlass")
    outcome = execute(
        Program(
            [
                Upload("kernel", "module", source="def main(src, dst):\n    pass\n"),
                Run("input", "builtin.randn", [{"shape": [8], "dtype": "float32"}]),
                Run("compiled", "builtin.compile_cutedsl", [ref("kernel"), ref("input")]),
            ]
        ),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED"
    assert outcome.error["kind"] == "compile"
    assert "@cute.jit" in outcome.error["message"]
    assert "\x1b[" not in outcome.error["message"]


def test_triton_source_compiles_on_the_server():
    """A Triton kernel is uploaded and launched without the client writing a
    launcher: the grid travels as data and ``compile_triton`` binds it."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("Triton compilation requires triton")
    n = 4096
    program = Program(
        [
            Upload("kernel", "module", source=TRITON_KERNEL, entry="add_one"),
            Upload("reference", "module", source=REF),
            Run("input", "builtin.randn", [{"shape": [n], "dtype": "float32", "seed": 0}]),
            Run("output", "builtin.empty", [{"shape": [n], "dtype": "float32"}]),
            Run(
                "compiled",
                "builtin.compile_triton",
                [ref("kernel"), ref("input"), ref("output"), n, 256, {"grid": [n // 256]}],
            ),
            Run("invoke", ref("compiled"), [ref("input"), ref("output"), n, 256]),
            Run("expected", ref("reference"), [ref("input")]),
            Run("check", "builtin.check_close", [ref("output"), ref("expected")]),
            Run(
                "timing",
                "builtin.benchmark",
                [ref("compiled"), ref("input"), ref("output"), n, 256, {"repeat": 20}],
            ),
            Return("check", ref("check")),
            Return("timing", ref("timing")),
        ]
    )
    outcome = execute(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    assert decode_structural(outcome.results["check"])["max_abs_err"] == 0
    assert decode_structural(outcome.results["timing"])["latency_ms_median"] > 0


def test_library_with_a_wrong_entry_fails_to_compile(tmp_path):
    data = build_library(CUDA_KERNEL, "add_one", tmp_path)
    digest = compute_blob_hash(data)
    outcome = execute(
        Program(
            [Upload("kernel", "library", blob=digest, entry="not_there")],
            blob_bytes={digest: data},
        ),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "compile"
    assert "not_there" in outcome.error["message"]


def test_library_cache_is_only_a_memoization(tmp_path):
    """A cold worker must behave exactly like a warm one: every request carries the
    bytes, so dropping the cache changes speed and nothing else."""
    from kcoral import gpu_runtime

    data = build_library(CUDA_KERNEL, "add_one", tmp_path)
    digest = compute_blob_hash(data)

    def run_once():
        program = Program(
            [
                Upload("kernel", "library", blob=digest, entry="add_one"),
                Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32", "seed": 0}]),
                Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
                Run("invoke", ref("kernel"), [ref("input"), ref("output")]),
                Return("output", ref("output")),
            ],
            blob_bytes={digest: data},
        )
        return execute(program, runtime(), UNSHARED_GPU)

    gpu_runtime._LOADED_LIBRARIES.clear()
    cold = run_once()
    warm = run_once()  # served from the in-process cache
    gpu_runtime._LOADED_LIBRARIES.clear()
    cold_again = run_once()  # a respawned worker starts empty and must still work
    for outcome in (cold, warm, cold_again):
        assert outcome.status == "COMPLETED", outcome.error
    assert cold.binary_parts == warm.binary_parts == cold_again.binary_parts


def test_compile_tirx_reuses_an_already_compiled_kernel():
    from kcoral.builtin_ops import tirx

    tirx._COMPILED.clear()
    rt = runtime()
    kernel = rt.load_module(KERNEL)
    first = tirx.compile_tirx(kernel, {"N": 256})
    second = tirx.compile_tirx(kernel, {"N": 256})
    assert first is second  # same Executable, so codegen ran once
    assert tirx.compile_tirx(kernel, {"N": 512}) is not first  # a new shape still compiles


def test_benchmark_budget_counts_and_no_flush():
    program = Program(
        [
            Upload("kernel", "module", source=KERNEL),
            Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32", "seed": 0}]),
            Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
            Run("compiled", "builtin.compile_tirx", [ref("kernel"), {"N": 256}]),
            Run(
                "timing",
                "builtin.benchmark",
                [
                    ref("compiled"),
                    ref("input"),
                    ref("output"),
                    {"warmup_ms": 5, "repeat_ms": 20, "flush_l2": False},
                ],
            ),
            Return("timing", ref("timing")),
        ]
    )
    outcome = execute(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED"
    timing = decode_structural(outcome.results["timing"])
    assert timing["latency_ms_median"] > 0 and timing["flush_l2"] is False
    assert timing["warmup"] >= 1 and timing["repeat"] > 10


def test_direct_cupti_benchmarks_multiple_gpu_activities_twice():
    import torch

    from kcoral.builtin_ops import resolve

    source = torch.randn(4096, dtype=torch.float32, device="cuda")
    output = torch.empty_like(source)

    def two_operations(source, output):
        torch.add(source, 1.0, out=output)
        torch.mul(output, 2.0, out=output)

    measure = resolve("builtin.benchmark")
    config = {"warmup": 1, "repeat": 3, "flush_l2": False}
    first = measure(two_operations, source, output, config)
    second = measure(two_operations, source, output, config)
    assert first["latency_ms_median"] > 0 and first["repeat"] == 3
    assert second["latency_ms_median"] > 0 and second["repeat"] == 3
    assert first["activities_stable"] and second["activities_stable"]


def test_data_dependent_work_is_measured_and_flagged_unstable():
    import torch

    from kcoral.builtin_ops import resolve

    source = torch.randn(4096, dtype=torch.float32, device="cuda")
    output = torch.empty_like(source)
    calls = 0

    def sometimes_two_operations(source, output):
        nonlocal calls
        calls += 1
        torch.add(source, 1.0, out=output)
        if calls % 3 == 0:
            torch.mul(output, 2.0, out=output)

    measure = resolve("builtin.benchmark")
    timing = measure(sometimes_two_operations, source, output, {"warmup": 1, "repeat": 9})
    assert timing["latency_ms_median"] > 0
    assert timing["activities_stable"] is False


def test_direct_cupti_cleans_up_after_the_callable_fails():
    import torch

    from kcoral.builtin_ops import resolve
    from kcoral.errors import ExecutionError

    source = torch.randn(4096, dtype=torch.float32, device="cuda")
    output = torch.empty_like(source)
    calls = 0

    def fail_during_measurement(source, output):
        nonlocal calls
        calls += 1
        torch.add(source, 1.0, out=output)
        if calls == 2:  # one warmup call, then fail inside the CUPTI session
            raise RuntimeError("intentional benchmark failure")

    measure = resolve("builtin.benchmark")
    config = {"warmup": 1, "repeat": 2, "flush_l2": False}
    with pytest.raises(ExecutionError) as error:
        measure(fail_during_measurement, source, output, config)
    assert error.value.kind == "runtime"

    def add_one(source, output):
        torch.add(source, 1.0, out=output)

    recovered = measure(add_one, source, output, config)
    assert recovered["latency_ms_median"] > 0 and recovered["repeat"] == 2


def test_python_syntax_error_is_parse_failure():
    outcome = execute(
        Program([Upload("kernel", "module", source="def bad(:\n    pass\n")]),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "parse"
    assert outcome.error["instruction_index"] == 0


def test_tirx_error_is_parse_failure():
    bad_kernel = KERNEL.replace("B[i] = A[i] + 1.0", "B[i] = A[i] + undefined_symbol")
    outcome = execute(
        Program(
            [
                Upload("kernel", "module", source=bad_kernel),
                Run("compiled", "builtin.compile_tirx", [ref("kernel"), {"N": 16}]),
            ]
        ),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "parse"
    assert outcome.error["instruction_index"] == 1


def test_compile_on_non_kernel_is_compile_failure():
    outcome = execute(
        Program(
            [
                Run("input", "builtin.randn", [{"shape": [4], "dtype": "float32"}]),
                Run("compiled", "builtin.compile_tirx", [ref("input")]),
            ]
        ),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "compile"
    assert outcome.error["instruction_index"] == 1


def test_prim_func_kernel_compiles_directly():
    outcome = execute(
        Program(
            [
                Upload("kernel", "module", source=PRIM_KERNEL),
                Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32"}]),
                Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
                Run("compiled", "builtin.compile_tirx", [ref("kernel")]),
                Run("invoke", ref("compiled"), [ref("input"), ref("output")]),
            ]
        ),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "COMPLETED"


def test_prim_func_with_bindings_is_compile_failure():
    outcome = execute(
        Program(
            [
                Upload("kernel", "module", source=PRIM_KERNEL),
                Run("compiled", "builtin.compile_tirx", [ref("kernel"), {"N": 256}]),
            ]
        ),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "compile"
    assert outcome.error["instruction_index"] == 1


def test_bad_binding_name_is_compile_failure():
    outcome = execute(
        Program(
            [
                Upload("kernel", "module", source=KERNEL),
                Run("compiled", "builtin.compile_tirx", [ref("kernel"), {"WRONG": 1}]),
            ]
        ),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "compile"
    assert outcome.error["instruction_index"] == 1


def test_assert_close_failure_stops_without_results():
    wrong = KERNEL.replace("A[i] + 1.0", "A[i] + 2.0")
    program = Program(
        [
            Upload("kernel", "module", source=wrong),
            Upload("reference", "module", source=REF),
            Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32"}]),
            Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
            Run("compiled", "builtin.compile_tirx", [ref("kernel"), {"N": 256}]),
            Run("invoke", ref("compiled"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            Run("check", "builtin.assert_close", [ref("output"), ref("expected")]),
            Return("output", ref("output")),
        ]
    )
    outcome = execute(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "FAILED" and outcome.results == {}
    assert outcome.error["kind"] == "correctness" and outcome.error["instruction_index"] == 7
    assert outcome.error["instruction_op"] == "run" and outcome.error["instruction_id"] == "check"


def test_timing_returned_before_a_correctness_failure_is_kept():
    """Returning timing before the check keeps the benchmark when the kernel is wrong."""
    wrong = KERNEL.replace("A[i] + 1.0", "A[i] + 2.0")
    program = Program(
        [
            Upload("kernel", "module", source=wrong),
            Upload("reference", "module", source=REF),
            Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32"}]),
            Run("output", "builtin.empty", [{"shape": [256], "dtype": "float32"}]),
            Run("compiled", "builtin.compile_tirx", [ref("kernel"), {"N": 256}]),
            Run(
                "timing",
                "builtin.benchmark",
                [ref("compiled"), ref("input"), ref("output"), {"warmup": 5, "repeat": 20}],
            ),
            Return("timing", ref("timing")),
            Run("expected", ref("reference"), [ref("input")]),
            Run("check", "builtin.assert_close", [ref("output"), ref("expected")]),
            Return("output", ref("output")),
        ]
    )
    outcome = execute(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "FAILED"
    assert set(outcome.results) == {"timing"}
    assert decode_structural(outcome.results["timing"])["latency_ms_median"] > 0
    assert outcome.error["kind"] == "correctness" and outcome.error["instruction_id"] == "check"


def test_uploaded_and_returned_tensor_bytes():
    import numpy as np

    array = np.arange(16, dtype=np.float32)
    raw = array.tobytes()
    digest = compute_blob_hash(raw)
    program = Program(
        [
            Upload("tensor", "tensor", blob=digest, dtype="float32", shape=[16]),
            Return("tensor", ref("tensor")),
        ],
        blob_bytes={digest: raw},
    )
    outcome = execute(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED"
    np.testing.assert_array_equal(
        np.frombuffer(outcome.binary_parts["return:0"], dtype=np.float32), array
    )


BUILTIN_CALLER = """from kcoral import builtin

def main(x):
    y = builtin.randn({"shape": [256], "dtype": "float32", "seed": 0})
    return builtin.check_close(x, y)
"""


def test_uploaded_code_calls_builtins_directly():
    """The same seed on both sides must produce identical tensors, which proves
    ``kcoral.builtin`` hands uploaded code the very functions the run
    instructions name."""
    program = Program(
        [
            Upload("caller", "module", source=BUILTIN_CALLER),
            Run("input", "builtin.randn", [{"shape": [256], "dtype": "float32", "seed": 0}]),
            Run("check", ref("caller"), [ref("input")]),
            Return("check", ref("check")),
        ]
    )
    outcome = execute(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    check = decode_structural(outcome.results["check"])
    assert check["passed"] and check["max_abs_err"] == 0


def test_benchmark_budgets_survive_the_l2_flush():
    """The iteration estimate must not include the first touch of the flush
    buffer: freshly allocated and 2x L2, it costs ~16x a warm one and would
    silently shrink both budgets by that factor."""
    import torch

    from kcoral.builtin_ops import resolve

    x = torch.randn(4096, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)

    def add_one(src, dst):
        torch.add(src, 1.0, out=dst)

    result = resolve("builtin.benchmark")(add_one, x, y, {})
    # A microsecond kernel against a 25/100 ms budget: hundreds and thousands,
    # not the tens an inflated estimate produced.
    assert result["latency_ms_median"] < 0.1
    assert result["warmup"] > 100 and result["repeat"] > 500


MODULE_SCOPE_ALLOCATION = """import torch

WEIGHTS = torch.empty(1024**3, dtype=torch.int8, device="cuda")  # 1 GiB at module scope


def main():
    return WEIGHTS.sum()
"""


def test_module_scope_allocations_do_not_survive_the_request():
    """Without the `gc.collect()` in `reset()` this grew by the full 1 GiB per
    request: the namespace cycle kept the module-scope tensor live."""
    import torch

    rt = runtime()
    rt.reset()
    baseline = torch.cuda.mem_get_info()[0]
    for _ in range(3):
        program = Program(
            [
                Upload("kernel", "module", source=MODULE_SCOPE_ALLOCATION),
                Run("total", ref("kernel"), []),
                Return("total", ref("total")),
            ]
        )
        outcome = execute(program, rt, UNSHARED_GPU)
        assert outcome.status == "COMPLETED", outcome.error
    # Generous slack: free memory is shared with co-tenants, and the regression
    # this guards is 1 GiB per request.
    leaked = baseline - torch.cuda.mem_get_info()[0]
    assert leaked < 512 * 1024**2, f"{leaked / 1024**2:.0f} MiB not reclaimed across 3 requests"


# How flashinfer-bench-evolve's worker loads a candidate: a uniquely-named module
# from a temp file, registered in sys.modules and popped in a finally.
CANDIDATE_HARNESS = '''
import importlib.util, pathlib, sys, tempfile

CANDIDATE_SOURCE = """
import torch

STATE = torch.empty(512 * 1024**2, dtype=torch.int8, device="cuda")  # setup state


def run():
    return STATE.sum()
"""


def main(tag):
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix=f"candidate_{tag}_", delete=False
    ) as handle:
        handle.write(CANDIDATE_SOURCE)
        path = pathlib.Path(handle.name)
    name = f"_tirx_candidate_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    try:
        spec.loader.exec_module(loaded)
        prepare = lambda: loaded.run()  # noqa: E731 - the closure the suite is given
        return float(prepare())
    finally:
        sys.modules.pop(name, None)
        path.unlink(missing_ok=True)
'''


def test_popped_candidate_modules_do_not_survive_the_request():
    """The kernel-evolution pattern: one candidate loaded and unregistered per
    request. Unregistering drops one reference but leaves the namespace cycle, so
    without the collection every candidate's setup state stayed resident."""
    import torch

    rt = runtime()
    rt.reset()
    baseline = torch.cuda.mem_get_info()[0]
    for tag in range(4):
        program = Program(
            [
                Upload("harness", "module", source=CANDIDATE_HARNESS),
                Run("evaluated", ref("harness"), [tag]),
                Return("evaluated", ref("evaluated")),
            ]
        )
        outcome = execute(program, rt, UNSHARED_GPU)
        assert outcome.status == "COMPLETED", outcome.error
    leaked = baseline - torch.cuda.mem_get_info()[0]
    assert leaked < 512 * 1024**2, f"{leaked / 1024**2:.0f} MiB not reclaimed across 4 candidates"
