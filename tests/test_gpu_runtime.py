"""GPU integration tests for the runtime with uploaded TIRx and CUDA C harnesses."""

import importlib.util
import os
import pathlib

import pytest
from support.programs import harness_call, python_call

from kcoral.keys import compute_blob_hash
from kcoral.schemas import GetFunction, Program, Ref, Return, Run, Upload
from kcoral.testing import UNSHARED_GPU, execute_for_test

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

MULTI_ENTRY_CUDA_KERNEL = r"""
#include <tvm/ffi/extra/c_env_api.h>

__global__ void add_value_kernel(const float* x, float* y, int n, float value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + value;
}

void add_value(tvm::ffi::TensorView x, tvm::ffi::TensorView y, float value) {
  int n = static_cast<int>(x.numel());
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
  add_value_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      static_cast<const float*>(x.data_ptr()), static_cast<float*>(y.data_ptr()), n, value);
}

void add_one(tvm::ffi::TensorView x, tvm::ffi::TensorView y) { add_value(x, y, 1.0f); }
void add_two(tvm::ffi::TensorView x, tvm::ffi::TensorView y) { add_value(x, y, 2.0f); }
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
    from support.cuda import _cuda_arch_list
    from tvm_ffi.cpp import extension

    os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", _cuda_arch_list())
    cu = tmp_path / f"{entry}.cu"
    cu.write_text(extension._decorate_with_tvm_ffi(source, {entry: ""}))
    return pathlib.Path(tvm_ffi.cpp.build(name=entry, cuda_files=[str(cu)])).read_bytes()


def build_multi_entry_library(source, entries, tmp_path):
    """Build one TVM-FFI library that exports every name in ``entries``."""
    import tvm_ffi.cpp
    from support.cuda import _cuda_arch_list
    from tvm_ffi.cpp import extension

    os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", _cuda_arch_list())
    name = "multi_entry_" + "_".join(entries)
    cu = tmp_path / f"{name}.cu"
    cu.write_text(extension._decorate_with_tvm_ffi(source, dict.fromkeys(entries, "")))
    return pathlib.Path(tvm_ffi.cpp.build(name=name, cuda_files=[str(cu)])).read_bytes()


def build_tirx_library(tmp_path):
    """Compile TIRx the way a client would, with an explicit target so no GPU is
    consulted. The exported name comes from the function, not the IRModule key."""
    import tvm

    from kcoral.gpu_runtime import GPURuntime, describe_target

    runtime = GPURuntime()
    module = runtime.load_module(PRIM_KERNEL.replace("def main(", "def add_one("))
    prim_func = runtime.get_function(module, "add_one")
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


def test_python_module_get_function_selects_named_objects():
    rt = runtime()
    module = rt.load_module(
        "def add_one(value):\n    return value + 1\n\ndef times_two(value):\n    return value * 2\n"
    )

    add_one = rt.get_function(module, "add_one")
    times_two = rt.get_function(module, "times_two")

    assert times_two(add_one(20)) == 42


def tirx_benchmark_program():
    return Program(
        [
            Upload("kernel_module", "module", source=KERNEL),
            GetFunction("kernel", ref("kernel_module"), "main"),
            Upload("reference_module", "module", source=REF),
            GetFunction("reference", ref("reference_module"), "main"),
            *python_call(
                "input",
                """import torch
def main(shape):
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
""",
                [[256]],
            ),
            *python_call(
                "output",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            *harness_call("compiled", "compile_tirx", [ref("kernel"), {"N": 256}]),
            Run("invoke", ref("compiled"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            *harness_call("check", "check_close", [ref("output"), ref("expected")]),
            *harness_call(
                "timing",
                "benchmark",
                [ref("compiled"), ref("input"), ref("output"), {"warmup": 5, "repeat": 20}],
            ),
            Return("check", ref("check")),
            Return("timing", ref("timing")),
        ]
    )


def test_compile_correctness_and_benchmark():
    outcome = execute_for_test(tirx_benchmark_program(), runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    check = decode_structural(outcome.results["check"])
    timing = decode_structural(outcome.results["timing"])
    assert check["passed"] and check["max_abs_err"] == 0
    assert timing["latency_ms_median"] > 0 and timing["repeat"] == 20


CPU_REFERENCE = (
    "import torch\n\n"
    "def main(n):\n"
    "    x = torch.arange(n, dtype=torch.float32)\n"
    "    return float(x @ x)\n"
)
CUDA_IN_CPU_ONLY = (
    "import torch\n\ndef main(n):\n    return float(torch.zeros(n, device='cuda').sum())\n"
)


def cpu_only_call(source, gpu_runtime):
    return execute_for_test(
        Program(
            [
                Upload("module", "module", source=source),
                GetFunction("fn", ref("module"), "main", cpu_only=True),
                Run("value", ref("fn"), [256]),
                Return("value", ref("value")),
            ]
        ),
        gpu_runtime,
        UNSHARED_GPU,
    )


def test_a_cpu_only_function_is_checked_against_the_cuda_api():
    gpu_runtime = runtime()
    host = cpu_only_call(CPU_REFERENCE, gpu_runtime)
    assert host.status == "COMPLETED", host.error
    assert host.results["value"]["value"] == sum(i * i for i in range(256))

    device = cpu_only_call(CUDA_IN_CPU_ONLY, gpu_runtime)
    assert device.status == "FAILED"
    error = device.error
    assert error["kind"] == "gpu_access" and error["instruction_id"] == "value"
    assert error["cuda_call"].startswith("cu")
    assert error["location"].startswith("<uploaded:") and error["location"].endswith(" in main")
    assert "torch.zeros" in error["traceback"]

    # CUPTI is shared with the test harness: each leaves it usable by the other.
    timed = execute_for_test(tirx_benchmark_program(), gpu_runtime, UNSHARED_GPU)
    assert timed.status == "COMPLETED", timed.error
    assert cpu_only_call(CUDA_IN_CPU_ONLY, gpu_runtime).error["kind"] == "gpu_access"


HOST_LIBRARY = """
int64_t add_one(int64_t value) { return value + 1; }
"""


def test_a_cpu_only_library_function_runs_off_the_gpu(tmp_path):
    """Calling into a library must not touch CUDA on the function's behalf."""
    data = build_library(HOST_LIBRARY, "add_one", tmp_path)
    outcome = execute_for_test(
        Program(
            [
                Upload("library", "library", blob=compute_blob_hash(data)),
                GetFunction("add_one", ref("library"), "add_one", cpu_only=True),
                Run("value", ref("add_one"), [41]),
                Return("value", ref("value")),
            ],
            blob_bytes={compute_blob_hash(data): data},
        ),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "COMPLETED", outcome.error
    assert outcome.results["value"] == {"type": "integer", "value": 42}


def test_a_cpu_reference_compares_against_a_gpu_tensor_as_is():
    source = "import torch\n\ndef main(n):\n    return torch.zeros(n)\n"
    outcome = execute_for_test(
        Program(
            [
                Upload("module", "module", source=source),
                GetFunction("reference", ref("module"), "main", cpu_only=True),
                Run("expected", ref("reference"), [256]),
                *python_call(
                    "actual",
                    """import torch
def main(shape):
    return torch.zeros(shape, dtype=torch.float32, device="cuda")
""",
                    [[256]],
                ),
                *harness_call("check", "check_close", [ref("actual"), ref("expected")]),
                Return("check", ref("check")),
            ]
        ),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "COMPLETED", outcome.error
    assert decode_structural(outcome.results["check"])["passed"]


def test_cuda_c_compile_correctness_and_benchmark():
    program = Program(
        [
            Upload("kernel_module", "module", source=CUDA_KERNEL, language="cuda"),
            GetFunction("kernel", ref("kernel_module"), "add_one"),
            Upload("reference_module", "module", source=REF),
            GetFunction("reference", ref("reference_module"), "main"),
            *python_call(
                "input",
                """import torch
def main(shape):
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
""",
                [[256]],
            ),
            *python_call(
                "output",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            *harness_call("compiled", "compile_cuda", [ref("kernel")]),
            Run("invoke", ref("compiled"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            *harness_call("check", "check_close", [ref("output"), ref("expected")]),
            *harness_call(
                "timing",
                "benchmark",
                [ref("compiled"), ref("input"), ref("output"), {"warmup": 5, "repeat": 20}],
            ),
            Return("check", ref("check")),
            Return("timing", ref("timing")),
        ]
    )
    outcome = execute_for_test(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    check = decode_structural(outcome.results["check"])
    timing = decode_structural(outcome.results["timing"])
    assert check["passed"] and check["max_abs_err"] == 0
    assert timing["latency_ms_median"] > 0 and timing["repeat"] == 20


def test_cuda_c_nvcc_error_is_compile_failure_and_names_the_mistake():
    bad = CUDA_KERNEL.replace("x.data_ptr()", "undeclared_symbol")
    outcome = execute_for_test(
        Program(
            [
                Upload("kernel_module", "module", source=bad, language="cuda"),
                GetFunction("kernel", ref("kernel_module"), "add_one"),
                *harness_call("compiled", "compile_cuda", [ref("kernel")]),
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
    outcome = execute_for_test(
        Program(
            [
                Upload("kernel_module", "module", source=TMEM_KERNEL, language="cuda"),
                GetFunction("kernel", ref("kernel_module"), "tmem_roundtrip"),
                *python_call(
                    "out",
                    """import torch
def main(shape):
    return torch.zeros(shape, dtype=torch.int32, device="cuda")
""",
                    [[32]],
                ),
                *harness_call("compiled", "compile_cuda", [ref("kernel")]),
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
            Upload("kernel_module", "library", blob=digest),
            GetFunction("kernel", ref("kernel_module"), "add_one"),
            Upload("reference_module", "module", source=REF),
            GetFunction("reference", ref("reference_module"), "main"),
            *python_call(
                "input",
                """import torch
def main(shape):
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
""",
                [[256]],
            ),
            *python_call(
                "output",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            Run("invoke", ref("kernel"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            *harness_call("check", "check_close", [ref("output"), ref("expected")]),
            *harness_call(
                "timing",
                "benchmark",
                [ref("kernel"), ref("input"), ref("output"), {"warmup": 5, "repeat": 20}],
            ),
            Return("check", ref("check")),
            Return("timing", ref("timing")),
        ],
        blob_bytes={digest: data},
    )
    outcome = execute_for_test(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    assert decode_structural(outcome.results["check"])["max_abs_err"] == 0
    assert decode_structural(outcome.results["timing"])["latency_ms_median"] > 0


def test_prebuilt_library_module_binds_and_runs_multiple_functions(tmp_path):
    data = build_multi_entry_library(
        MULTI_ENTRY_CUDA_KERNEL,
        ["add_one", "add_two"],
        tmp_path,
    )
    digest = compute_blob_hash(data)
    reference = "def one(a):\n    return a + 1.0\n\ndef two(a):\n    return a + 2.0\n"
    program = Program(
        [
            Upload("kernels", "library", blob=digest),
            GetFunction("add_one", ref("kernels"), "add_one"),
            GetFunction("add_two", ref("kernels"), "add_two"),
            Upload("reference", "module", source=reference),
            GetFunction("one", ref("reference"), "one"),
            GetFunction("two", ref("reference"), "two"),
            *python_call(
                "input",
                """import torch
def main(shape):
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
""",
                [[256]],
            ),
            *python_call(
                "output_one",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            *python_call(
                "output_two",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            Run("invoke_one", ref("add_one"), [ref("input"), ref("output_one")]),
            Run("invoke_two", ref("add_two"), [ref("input"), ref("output_two")]),
            Run("expected_one", ref("one"), [ref("input")]),
            Run("expected_two", ref("two"), [ref("input")]),
            *harness_call("check_one", "check_close", [ref("output_one"), ref("expected_one")]),
            *harness_call("check_two", "check_close", [ref("output_two"), ref("expected_two")]),
            *harness_call(
                "timing",
                "benchmark",
                [
                    ref("add_two"),
                    ref("input"),
                    ref("output_two"),
                    {"warmup": 5, "repeat": 20},
                ],
            ),
            Return("check_one", ref("check_one")),
            Return("check_two", ref("check_two")),
            Return("timing", ref("timing")),
        ],
        blob_bytes={digest: data},
    )

    outcome = execute_for_test(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    assert decode_structural(outcome.results["check_one"])["max_abs_err"] == 0
    assert decode_structural(outcome.results["check_two"])["max_abs_err"] == 0
    assert decode_structural(outcome.results["timing"])["latency_ms_median"] > 0


def test_prebuilt_tirx_library_runs(tmp_path):
    """An export_library artifact carries its device code as an embedded blob, so
    this is the path that needs the TVM CUDA runtime loader registered."""
    data = build_tirx_library(tmp_path)
    digest = compute_blob_hash(data)
    program = Program(
        [
            Upload("kernel_module", "library", blob=digest),
            GetFunction("kernel", ref("kernel_module"), "add_one"),
            Upload("reference_module", "module", source=REF),
            GetFunction("reference", ref("reference_module"), "main"),
            *python_call(
                "input",
                """import torch
def main(shape):
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
""",
                [[256]],
            ),
            *python_call(
                "output",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            Run("invoke", ref("kernel"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            *harness_call("check", "check_close", [ref("output"), ref("expected")]),
            Return("check", ref("check")),
        ],
        blob_bytes={digest: data},
    )
    outcome = execute_for_test(program, runtime(), UNSHARED_GPU)
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
            Upload("kernel_module", "library", blob=digest),
            GetFunction("kernel", ref("kernel_module"), "add_one"),
            Upload("reference_module", "module", source=REF),
            GetFunction("reference", ref("reference_module"), "main"),
            *python_call(
                "input",
                """import torch
def main(shape):
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
""",
                [[256]],
            ),
            *python_call(
                "output",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            Run("invoke", ref("kernel"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            *harness_call("check", "check_close", [ref("output"), ref("expected")]),
            Return("check", ref("check")),
        ],
        blob_bytes={digest: data},
    )
    outcome = execute_for_test(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    assert decode_structural(outcome.results["check"])["max_abs_err"] == 0


def test_cutedsl_source_compiles_on_the_server():
    """The other CuTeDSL route: upload the kernel as text, so the client needs no
    CUDA toolchain of its own."""
    if importlib.util.find_spec("cutlass") is None:
        pytest.skip("CuTeDSL compilation requires cutlass")
    program = Program(
        [
            Upload("kernel_module", "module", source=CUTEDSL_KERNEL),
            GetFunction("kernel", ref("kernel_module"), "add_one"),
            Upload("reference_module", "module", source=REF),
            GetFunction("reference", ref("reference_module"), "main"),
            *python_call(
                "input",
                """import torch
def main(shape):
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
""",
                [[256]],
            ),
            *python_call(
                "output",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            # Compiling specializes on these tensors; the result takes torch ones.
            *harness_call(
                "compiled", "compile_cutedsl", [ref("kernel"), ref("input"), ref("output")]
            ),
            Run("invoke", ref("compiled"), [ref("input"), ref("output")]),
            Run("expected", ref("reference"), [ref("input")]),
            *harness_call("check", "check_close", [ref("output"), ref("expected")]),
            Return("check", ref("check")),
        ]
    )
    outcome = execute_for_test(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    assert decode_structural(outcome.results["check"])["max_abs_err"] == 0


def test_triton_source_compiles_on_the_server():
    """A Triton kernel is uploaded and launched without the client writing a
    launcher: the grid travels as data and ``compile_triton`` binds it."""
    if importlib.util.find_spec("triton") is None:
        pytest.skip("Triton compilation requires triton")
    n = 4096
    program = Program(
        [
            Upload("kernel_module", "module", source=TRITON_KERNEL),
            GetFunction("kernel", ref("kernel_module"), "add_one"),
            Upload("reference_module", "module", source=REF),
            GetFunction("reference", ref("reference_module"), "main"),
            *python_call(
                "input",
                """import torch
def main(shape):
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
""",
                [[n]],
            ),
            *python_call(
                "output",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[n]],
            ),
            *harness_call(
                "compiled",
                "compile_triton",
                [ref("kernel"), ref("input"), ref("output"), n, 256, {"grid": [n // 256]}],
            ),
            Run("invoke", ref("compiled"), [ref("input"), ref("output"), n, 256]),
            Run("expected", ref("reference"), [ref("input")]),
            *harness_call("check", "check_close", [ref("output"), ref("expected")]),
            *harness_call(
                "timing",
                "benchmark",
                [ref("compiled"), ref("input"), ref("output"), n, 256, {"repeat": 20}],
            ),
            Return("check", ref("check")),
            Return("timing", ref("timing")),
        ]
    )
    outcome = execute_for_test(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED", outcome.error
    assert decode_structural(outcome.results["check"])["max_abs_err"] == 0
    assert decode_structural(outcome.results["timing"])["latency_ms_median"] > 0


def test_library_with_a_missing_function_fails_to_compile(tmp_path):
    data = build_library(CUDA_KERNEL, "add_one", tmp_path)
    digest = compute_blob_hash(data)
    outcome = execute_for_test(
        Program(
            [
                Upload("kernel_module", "library", blob=digest),
                GetFunction("kernel", ref("kernel_module"), "not_there"),
            ],
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
                Upload("kernel_module", "library", blob=digest),
                GetFunction("kernel", ref("kernel_module"), "add_one"),
                *python_call(
                    "input",
                    """import torch
def main(shape):
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
""",
                    [[256]],
                ),
                *python_call(
                    "output",
                    """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                    [[256]],
                ),
                Run("invoke", ref("kernel"), [ref("input"), ref("output")]),
                Return("output", ref("output")),
            ],
            blob_bytes={digest: data},
        )
        return execute_for_test(program, runtime(), UNSHARED_GPU)

    gpu_runtime._LOADED_LIBRARIES.clear()
    cold = run_once()
    warm = run_once()  # served from the in-process cache
    gpu_runtime._LOADED_LIBRARIES.clear()
    cold_again = run_once()  # a respawned worker starts empty and must still work
    for outcome in (cold, warm, cold_again):
        assert outcome.status == "COMPLETED", outcome.error
    assert cold.binary_parts == warm.binary_parts == cold_again.binary_parts


def test_python_syntax_error_is_parse_failure():
    outcome = execute_for_test(
        Program([Upload("kernel", "module", source="def bad(:\n    pass\n")]),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "parse"
    assert outcome.error["instruction_index"] == 0


def test_tirx_error_is_parse_failure():
    bad_kernel = KERNEL.replace("B[i] = A[i] + 1.0", "B[i] = A[i] + undefined_symbol")
    outcome = execute_for_test(
        Program(
            [
                Upload("kernel_module", "module", source=bad_kernel),
                GetFunction("kernel", ref("kernel_module"), "main"),
                *harness_call("compiled", "compile_tirx", [ref("kernel"), {"N": 16}]),
            ]
        ),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "FAILED" and outcome.error["kind"] == "parse"
    assert outcome.error["instruction_index"] == 4


def test_prim_func_kernel_compiles_directly():
    outcome = execute_for_test(
        Program(
            [
                Upload("kernel_module", "module", source=PRIM_KERNEL),
                GetFunction("kernel", ref("kernel_module"), "main"),
                *python_call(
                    "input",
                    """import torch
def main(shape):
    return torch.randn(shape, dtype=torch.float32, device="cuda")
""",
                    [[256]],
                ),
                *python_call(
                    "output",
                    """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                    [[256]],
                ),
                *harness_call("compiled", "compile_tirx", [ref("kernel")]),
                Run("invoke", ref("compiled"), [ref("input"), ref("output")]),
            ]
        ),
        runtime(),
        UNSHARED_GPU,
    )
    assert outcome.status == "COMPLETED"


def test_timing_returned_before_a_correctness_failure_is_kept():
    """Returning timing before the check keeps the benchmark when the kernel is wrong."""
    wrong = KERNEL.replace("A[i] + 1.0", "A[i] + 2.0")
    program = Program(
        [
            Upload("kernel_module", "module", source=wrong),
            GetFunction("kernel", ref("kernel_module"), "main"),
            Upload("reference_module", "module", source=REF),
            GetFunction("reference", ref("reference_module"), "main"),
            *python_call(
                "input",
                """import torch
def main(shape):
    return torch.randn(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            *python_call(
                "output",
                """import torch
def main(shape):
    return torch.empty(shape, dtype=torch.float32, device="cuda")
""",
                [[256]],
            ),
            *harness_call("compiled", "compile_tirx", [ref("kernel"), {"N": 256}]),
            *harness_call(
                "timing",
                "benchmark",
                [ref("compiled"), ref("input"), ref("output"), {"warmup": 5, "repeat": 20}],
            ),
            Return("timing", ref("timing")),
            Run("expected", ref("reference"), [ref("input")]),
            *harness_call("check", "assert_close", [ref("output"), ref("expected")]),
            Return("output", ref("output")),
        ]
    )
    outcome = execute_for_test(program, runtime(), UNSHARED_GPU)
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
    outcome = execute_for_test(program, runtime(), UNSHARED_GPU)
    assert outcome.status == "COMPLETED"
    np.testing.assert_array_equal(
        np.frombuffer(outcome.binary_parts["return:0"], dtype=np.float32), array
    )


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
                Upload("kernel_module", "module", source=MODULE_SCOPE_ALLOCATION),
                GetFunction("kernel", ref("kernel_module"), "main"),
                Run("total", ref("kernel"), []),
                Return("total", ref("total")),
            ]
        )
        outcome = execute_for_test(program, rt, UNSHARED_GPU)
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
                Upload("harness_module", "module", source=CANDIDATE_HARNESS),
                GetFunction("harness", ref("harness_module"), "main"),
                Run("evaluated", ref("harness"), [tag]),
                Return("evaluated", ref("evaluated")),
            ]
        )
        outcome = execute_for_test(program, rt, UNSHARED_GPU)
        assert outcome.status == "COMPLETED", outcome.error
    leaked = baseline - torch.cuda.mem_get_info()[0]
    assert leaked < 512 * 1024**2, f"{leaked / 1024**2:.0f} MiB not reclaimed across 4 candidates"


def test_compile_tirx_reuses_an_already_compiled_kernel():
    from kcoral import builtins as tirx

    tirx._COMPILED.clear()
    rt = runtime()
    module = rt.load_module(KERNEL)
    kernel = rt.get_function(module, "main")
    first = tirx.compile_tirx(kernel, {"N": 256})
    second = tirx.compile_tirx(kernel, {"N": 256})
    assert first is second  # same Executable, so codegen ran once
    assert tirx.compile_tirx(kernel, {"N": 512}) is not first


def test_direct_cupti_benchmarks_multiple_gpu_activities_twice():
    import torch

    from kcoral.builtins import benchmark

    source = torch.randn(4096, dtype=torch.float32, device="cuda")
    output = torch.empty_like(source)

    def two_operations(source, output):
        torch.add(source, 1.0, out=output)
        torch.mul(output, 2.0, out=output)

    measure = benchmark
    config = {"warmup": 1, "repeat": 3, "flush_l2": False}
    first = measure(two_operations, source, output, config)
    second = measure(two_operations, source, output, config)
    assert first["latency_ms_median"] > 0 and first["repeat"] == 3
    assert second["latency_ms_median"] > 0 and second["repeat"] == 3
    assert first["activities_stable"] and second["activities_stable"]


def test_data_dependent_work_is_measured_and_flagged_unstable():
    import torch

    from kcoral.builtins import benchmark

    source = torch.randn(4096, dtype=torch.float32, device="cuda")
    output = torch.empty_like(source)
    calls = 0

    def sometimes_two_operations(source, output):
        nonlocal calls
        calls += 1
        torch.add(source, 1.0, out=output)
        if calls % 3 == 0:
            torch.mul(output, 2.0, out=output)

    measure = benchmark
    timing = measure(sometimes_two_operations, source, output, {"warmup": 1, "repeat": 9})
    assert timing["latency_ms_median"] > 0
    assert timing["activities_stable"] is False


def test_direct_cupti_cleans_up_after_the_callable_fails():
    import torch

    from kcoral.builtins import benchmark
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

    measure = benchmark
    config = {"warmup": 1, "repeat": 2, "flush_l2": False}
    with pytest.raises(ExecutionError) as error:
        measure(fail_during_measurement, source, output, config)
    assert error.value.kind == "runtime"

    def add_one(source, output):
        torch.add(source, 1.0, out=output)

    recovered = measure(add_one, source, output, config)
    assert recovered["latency_ms_median"] > 0 and recovered["repeat"] == 2


def test_benchmark_budgets_survive_the_l2_flush():
    """The iteration estimate must not include the first touch of the flush
    buffer: freshly allocated and 2x L2, it costs ~16x a warm one and would
    silently shrink both budgets by that factor."""
    import torch

    from kcoral.builtins import benchmark

    x = torch.randn(4096, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)

    def add_one(src, dst):
        torch.add(src, 1.0, out=dst)

    result = benchmark(add_one, x, y, {})
    # A microsecond kernel against a 25/100 ms budget: hundreds and thousands,
    # not the tens an inflated estimate produced.
    assert result["latency_ms_median"] < 0.1
    assert result["warmup"] > 100 and result["repeat"] > 500
