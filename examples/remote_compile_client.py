"""Compile, check and time TIRx, CuTeDSL, CUDA C and Triton kernels remotely.

The client uploads kernel and compiler code and needs no local CUDA toolchain.
Compilation retains the GPU lease because it
can call CUDA; the CPU reference explicitly releases it. Timing uses CUPTI GPU activity spans.
"""

from __future__ import annotations

import os

import numpy as np

from kcoral import Client, Program

N = 256

TIRX_KERNEL = r"""
from __future__ import annotations
from tvm.script import tirx as T


@T.jit
def main(
    A: T.Buffer((N,), "float32"),
    B: T.Buffer((N,), "float32"),
    *,
    N: T.constexpr,
):
    T.device_entry()
    i = T.cta_id([N])
    t = T.thread_id([1])
    B[i] = A[i] + 1.0
"""

CUTEDSL_KERNEL = r"""
import cutlass.cute as cute


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

# The selected function is exported through TVM FFI, so it takes TensorView parameters and
# returns void; the inline compiler adds the includes and export macro.
CUDA_KERNEL = r"""
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


TRITON_KERNEL = r"""
import triton
import triton.language as tl


@triton.jit
def add_one(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1.0, mask=mask)
"""


# The reference needs no GPU, so it is declared cpu_only when selected: the worker
# hands the GPU over for the call and fails it should it touch CUDA after all.
# assert_close takes its CPU result as is.
CPU_REFERENCE = r"""
import torch


def expected(n):
    return torch.arange(n, dtype=torch.float32) + 1.0
"""


# These functions and imports are uploaded to the server.
OPERATIONS = r"""
from kcoral.builtins import benchmark, compile_tirx


def empty(spec):
    import torch

    return torch.empty(spec["shape"], dtype=getattr(torch, spec["dtype"]), device="cuda")


def assert_close(actual, expected):
    import torch

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-2, atol=1e-3)
    return {"ok": True}


def compile_cuda_binary(source, cfg):
    import os
    from pathlib import Path

    import tvm_ffi.cpp

    arch = cfg["arch"].removeprefix("sm_")
    suffix = "a" if arch.endswith("a") else ""
    digits = arch.removesuffix("a")
    key = "TVM_FFI_CUDA_ARCH_LIST"
    previous = os.environ.get(key)
    os.environ[key] = f"{int(digits[:-1])}.{digits[-1]}{suffix}"
    try:
        path = tvm_ffi.cpp.build_inline(
            name=f"example_{source.name}",
            cuda_sources=source.source,
            functions=source.name,
            backend="cuda",
            extra_cuda_cflags=cfg.get("extra_cuda_cflags"),
        )
        return Path(path).read_bytes()
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def compile_cuda(source):
    import tempfile
    from pathlib import Path

    import torch
    import tvm_ffi

    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}" + ("a" if major >= 9 else "")
    data = compile_cuda_binary(source, {"arch": arch})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "kernel.so"
        path.write_bytes(data)
        module = tvm_ffi.load_module(str(path))
    function = module.get_function(source.name)

    # Keep the defining module alive with its callable.
    def invoke(*args):
        _ = module
        return function(*args)

    return invoke


def compile_cutedsl(kernel, *tensors):
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    return cute.compile(kernel, *(from_dlpack(tensor) for tensor in tensors))


def compile_triton(kernel, *args):
    *operands, cfg = args
    options = dict(cfg)
    grid = tuple(options.pop("grid"))
    kernel.warmup(*operands, grid=grid, **options)
    return lambda *values: kernel[grid](*values, **options)
"""


def check_against_cpu_reference(program: Program, dst, assert_close) -> None:
    module = program.upload(id="reference_module", kind="module", source=CPU_REFERENCE)
    reference = program.get_function(id="reference", module=module, name="expected", cpu_only=True)
    expected = program.run(id="expected", fn=reference, args=[N])
    program.run(id="check", fn=assert_close, args=[dst, expected])


def tirx_program() -> Program:
    program = Program()
    operations = program.upload(id="operations", kind="module", source=OPERATIONS)
    empty = program.get_function(id="empty", module=operations, name="empty")
    assert_close = program.get_function(id="assert_close", module=operations, name="assert_close")
    benchmark = program.get_function(id="benchmark", module=operations, name="benchmark")
    compile_tirx = program.get_function(id="compile_tirx", module=operations, name="compile_tirx")
    module = program.upload(id="kernel_module", kind="module", source=TIRX_KERNEL)
    kernel = program.get_function(id="kernel", module=module, name="main")
    src = program.upload(id="src", kind="tensor", value=np.arange(N, dtype=np.float32))
    dst = program.run(id="dst", fn=empty, args=[{"shape": [N], "dtype": "float32"}])

    # `bindings` supplies the T.constexpr values the @T.jit kernel specializes on.
    compiled = program.run(id="compiled", fn=compile_tirx, args=[kernel, {"N": N}])
    program.run(id="invoke", fn=compiled, args=[src, dst])
    check_against_cpu_reference(program, dst, assert_close)
    timing = program.run(
        id="timing",
        fn=benchmark,
        args=[compiled, src, dst, {"warmup_ms": 25, "repeat_ms": 100}],
    )
    program.return_(key="timing", value=timing)
    program.return_(key="dst", value=dst)
    return program


def cutedsl_program() -> Program:
    program = Program()
    operations = program.upload(id="operations", kind="module", source=OPERATIONS)
    empty = program.get_function(id="empty", module=operations, name="empty")
    assert_close = program.get_function(id="assert_close", module=operations, name="assert_close")
    benchmark = program.get_function(id="benchmark", module=operations, name="benchmark")
    compile_cutedsl = program.get_function(
        id="compile_cutedsl", module=operations, name="compile_cutedsl"
    )
    module = program.upload(id="kernel_module", kind="module", source=CUTEDSL_KERNEL)
    kernel = program.get_function(id="kernel", module=module, name="add_one")
    src = program.upload(id="src", kind="tensor", value=np.arange(N, dtype=np.float32))
    dst = program.run(id="dst", fn=empty, args=[{"shape": [N], "dtype": "float32"}])

    # CuTeDSL specializes on the tensors, so compiling takes them too; what comes
    # back is called with the same plain ones.
    compiled = program.run(id="compiled", fn=compile_cutedsl, args=[kernel, src, dst])
    program.run(id="invoke", fn=compiled, args=[src, dst])
    check_against_cpu_reference(program, dst, assert_close)
    timing = program.run(
        id="timing",
        fn=benchmark,
        args=[compiled, src, dst, {"warmup_ms": 25, "repeat_ms": 100}],
    )
    program.return_(key="timing", value=timing)
    program.return_(key="dst", value=dst)
    return program


def cuda_program() -> Program:
    program = Program()
    operations = program.upload(id="operations", kind="module", source=OPERATIONS)
    empty = program.get_function(id="empty", module=operations, name="empty")
    assert_close = program.get_function(id="assert_close", module=operations, name="assert_close")
    benchmark = program.get_function(id="benchmark", module=operations, name="benchmark")
    compile_cuda = program.get_function(id="compile_cuda", module=operations, name="compile_cuda")
    # `language` makes this a CUDA C source module; selecting a function from it
    # creates the source descriptor consumed by compile_cuda in OPERATIONS.
    module = program.upload(id="kernel_module", kind="module", source=CUDA_KERNEL, language="cuda")
    kernel = program.get_function(id="kernel", module=module, name="add_one")
    src = program.upload(id="src", kind="tensor", value=np.arange(N, dtype=np.float32))
    dst = program.run(id="dst", fn=empty, args=[{"shape": [N], "dtype": "float32"}])

    # Built for the worker GPU's arch, and cached on disk by source and flags, so
    # recompiling the same source is much cheaper.
    compiled = program.run(id="compiled", fn=compile_cuda, args=[kernel])
    program.run(id="invoke", fn=compiled, args=[src, dst])
    check_against_cpu_reference(program, dst, assert_close)
    timing = program.run(
        id="timing",
        fn=benchmark,
        args=[compiled, src, dst, {"warmup_ms": 25, "repeat_ms": 100}],
    )
    program.return_(key="timing", value=timing)
    program.return_(key="dst", value=dst)
    return program


def triton_program() -> Program:
    program = Program()
    operations = program.upload(id="operations", kind="module", source=OPERATIONS)
    empty = program.get_function(id="empty", module=operations, name="empty")
    assert_close = program.get_function(id="assert_close", module=operations, name="assert_close")
    benchmark = program.get_function(id="benchmark", module=operations, name="benchmark")
    compile_triton = program.get_function(
        id="compile_triton", module=operations, name="compile_triton"
    )
    module = program.upload(id="kernel_module", kind="module", source=TRITON_KERNEL)
    kernel = program.get_function(id="kernel", module=module, name="add_one")
    src = program.upload(id="src", kind="tensor", value=np.arange(N, dtype=np.float32))
    dst = program.run(id="dst", fn=empty, args=[{"shape": [N], "dtype": "float32"}])

    # A Triton kernel computes its grid at launch, so the grid travels as data
    # rather than as a launcher the client writes. Every other `cfg` key is a
    # launch keyword — num_warps, num_stages, a constexpr by name.
    compiled = program.run(
        id="compiled",
        fn=compile_triton,
        args=[kernel, src, dst, N, 256, {"grid": [1], "num_warps": 4}],
    )
    program.run(id="invoke", fn=compiled, args=[src, dst, N, 256])
    check_against_cpu_reference(program, dst, assert_close)
    timing = program.run(
        id="timing",
        fn=benchmark,
        args=[compiled, src, dst, N, 256, {"warmup_ms": 25, "repeat_ms": 100}],
    )
    program.return_(key="timing", value=timing)
    program.return_(key="dst", value=dst)
    return program


def main() -> None:
    expected = np.arange(N, dtype=np.float32) + 1.0
    with Client(os.environ.get("KCORAL_URL", "http://localhost:8000")) as client:
        programs = (
            ("TIRx", tirx_program()),
            ("CuTeDSL", cutedsl_program()),
            ("CUDA C", cuda_program()),
            ("Triton", triton_program()),
        )
        for language, program in programs:
            result = client.execute(program, timeout_seconds=120)
            if result.status != "COMPLETED":
                print(f"{language}: {result.status} — {result.error}")
                continue
            # Only lease_held_ms occupied the GPU; the CPU reference ran off it.
            print(
                f"{language}: {result.elapsed_ms:.0f} ms total, "
                f"{result.lease_held_ms:.0f} ms on the GPU, "
                f"{result.elapsed_ms - result.lease_held_ms:.0f} ms off it"
            )
            print(f"  kernel {result.results['timing']['latency_ms_median'] * 1e3:.1f} us")
            np.testing.assert_allclose(result.results["dst"], expected)


if __name__ == "__main__":
    main()
