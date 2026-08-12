"""Compile a kernel on the server, in TIRx, CuTeDSL, CUDA C and Triton.

All four programs have the same shape — upload the kernel as text, compile it
with a builtin, run and time the result — so the only difference is the language
and which `compile_*` builtin reads it. None needs a CUDA toolchain on the
client, and the compile runs off the GPU lease, leaving the card to another
worker.
"""

from __future__ import annotations

import os

import numpy as np

from benchmark_server import Client, Program

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

# The entry is exported through TVM FFI, so it takes TensorView parameters and
# returns void; the includes and the export macro come from the server.
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


def tirx_program() -> Program:
    program = Program()
    kernel = program.upload(id="kernel", kind="module", source=TIRX_KERNEL)
    src = program.upload(id="src", kind="tensor", value=np.arange(N, dtype=np.float32))
    dst = program.run(id="dst", fn="builtin.empty", args=[{"shape": [N], "dtype": "float32"}])

    # `bindings` supplies the T.constexpr values the @T.jit kernel specializes on.
    compiled = program.run(id="compiled", fn="builtin.compile_tirx", args=[kernel, {"N": N}])
    program.run(id="invoke", fn=compiled, args=[src, dst])
    timing = program.run(
        id="timing",
        fn="builtin.benchmark",
        args=[compiled, src, dst, {"warmup_ms": 25, "repeat_ms": 100}],
    )
    program.return_(key="timing", value=timing)
    program.return_(key="dst", value=dst)
    return program


def cutedsl_program() -> Program:
    program = Program()
    # `entry` names the @cute.jit launcher: with the kernel beside it there are
    # two top-level definitions and no `main`, which would be ambiguous.
    kernel = program.upload(id="kernel", kind="module", source=CUTEDSL_KERNEL, entry="add_one")
    src = program.upload(id="src", kind="tensor", value=np.arange(N, dtype=np.float32))
    dst = program.run(id="dst", fn="builtin.empty", args=[{"shape": [N], "dtype": "float32"}])

    # CuTeDSL specializes on the tensors, so compiling takes them too; what comes
    # back is called with the same plain ones.
    compiled = program.run(id="compiled", fn="builtin.compile_cutedsl", args=[kernel, src, dst])
    program.run(id="invoke", fn=compiled, args=[src, dst])
    timing = program.run(
        id="timing",
        fn="builtin.benchmark",
        args=[compiled, src, dst, {"warmup_ms": 25, "repeat_ms": 100}],
    )
    program.return_(key="timing", value=timing)
    program.return_(key="dst", value=dst)
    return program


def cuda_program() -> Program:
    program = Program()
    # `language` makes the source CUDA C, and `entry` is required: nothing is
    # executed, so there is no namespace to infer a name from. C++ reserves `main`.
    kernel = program.upload(
        id="kernel", kind="module", source=CUDA_KERNEL, entry="add_one", language="cuda"
    )
    src = program.upload(id="src", kind="tensor", value=np.arange(N, dtype=np.float32))
    dst = program.run(id="dst", fn="builtin.empty", args=[{"shape": [N], "dtype": "float32"}])

    # Built for the worker GPU's arch, and cached on disk by source and flags, so
    # recompiling the same source is much cheaper. `cfg` takes extra_cuda_cflags.
    compiled = program.run(id="compiled", fn="builtin.compile_cuda", args=[kernel])
    program.run(id="invoke", fn=compiled, args=[src, dst])
    timing = program.run(
        id="timing",
        fn="builtin.benchmark",
        args=[compiled, src, dst, {"warmup_ms": 25, "repeat_ms": 100}],
    )
    program.return_(key="timing", value=timing)
    program.return_(key="dst", value=dst)
    return program


def triton_program() -> Program:
    program = Program()
    kernel = program.upload(id="kernel", kind="module", source=TRITON_KERNEL, entry="add_one")
    src = program.upload(id="src", kind="tensor", value=np.arange(N, dtype=np.float32))
    dst = program.run(id="dst", fn="builtin.empty", args=[{"shape": [N], "dtype": "float32"}])

    # A Triton kernel computes its grid at launch, so the grid travels as data
    # rather than as a launcher the client writes. Every other `cfg` key is a
    # launch keyword — num_warps, num_stages, a constexpr by name.
    compiled = program.run(
        id="compiled",
        fn="builtin.compile_triton",
        args=[kernel, src, dst, N, 256, {"grid": [1], "num_warps": 4}],
    )
    program.run(id="invoke", fn=compiled, args=[src, dst, N, 256])
    timing = program.run(
        id="timing",
        fn="builtin.benchmark",
        args=[compiled, src, dst, N, 256, {"warmup_ms": 25, "repeat_ms": 100}],
    )
    program.return_(key="timing", value=timing)
    program.return_(key="dst", value=dst)
    return program


def main() -> None:
    expected = np.arange(N, dtype=np.float32) + 1.0
    with Client(os.environ.get("BENCH_URL", "http://localhost:8000")) as client:
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
            # Only lease_held_ms occupied the GPU; the rest compiled off it.
            print(
                f"{language}: {result.elapsed_ms:.0f} ms total, "
                f"{result.lease_held_ms:.0f} ms on the GPU, "
                f"{result.elapsed_ms - result.lease_held_ms:.0f} ms off it"
            )
            print(f"  kernel {result.results['timing']['latency_ms_median'] * 1e3:.1f} us")
            np.testing.assert_allclose(result.results["dst"], expected)


if __name__ == "__main__":
    main()
