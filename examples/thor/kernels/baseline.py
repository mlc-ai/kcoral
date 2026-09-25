"""Baseline fp16 GEMM for the Thor example: ``D = A @ B.T``.

The textbook shared-memory tiled kernel. A 16x16 thread block computes a 16x16
tile of ``D``: for each 16-wide slice of K it stages one tile of ``A`` and one of
``B`` in shared memory, then every thread accumulates its output element in an
fp32 register with scalar FMAs. There are no tensor cores, no register
blocking, no vectorized or asynchronous copies and no pipelining, which leaves
a large and well-understood optimization headroom.

Kernel contract (every candidate implements the same interface):

``build(M, N, K) -> fn``
    Compile a kernel for one problem shape and return ``fn(A, B, D)``.
    ``A`` is ``[M, K]``, ``B`` is ``[N, K]`` and ``D`` is ``[M, N]``; all three are
    contiguous row-major CUDA ``torch.float16`` tensors. Every call of ``fn``
    must read ``A`` and ``B`` and overwrite ``D``.
"""

from __future__ import annotations

import tvm
from tvm.script import tirx as T

BLOCK = 16  # a BLOCK x BLOCK thread block computes a BLOCK x BLOCK tile of D


@T.jit
def gemm_tiled(
    # M, N and K are the constexpr parameters below; TIRx resolves them in specialize().
    A: T.Buffer((M, K), "float16"),  # noqa: F821
    B: T.Buffer((N, K), "float16"),  # noqa: F821
    D: T.Buffer((M, N), "float16"),  # noqa: F821
    *,
    M: T.constexpr,
    N: T.constexpr,
    K: T.constexpr,
):
    T.device_entry()
    bx, by = T.cta_id([N // BLOCK, M // BLOCK])
    tx, ty = T.thread_id([BLOCK, BLOCK])
    A_tile = T.alloc_shared((BLOCK, BLOCK), "float16")
    B_tile = T.alloc_shared((BLOCK, BLOCK), "float16")
    acc = T.alloc_local((1,), "float32")
    acc[0] = T.float32(0)
    for kb in T.serial(K // BLOCK):
        # Each thread stages one element of each tile; consecutive tx read consecutive k.
        A_tile[ty, tx] = A[by * BLOCK + ty, kb * BLOCK + tx]
        B_tile[ty, tx] = B[bx * BLOCK + ty, kb * BLOCK + tx]
        T.cuda.cta_sync()
        for kk in T.serial(BLOCK):
            acc[0] += T.Cast("float32", A_tile[ty, kk]) * T.Cast("float32", B_tile[tx, kk])
        T.cuda.cta_sync()
    D[by * BLOCK + ty, bx * BLOCK + tx] = T.Cast("float16", acc[0])


def cuda_target() -> tvm.target.Target:
    """Target for the server's GPU, e.g. ``sm_110a`` on Thor (arch-specific features enabled)."""
    major, minor = tvm.cuda(0).compute_version.split(".")
    return tvm.target.Target({"kind": "cuda", "arch": f"sm_{major}{minor}a"})


def build(M: int, N: int, K: int):
    assert M % BLOCK == 0 and N % BLOCK == 0 and K % BLOCK == 0, "shape must tile by BLOCK"
    target = cuda_target()
    kernel = gemm_tiled.specialize(M=M, N=N, K=K)
    with target:
        return tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
