import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_AM,
    stride_AK,
    stride_BN,
    stride_BK,
    stride_CM,
    stride_CN,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute C[pid_m, pid_n] = A[pid_m] @ B[pid_n]^T via tiled reduction over K."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_K)
    for start_k in range(0, num_k_blocks):
        offs_k = start_k * BLOCK_K + tl.arange(0, BLOCK_K)

        a_ptrs = A + offs_m[:, None] * stride_AM + offs_k[None, :] * stride_AK
        b_ptrs = B + offs_n[:, None] * stride_BN + offs_k[None, :] * stride_BK

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K),
            other=0.0,
        )

        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_N, BLOCK_K]
        # b.T: [BLOCK_K, BLOCK_N]
        # result: [BLOCK_M, BLOCK_N]
        acc = tl.dot(a, b.T, acc)

    c_ptrs = C + offs_m[:, None] * stride_CM + offs_n[None, :] * stride_CN
    tl.store(
        c_ptrs,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def run(A, B, C):
    """Destination-passing entry: compute C = A @ B.T into preallocated C (bf16)."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape

    # Tile configuration tuned for Hopper BF16 WGMMA:
    #   BLOCK_K=64 matches WGMMA instruction granularity.
    #   BLOCK_M=128, BLOCK_N=128 gives a balanced 2-D grid.
    #   num_warps=4 provides sufficient thread count for latency hiding.
    #   num_stages=4 pipelines async loads through shared memory.
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=4,
    )