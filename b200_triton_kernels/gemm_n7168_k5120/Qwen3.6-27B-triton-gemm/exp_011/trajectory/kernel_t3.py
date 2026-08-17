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
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Optimized tiled GEMM with BF16 inputs, FP32 accumulator."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Precompute mask tiles (static shapes after broadcast)
    m_mask = (offs_m < M)[None, :]       # [1, BLOCK_M]
    n_mask = (offs_n < N)[:, None]       # [BLOCK_N, 1]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_k in range(0, K, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K], load A[m, k]
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=m_mask & (offs_k < K)[None, :], other=0.0)

        # B tile: B stored as [N, K], we need B^T effectively as [K, N]
        # B[n, k] loaded as [BLOCK_N, BLOCK_K] then transposed
        b_ptrs = B + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
        b = tl.load(b_ptrs, mask=n_mask & (offs_k < K)[None, :], other=0.0)

        # a: [BLOCK_M, BLOCK_K], b: [BLOCK_N, BLOCK_K]
        # b.T: [BLOCK_K, BLOCK_N] => dot gives [BLOCK_M, BLOCK_N]
        acc = tl.dot(a, b.T, acc)

    # Store result
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=m_mask & n_mask)


def run(A, B, C):
    """Destination-passing: compute C = A @ B.T into preallocated bf16 buffer."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    # Tuned tile sizes for Hopper BF16 WGMMA
    #   BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m, num_pid_n)

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