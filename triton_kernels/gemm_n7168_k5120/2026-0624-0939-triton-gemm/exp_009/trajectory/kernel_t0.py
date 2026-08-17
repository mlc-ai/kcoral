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
    stride_bk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Standard 2D-grid tiled GEMM. Computes C = A @ B.T
    A: [M, K], B: [N, K], C: [M, N]
    Each program handles one (M, N) output tile.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_step in range(0, tl.cdiv(K, BLOCK_K)):
        k = k_step * BLOCK_K + tl.arange(0, BLOCK_K)

        a_ptr = A + offs_m[:, None] * stride_am + k[None, :] * 1
        a_tile = tl.load(
            a_ptr, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0
        )

        b_ptr = B + offs_n[:, None] * stride_bk + k[None, :] * 1
        b_tile = tl.load(
            b_ptr, mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0.0
        )

        acc = tl.dot(a_tile, b_tile.T, acc)

    out_ptr = C + offs_m[:, None] * stride_cm + offs_n[None, :] * 1
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr, acc.to(tl.bfloat16), mask=mask)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    stride_am = A.stride(0)
    stride_bk = B.stride(0)
    stride_cm = C.stride(0)
    
    block_m = 64
    block_n = 128
    block_k = 64
    
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    
    _gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        stride_am,
        stride_bk,
        stride_cm,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )