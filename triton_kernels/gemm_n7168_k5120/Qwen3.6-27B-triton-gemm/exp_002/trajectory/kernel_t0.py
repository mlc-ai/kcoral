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
    """Blocked GEMM: C += A @ B.T  where B is stored in [N, K] layout."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k in range(0, num_k_iters):
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # A tile: shape [BLOCK_M, BLOCK_K], a[m,k] = A[m,k]
        a_ptrs = A + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_tile = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        )

        # B tile: shape [BLOCK_K, BLOCK_N], b[k,n] = B[n,k] (= B.T[k,n])
        # B stored [N,K]: B[n,k] = base + n*stride_bn + k*stride_bk
        b_ptrs = B + offs_n[None, :] * stride_bn + k_offs[:, None] * stride_bk
        b_tile = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & (k_offs[:, None] < K),
            other=0.0,
        )

        acc = tl.dot(a_tile, b_tile, acc)

    # Store accumulated result to C
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def run(A, B, C):
    """
    Compute C = A @ B.T into preallocated output C.
    
    A: [M, K] bfloat16
    B: [N, K] bfloat16
    C: [M, N] bfloat16 (output)
    """
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = B.shape[1]

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    _gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )