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
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute C += A @ B.T for a single [BLOCK_M, BLOCK_N] tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Accumulator in FP32 for numerical quality
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Precompute row/column program offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    num_stage_k = tl.cdiv(K, BLOCK_K)

    for stage_k in range(num_stage_k):
        rk = stage_k * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load tile from A: [BLOCK_M, BLOCK_K]
        off_a = rm[:, None] * stride_am + rk[None, :] * stride_ak
        mask_a = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(A + off_a, mask=mask_a, other=0.0)

        # Load tile from B: [BLOCK_N, BLOCK_K] physically stored as [N, K]
        off_b = rk[:, None] * stride_bk + rn[None, :] * stride_bn
        mask_b = (rk[:, None] < K) & (rn[None, :] < N)
        b = tl.load(B + off_b, mask=mask_b, other=0.0)

        # Dot: A[BLOCK_M, BLOCK_K] @ B.T[BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc = tl.dot(a, b, acc=acc)

    # Store result to C with proper masking for partial-M tiles
    off_c = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C + off_c, acc.to(C.dtype.element_ty), mask=mask_c)


def run(A, B, C):
    """Compute C = A @ B.T into a preallocated output tensor.

    Args:
        A: [M, K] bfloat16 input matrix
        B: [N, K] bfloat16 input matrix
        C: [M, N] bfloat16 preallocated output matrix
    """
    torch.cuda.set_device(A.device)

    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]

    BLOCK_M = 128
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
        B.stride(1),
        B.stride(0),
        C.stride(0),
        C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )