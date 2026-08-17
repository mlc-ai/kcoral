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
    """Compute C += A @ B.T for a single [BLOCK_M, BLOCK_N] tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)

    for ki in range(num_k_iters):
        rk = ki * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        off_a = rm[:, None] * stride_am + rk[None, :] * stride_ak
        mask_a = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(A + off_a, mask=mask_a, other=0.0)

        # Load B tile: [BLOCK_N, BLOCK_K] (physical [N, K])
        off_b = rn[:, None] * stride_bn + rk[None, :] * stride_bk
        mask_b = (rn[:, None] < N) & (rk[None, :] < K)
        b = tl.load(B + off_b, mask=mask_b, other=0.0)

        acc = tl.dot(a, b.T, acc=acc)

    # Store result to C with masking
    off_c = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C + off_c, acc.to(tl.bfloat16), mask=mask_c)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _autotuned_gemm_kernel(
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)

    for ki in range(num_k_iters):
        rk = ki * BLOCK_K + tl.arange(0, BLOCK_K)

        off_a = rm[:, None] * stride_am + rk[None, :] * stride_ak
        mask_a = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(A + off_a, mask=mask_a, other=0.0)

        off_b = rn[:, None] * stride_bn + rk[None, :] * stride_bk
        mask_b = (rn[:, None] < N) & (rk[None, :] < K)
        b = tl.load(B + off_b, mask=mask_b, other=0.0)

        acc = tl.dot(a, b.T, acc=acc)

    off_c = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C + off_c, acc.to(tl.bfloat16), mask=mask_c)


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output tensor."""
    torch.cuda.set_device(A.device)

    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]

    assert A.shape == (M, K)
    assert B.shape == (N, K)
    assert C.shape == (M, N)

    grid = (
        triton.cdiv(M, 256),
        triton.cdiv(N, 128),
    )

    _autotuned_gemm_kernel[grid](
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
    )