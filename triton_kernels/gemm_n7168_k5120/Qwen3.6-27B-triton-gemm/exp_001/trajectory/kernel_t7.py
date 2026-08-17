import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Large accumulators with good occupancy
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Precompute masks for outer dims
    mask_m = rm[:, None] < M
    mask_n = rn[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_range = tl.cdiv(K, BLOCK_K)

    for ki in range(k_range):
        rk = ki * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k_a = rk[None, :] < K
        mask_k_b = rk[None, :] < K

        # Load A: [BLOCK_M, BLOCK_K]
        off_a = rm[:, None] * stride_am + rk[None, :] * stride_ak
        a = tl.load(A + off_a, mask=mask_m & mask_k_a, other=0.0)

        # Load B: physical [N,K], tile [BLOCK_N, BLOCK_K]
        off_b = rn[:, None] * stride_bn + rk[None, :] * stride_bk
        b = tl.load(B + off_b, mask=(rn[:, None] < N) & mask_k_b, other=0.0)

        acc = tl.dot(a, b.T, acc=acc)

    # Store output
    off_c = rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C + off_c, acc.to(tl.bfloat16), mask=mask_m & mask_n)


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output tensor."""
    torch.cuda.set_device(A.device)

    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]

    assert A.shape == (M, K)
    assert B.shape == (N, K)
    assert C.shape == (M, N)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
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
    )