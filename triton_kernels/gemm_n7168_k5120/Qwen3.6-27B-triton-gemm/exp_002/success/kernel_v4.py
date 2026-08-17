import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Standard tiled configs with good Hopper WGMMA shapes
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=4),
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
    """Blocked GEMM: C = A @ B.T where B is stored in [N, K] layout."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k in range(0, num_k_iters):
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k_a = k_offs[None, :] < K
        mask_k_b = k_offs[:, None] < K

        # A tile: shape [BLOCK_M, BLOCK_K], a[m,k] = A[m,k]
        a_ptrs = A + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_tile = tl.load(
            a_ptrs,
            mask=mask_m & mask_k_a,
            other=0.0,
        )

        # B tile: load [BLOCK_N, BLOCK_K] from B[N,K], then transpose for dot
        b_ptrs = B + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_tile = tl.load(
            b_ptrs,
            mask=mask_k_b & mask_n,
            other=0.0,
        )

        acc = tl.dot(a_tile, b_tile, acc)

    # Store accumulated result to C
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(tl.bfloat16),
        mask=mask_m & mask_n,
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

    # Lambda grid adapts to each autotune config's block sizes
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