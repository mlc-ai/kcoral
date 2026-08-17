import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
    ],
    key=["M"],
)
@triton.jit
def _gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/column offsets for this program instance
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Boundary masks
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    # Accumulator in FP32 for numerical precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over the K reduction dimension
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        # A shape is [M, K], stride_am along rows, stride_ak along columns
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=m_mask & k_mask[None, :], other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N] representing a tile of B.T
        # B physical shape is [N, K], so B.T[k, n] = B[n, k]
        # Broadcasting offs_k[:, None] * stride_bk with offs_n[None, :] * stride_bn
        # yields shape [BLOCK_K, BLOCK_N] where element (ki, ni) is B[ni, ki] = B.T[ki, ni]
        b_ptrs = B + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk
        b = tl.load(b_ptrs, mask=n_mask & k_mask[:, None], other=0.0)

        # Matrix multiply accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc = tl.dot(a, b, acc)

    # Convert accumulator to bfloat16 and store into C
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=m_mask & n_mask)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C.
    
    Args:
        A: Input tensor of shape [M, 5120], dtype bfloat16
        B: Input tensor of shape [7168, 5120], dtype bfloat16
        C: Preallocated output tensor of shape [M, 7168], dtype bfloat16
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]  # 7168
    K = A.shape[1]  # 5120

    # Lambda grid adapts to whatever block sizes autotune selects
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )