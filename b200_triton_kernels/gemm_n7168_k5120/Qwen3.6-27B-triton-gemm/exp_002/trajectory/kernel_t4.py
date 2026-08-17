import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Grouped-M approach with large blocks for Hopper WGMMA
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=4),
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
    GROUP_M: tl.constexpr,
):
    """Blocked GEMM with grouped-M scheduling for better L2 reuse."""
    # Compute how many M-groups exist
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid = tl.program_id(0)
    num_pids = num_pid_m * num_pid_n
    
    # Grouped schedule: walk through M-dimension in groups
    g_id = pid // (num_pid_n * GROUP_M)
    remaining_pids = pid % (num_pid_n * GROUP_M)
    
    m_start = g_id * GROUP_M
    if m_start + GROUP_M < num_pid_m:
        m_end = m_start + GROUP_M
    else:
        m_end = num_pid_m
    
    pid_m = m_start + remaining_pids % (m_end - m_start)
    pid_n = remaining_pids // (m_end - m_start)
    
    # Bounds check
    if pid >= num_pids:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k in range(0, num_k_iters):
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_tile = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        )

        # B tile: load B[n,k] for n in our range, k in current block
        b_ptrs = B + offs_n[None, :] * stride_bn + k_offs[:, None] * stride_bk
        b_tile = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & (k_offs[:, None] < K),
            other=0.0,
        )

        acc = tl.dot(a_tile, b_tile, acc)

    # Store result
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

    total_pids = triton.cdiv(M, 128) * triton.cdiv(N, 128)
    grid = lambda META: (total_pids,)

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