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
    GROUP_M: tl.constexpr,
):
    # --- Grouped 1-D to 2-D tile mapping for L2 locality ---
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    # --- Index tensors ---
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # --- Accumulator in fp32 ---
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # --- K-loop (reduction) ---
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_idx = k_start * BLOCK_K + offs_k

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Load B tile: B is physically [N, K].
        # We want [BLOCK_K, BLOCK_N] so tl.dot(A_tile, B_tile) sums over K.
        b_ptrs = B + k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_b = (k_idx[:, None] < K) & (offs_n[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc = tl.dot(a_tile, b_tile, acc)

    # --- Store result ---
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_c)


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output C.

    Args:
        A: [M, K] bfloat16 input
        B: [N, K] bfloat16 input
        C: [M, N] bfloat16 output (preallocated)
    """
    torch.cuda.set_device(A.device)

    M = A.shape[0]
    N = B.shape[0]   # 7168
    K = B.shape[1]   # 5120

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_M = 8

    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
    )

    _gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),          # stride_am
        A.stride(1),          # stride_ak
        B.stride(1),          # stride_bk  (K-dim of B)
        B.stride(0),          # stride_bn  (N-dim of B)
        C.stride(0),          # stride_cm
        C.stride(1),          # stride_cn
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=4,
        num_stages=3,
    )