import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
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
    # Grouped tile ordering for better L2 reuse of B
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    # Row/column offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Base pointers (avoid recomputing per-K iteration)
    a_ptrs_base = a_ptr + offs_m[:, None] * stride_am
    b_ptrs_base = b_ptr + offs_n[:, None] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)

        a_ptrs = a_ptrs_base + offs_k[None, :] * stride_ak
        b_ptrs = b_ptrs_base + offs_k[None, :] * stride_bk

        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # B is [BLOCK_N, BLOCK_K]; .T makes it [BLOCK_K, BLOCK_N]
        # tl.dot([BLOCK_M,BLOCK_K], [BLOCK_K,BLOCK_N]) -> [BLOCK_M,BLOCK_N]
        acc = tl.dot(a, b.T, acc)

    # Store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=mask_c)


def run(A, B, C):
    """Compute C = A @ B.T into preallocated output tensor."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape

    # Tile configuration tuned for Hopper BF16 WGMMA
    # N=7168 divisible by 128, K=5120 divisible by 128 → no partial K/N tiles
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    GROUP_M = 8
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )