import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=5),
    ],
    key=['M'],
)
@triton.jit
def _gemm_kernel_persistent(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute total number of output tiles in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles_in_group = GROUP_SIZE_M * num_pid_n
    
    pid = tl.program_id(0)
    
    # Group-based scheduling: process tiles in groups of GROUP_SIZE_M along M
    group_id = pid // (num_tiles_in_group // num_pid_n if num_pid_n > 0 else 1)
    pid_in_group = pid % num_pid_n
    
    # Determine which set of M pids this group covers
    start_pid_m = group_id * GROUP_SIZE_M
    end_pid_m = min(start_pid_m + GROUP_SIZE_M, num_pid_m)
    
    # Iterate over all valid M pids within this group
    for pid_m_offset in range(end_pid_m - start_pid_m):
        pid_m = start_pid_m + pid_m_offset
        pid_n = pid_in_group
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

            # Load A tile [BLOCK_M, BLOCK_K] from A [M, K]
            a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # B is [N, K]. We want B^T so we load [BLOCK_K, BLOCK_N].
            b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)

            acc = tl.dot(a, b, acc)

        out_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def run(A, B, C):
    """Compute C = A @ B.T with A:[M,K], B:[N,K], C:[M,N], all bfloat16."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N = B.shape[0]

    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    # Rough estimate of total tiles; grid size capped at SM count
    # Use typical block sizes for estimation
    num_tiles_est = triton.cdiv(M, 128) * triton.cdiv(N, 128)
    grid_size = min(num_sms, max(1, num_tiles_est))

    grid = (grid_size,)

    _gemm_kernel_persistent[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        GROUP_SIZE_M=8,
    )