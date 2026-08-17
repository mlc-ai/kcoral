import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_tile_coordinates(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_M: tl.constexpr,
):
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(
        tile_id, num_pid_m, num_pid_n, GROUP_M
    )
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_off = m_start + tl.arange(0, BLOCK_M)
    n_off = n_start + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for phase in range(2):
        base_k = phase * 2560
        for k in range(2):
            k_start = base_k + k * 1280
            k_off = k_start + tl.arange(0, 1280)
            
            mask_a = (m_off[:, None] < M) & (k_off[None, :] < K)
            a = tl.load(a_ptr + m_off[:, None] * 1280 + k_off[None, :] * 1, mask=mask_a, other=0.0)
            
            mask_b = (n_off[:, None] < N) & (k_off[None, :] < K)
            b = tl.load(b_ptr + k_off[None, :] * 1 + n_off[:, None] * 5120, mask=mask_b, other=0.0)
            
            acc = tl.dot(a, b, acc)

    mask_c = (m_off[:, None] < M) & (n_off[None, :] < N)
    c_val = acc.to(tl.bfloat16)
    tl.store(c_ptr + m_off[:, None] * stride_cm + n_off[None, :] * stride_cn, c_val, mask=mask_c, eviction_policy="evict_first")


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    if A.numel() == 0:
        return

    M, K = A.shape
    N, K_b = B.shape
    
    grid = (triton.cdiv(M, 256) * triton.cdiv(N, 1024),)
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        stride_cm=N, stride_cn=1,
        BLOCK_M=256,
        BLOCK_N=1024,
        BLOCK_K=1280,
        GROUP_M=4,
        num_warps=4,
        num_stages=2,
        maxnreg=128,
    )