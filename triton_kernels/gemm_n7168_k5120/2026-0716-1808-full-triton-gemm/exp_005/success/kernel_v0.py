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
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(
        tile_id, num_pid_m, num_pid_n, GROUP_M
    )
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    offs_n = offset_n + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_iter in range(tl.cdiv(K, BLOCK_K)):
        k_step = k_iter * BLOCK_K + tl.arange(0, BLOCK_K)
        
        mask_a = (offs_m[:, None] < M) & (k_step[None, :] < K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + k_step[None, :] * stride_ak, 
                     mask=mask_a, other=0.0)
        
        mask_b = (offs_n[:, None] < N) & (k_step[None, :] < K)
        b = tl.load(b_ptr + offs_n[:, None] * stride_bn + k_step[None, :] * stride_bk, 
                     mask=mask_b, other=0.0)
        
        acc = tl.dot(a, b.T, acc)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_val = acc.to(tl.bfloat16)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c_val, mask=mask)


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    if A.numel() == 0:
        return

    M, K = A.shape
    N, K_b = B.shape
    assert K == K_b, f"K dimension mismatch: {K} vs {K_b}"
    
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        stride_am=K, stride_ak=1,
        stride_bn=K, stride_bk=1,
        stride_cm=N, stride_cn=1,
        BLOCK_M=128,
        BLOCK_N=128,
        BLOCK_K=128,
        GROUP_M=4,
        num_warps=4,
        num_stages=2,
    )