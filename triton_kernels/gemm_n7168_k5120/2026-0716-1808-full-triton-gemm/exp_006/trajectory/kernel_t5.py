import torch
import triton
import triton.language as tl
from __future__ import annotations


@triton.jit
def _grouped_tile_coordinates(
    tile_id,
    num_pid_m,
    num_pid_n,
    GROUP_M: tl.constexpr,
):
    """Convert a flat program ID into (pid_m, pid_n) using grouped ordering."""
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = tile_id % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Computes C = A @ B.T using standard pointer loads and Tensor Core math.
    A has shape [M, K], B has shape [N, K], C has shape [M, N].
    """
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = N // BLOCK_N
    
    pid_m, pid_n = _grouped_tile_coordinates(tile_id, num_pid_m, num_pid_n, GROUP_M)
    
    m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_cm = (m_idx[:, None] < M) & (n_idx[None, :] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    for k_tile in range(K // BLOCK_K):
        k_idx = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        
        mask_a = (m_idx[:, None] < M) & (k_idx[None, :] < K)
        a = tl.load(A_ptr + m_idx[:, None] * stride_am + k_idx[None, :] * stride_ak,
                    mask=mask_a, other=0.0)
        
        mask_b = (n_idx[:, None] < N) & (k_idx[None, :] < K)
        b = tl.load(B_ptr + n_idx[:, None] * stride_bn + k_idx[None, :] * stride_bk,
                    mask=mask_b, other=0.0)
        
        acc = tl.dot(a, b.T, acc)
    
    C_ptr = C_ptr + m_idx[:, None] * stride_cm + n_idx[None, :] * stride_cn
    tl.store(C_ptr, acc.to(tl.bfloat16), mask=mask_cm)


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bn, stride_bk = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)
    
    block_m = 256
    block_n = 512
    block_k = 1024
    
    num_pid_m = triton.cdiv(M, block_m)
    num_pid_n = N // block_n
    
    grid = (num_pid_m * num_pid_n,)
    
    gemm_kernel[grid](
        A, B, C,
        M=M, N=N, K=K,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_bn=stride_bn, stride_bk=stride_bk,
        stride_cm=stride_cm, stride_cn=stride_cn,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        GROUP_M=4,
        num_warps=8,
        num_stages=3,
    )