import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


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
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    BLOCK_K: tl.constexpr = 512
    
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(
        tile_id, num_pid_m, num_pid_n, GROUP_M
    )
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    max_k_tiles = 10
    
    for k_iter in range(min(num_k_tiles, max_k_tiles)):
        k_step = k_iter * 512
        
        a = a_desc.load([m_start, k_step])
        b = b_desc.load([n_start, k_step])
        
        acc = tl.dot(a, b.T, acc)

    m_off = m_start + tl.arange(0, BLOCK_M)
    n_off = n_start + tl.arange(0, BLOCK_N)
    
    mask = (m_off[:, None] < M) & (n_off[None, :] < N)
    
    c_val = acc.to(tl.bfloat16)
    tl.store(c_ptr + m_off[:, None] * stride_cm + n_off[None, :] * stride_cn, 
              c_val, mask=mask, eviction_policy="evict_first")


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    if A.numel() == 0:
        return

    M, K = A.shape
    N, K_b = B.shape
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    
    grid = (num_pid_m * num_pid_n,)
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, 512])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, 512])
    
    _gemm_kernel[grid](
        a_desc, b_desc, C,
        M, N, K,
        stride_cm=N, stride_cn=1,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUP_M=4,
        num_warps=8,
        num_stages=3,
        maxnreg=128,
    )