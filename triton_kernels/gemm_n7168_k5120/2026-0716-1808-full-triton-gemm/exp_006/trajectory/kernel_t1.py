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
    a_desc,
    b_desc,
    c_desc,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Computes C = A @ B.T using TMA descriptors.
    A has shape [M, K], B has shape [N, K], C has shape [M, N].
    """
    tile_id = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m, pid_n = _grouped_tile_coordinates(
        tile_id,
        num_pid_m,
        num_pid_n,
        GROUP_M,
    )

    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # Unroll over the constant K dimension (K=5120, BLOCK_K=1024 -> 5 steps)
    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        offset_k = k_tile * BLOCK_K
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        acc = tl.dot(a, b.T, acc, out_dtype=tl.float32)

    m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_cm = (m_idx[:, None] < M) & (n_idx[None, :] < N)
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16), mask=mask_cm)


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = 7168
    K = 5120
    
    block_m = 128
    block_n = 256
    block_k = 1024
    
    a_desc = TensorDescriptor.from_tensor(A, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(B, [block_n, block_k])
    c_desc = TensorDescriptor.from_tensor(C, [block_m, block_n])
    
    num_pid_m = triton.cdiv(M, block_m)
    num_pid_n = triton.cdiv(N, block_n)
    
    grid = (num_pid_m * num_pid_n,)
    
    # Launch with optimal configurations for Hopper architecture
    gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M=M, 
        N=N, 
        K=K,
        BLOCK_M=block_m, 
        BLOCK_N=block_n, 
        BLOCK_K=block_k,
        GROUP_M=132,
        num_warps=8,
        num_stages=3,
    )