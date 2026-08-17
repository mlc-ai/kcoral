import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.heuristics(values={"NUM_SMS": lambda args: 132})
@triton.jit
def _persistent_gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_tile = tl.program_id(0)
    
    num_pid_m = M // BLOCK_M
    num_pid_n = N // BLOCK_N
    num_tiles = num_pid_m * num_pid_n
    num_k_steps = K // BLOCK_K
    
    for tile_id in range(start_tile, num_tiles, NUM_SMS):
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        
        pid_in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_size_m)
        pid_n = pid_in_group // group_size_m
        
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_step in range(num_k_steps):
            offset_k = k_step * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a, b.T, acc)
        
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into the preallocated output tensor ``C``."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    block_m = 128
    block_n = 128
    block_k = 128
    
    a_desc = TensorDescriptor.from_tensor(A, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(B, [block_n, block_k])
    c_desc = TensorDescriptor.from_tensor(C, [block_m, block_n])
    
    num_pid_m = M // block_m
    num_pid_n = N // block_n
    num_tiles = num_pid_m * num_pid_n
    
    grid = (min(132, num_tiles),)
    
    _persistent_gemm_kernel[grid](
        a_desc, b_desc, c_desc, M, N, K,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=8, num_stages=3
    )