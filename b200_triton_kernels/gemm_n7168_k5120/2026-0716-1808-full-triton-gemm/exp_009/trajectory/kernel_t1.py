import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    A_desc,
    B_desc,
    C_desc,
    M,
    N_dim,
    K_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Use a 1-D grid to map programs to tiles sequentially, enabling software pipelining 
    # and improving L2 locality by allowing each warp to process adjacent tiles.
    start_tile = tl.program_id(0)
    stride = tl.num_programs(0)
    
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N_dim + BLOCK_N - 1) // BLOCK_N
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in tl.range(start_tile, num_tiles, stride):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        
        m_offset = pid_m * BLOCK_M
        n_offset = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        # Using static_range allows the compiler to completely unroll the K dimension loop.
        # Since both K_dim (5120) and BLOCK_K (64) are known at compile time, this generates
        # exactly 80 pipelined load and dot instruction groups with zero runtime branch overhead.
        for k0 in tl.static_range(0, K_dim // BLOCK_K):
            k_offset = k0 * BLOCK_K
            
            a_tile = A_desc.load([m_offset, k_offset])
            b_tile = B_desc.load([n_offset, k_offset])
            
            acc = tl.dot(a_tile, b_tile.T, acc)
        
        C_desc.store([m_offset, n_offset], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N_dim = B.shape[0]
    K_dim = B.shape[1]
    
    block_m = 128
    block_n = 128
    block_k = 64
    
    # Fixed 128-wide grid effectively implements persistent scheduling.
    # Triton's range() abstraction ensures that each program instance processes its assigned tiles sequentially.
    grid = (128,) 
    
    # Utilize TensorDescriptors for Hopper to enable Hardware Accelerated Memory Copy (TMA)
    # under the hood when loading contiguous tiles.
    A_desc = TensorDescriptor.from_tensor(A, [block_m, block_k])
    B_desc = TensorDescriptor.from_tensor(B, [block_n, block_k])
    C_desc = TensorDescriptor.from_tensor(C, [block_m, block_n])
    
    _gemm_kernel[grid](
        A_desc,
        B_desc,
        C_desc,
        M,
        N_dim,
        K_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )