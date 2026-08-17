import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def gemm_kernel(A_desc, B_desc, C_desc, M, N, K, 
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, 
                NUM_K_TILES: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    
    start_pid = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = num_m_tiles * num_n_tiles
    
    tile_stride = num_tiles // math.gcd(num_tiles, NUM_SMS)
    
    iterations = 0
    for current_tile in range(start_pid, num_tiles, tile_stride):
        
        num_tiles_per_group = GROUP_M * num_n_tiles
        group_id = current_tile // num_tiles_per_group
        first_m = group_id * GROUP_M
        m_range = min(num_m_tiles - first_m, GROUP_M)
        
        idx = current_tile % num_tiles_per_group
        m_tile = first_m + (idx % m_range)
        n_tile = idx // m_range
        
        offset_m = m_tile * BLOCK_M
        offset_n = n_tile * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in range(NUM_K_TILES):
            offset_k = k_tile * BLOCK_K
            a = A_desc.load([offset_m, offset_k])
            b = B_desc.load([offset_n, offset_k])
            acc = tl.dot(a, b.T, acc)
            
        C_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))
        
        iterations += 1


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(C.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    assert K == 5120
    assert N == 7168
    
    A_desc = TensorDescriptor.from_tensor(A, [128, 256])
    B_desc = TensorDescriptor.from_tensor(B, [512, 256])
    C_desc = TensorDescriptor.from_tensor(C, [128, 512])
    
    NUM_SMS = 132
    
    grid = (NUM_SMS,)
    gemm_kernel[grid](
        A_desc, B_desc, C_desc,
        M, N, K,
        BLOCK_M=128, BLOCK_N=512, BLOCK_K=256, 
        NUM_K_TILES=20, GROUP_M=4, NUM_SMS=NUM_SMS,
        num_warps=8, num_stages=3, maxnreg=128
    )