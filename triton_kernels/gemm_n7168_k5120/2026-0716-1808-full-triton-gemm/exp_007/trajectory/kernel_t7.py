import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def gemm_kernel(A_desc, B_desc, C_desc, M, N, K, 
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, 
                NUM_K_TILES: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    
    start_pid = tl.program_id(0)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_tiles = num_m_tiles * num_n_tiles
    
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
        num_tiles_per_group = GROUP_M * num_n_tiles
        group_id = tile_id // num_tiles_per_group
        first_m = group_id * GROUP_M
        m_range = min(GROUP_M, num_m_tiles - first_m)
        
        idx = tile_id % num_tiles_per_group
        n_tile = idx // m_range
        m_offset_in_group = idx % m_range
        
        offset_m = (first_m + m_offset_in_group) * BLOCK_M
        offset_n_left = n_tile * BLOCK_N
        offset_n_right = n_tile * BLOCK_N + BLOCK_N
        
        acc_left = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_right = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in range(NUM_K_TILES):
            offset_k = k_tile * BLOCK_K
            
            if k_tile + 1 < NUM_K_TILES:
                next_offset_k = (k_tile + 1) * BLOCK_K
                A_desc.prefetch([offset_m, next_offset_k], cta_rid=0)
                B_desc.prefetch([offset_n_left, next_offset_k], cta_rid=0)
                B_desc.prefetch([offset_n_right, next_offset_k], cta_rid=0)
                
            a = A_desc.load([offset_m, offset_k])
            b_left = B_desc.load([offset_n_left, offset_k])
            b_right = B_desc.load([offset_n_right, offset_k])
            
            acc_left = tl.dot(a, b_left.T, acc_left)
            acc_right = tl.dot(a, b_right.T, acc_right)
            
        C_desc.store([offset_m, offset_n_left], acc_left.to(tl.bfloat16))
        C_desc.store([offset_m, offset_n_right], acc_right.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(C.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    assert K == 5120
    assert N == 7168
    
    A_desc = TensorDescriptor.from_tensor(A, [256, 256])
    B_desc = TensorDescriptor.from_tensor(B, [256, 256])
    C_desc = TensorDescriptor.from_tensor(C, [256, 256])
    
    NUM_SMS = 132
    
    num_m_tiles = triton.cdiv(M, 256)
    num_n_tiles = triton.cdiv(N, 256)
    num_tiles = num_m_tiles * num_n_tiles
    num_programs = min(num_tiles, NUM_SMS)
    
    grid = (num_programs,)
    gemm_kernel[grid](
        A_desc, B_desc, C_desc,
        M, N, K,
        BLOCK_M=256, BLOCK_N=256, BLOCK_K=256, 
        NUM_K_TILES=triton.cdiv(K, 256), GROUP_M=4, NUM_SMS=NUM_SMS,
        num_warps=4, maxnreg=128
    )