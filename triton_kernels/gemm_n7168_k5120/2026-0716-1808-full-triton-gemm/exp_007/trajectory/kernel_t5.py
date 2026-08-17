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
    
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False, warp_specialize=True):
        num_tiles_per_group = GROUP_M * num_n_tiles
        group_id = tile_id // num_tiles_per_group
        first_m = group_id * GROUP_M
        m_range = min(num_m_tiles - first_m, GROUP_M)
        
        idx = tile_id % num_tiles_per_group
        m_tile = first_m + (idx % m_range)
        n_tile = idx // m_range
        
        offset_m = m_tile * BLOCK_M
        offset_n = n_tile * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in tl.range(NUM_K_TILES):
            # pragma "unroll 4"
            
            offset_k = k_tile * BLOCK_K
            a = A_desc.load([offset_m, offset_k])
            b = B_desc.load([offset_n, offset_k])
            acc = tl.dot(a, b.T, acc)
            
        C_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(C.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    assert K == 5120
    assert N == 7168
    
    A_desc = TensorDescriptor.from_tensor(A, [128, 128])
    B_desc = TensorDescriptor.from_tensor(B, [128, 128])
    C_desc = TensorDescriptor.from_tensor(C, [128, 128])
    
    NUM_SMS = 132
    
    num_m_tiles = triton.cdiv(M, 128)
    num_n_tiles = triton.cdiv(N, 128)
    num_programs = min(num_m_tiles * num_n_tiles, NUM_SMS)
    
    grid = (num_programs,)
    gemm_kernel[grid](
        A_desc, B_desc, C_desc,
        M, N, K,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, 
        NUM_K_TILES=triton.cdiv(K, 128), GROUP_M=4, NUM_SMS=NUM_SMS,
        num_warps=8, num_stages=2, maxnreg=128
    )