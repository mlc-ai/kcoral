import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def gemm_kernel(A_desc, B_desc, C_desc, M, N, K, 
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, 
                NUM_K_TILES: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
        num_tiles_per_group = GROUP_M * num_pid_n
        group_id = tile_id // num_tiles_per_group
        first_m = group_id * GROUP_M
        m_range = min(num_pid_m - first_m, GROUP_M)
        
        idx = tile_id % num_tiles_per_group
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


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(C.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    assert K == 5120
    assert N == 7168
    
    A_desc = TensorDescriptor.from_tensor(A, [256, 128])
    B_desc = TensorDescriptor.from_tensor(B, [256, 128])
    C_desc = TensorDescriptor.from_tensor(C, [256, 256])
    
    NUM_SMS = 132
    
    num_m_tiles = triton.cdiv(M, 256)
    num_n_tiles = triton.cdiv(N, 256)
    num_programs = min(num_m_tiles * num_n_tiles, NUM_SMS)
    
    grid = (num_programs,)
    gemm_kernel[grid](
        A_desc, B_desc, C_desc,
        M, N, K,
        BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, 
        NUM_K_TILES=40, GROUP_M=4, NUM_SMS=NUM_SMS,
        num_warps=8, num_stages=1
    )