import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


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
        m_range = min(num_m_tiles - first_m, GROUP_M)
        
        idx = tile_id % num_tiles_per_group
        m_tile = first_m + (idx % m_range)
        n_tile = idx // m_range
        
        offset_m = m_tile * BLOCK_M
        offset_n = n_tile * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_tile in tl.range(NUM_K_TILES, num_stages=1, loop_unroll_factor=4, flatten=True):
            offset_k = k_tile * BLOCK_K
            
            mask_a = mask_2d(A_desc, [offset_m, offset_k], [BLOCK_M, BLOCK_K], M, K)
            a = A_desc.load([offset_m, offset_k], mask=mask_a)
            
            mask_b = mask_2d(B_desc, [offset_n, offset_k], [BLOCK_N, BLOCK_K], N, K)
            b = B_desc.load([offset_n, offset_k], mask=mask_b)
            
            acc = tl.dot(a, b.T, acc)
            
        if M > offset_m and N > offset_n:
            C_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def mask_2d(desc, offset, shape, M, N):
    o0, o1 = offset
    s0, s1 = shape
    m0, m1 = M, N
    if o0 >= m0 or o1 >= m1:
        return desc.get_addr_mask(o0, o1, s0, s1)
    else:
        return desc.get_addr_mask(o0, o1, s0, s1)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(C.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    assert K == 5120
    assert N == 7168
    
    A_desc = TensorDescriptor.from_tensor(A, [256, 256])
    B_desc = TensorDescriptor.from_tensor(B, [512, 256])
    C_desc = TensorDescriptor.from_tensor(C, [256, 512])
    
    NUM_SMS = 132
    
    num_m_tiles = triton.cdiv(M, 256)
    num_n_tiles = triton.cdiv(N, 512)
    num_programs = min(num_m_tiles * num_n_tiles, NUM_SMS)
    
    grid = (num_programs,)
    gemm_kernel[grid](
        A_desc, B_desc, C_desc,
        M, N, K,
        BLOCK_M=256, BLOCK_N=512, BLOCK_K=256, 
        NUM_K_TILES=triton.cdiv(K, 256), GROUP_M=8, NUM_SMS=NUM_SMS,
        num_warps=4, num_stages=1, maxnreg=128
    )