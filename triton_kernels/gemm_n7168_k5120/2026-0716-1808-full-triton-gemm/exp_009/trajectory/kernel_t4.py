import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N_dim,
    K_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K_dim],
        strides=[K_dim, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero",
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[N_dim, K_dim],
        strides=[K_dim, 1],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero",
    )
    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[M, N_dim],
        strides=[N_dim, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    
    start_tile = tl.program_id(0)
    stride = tl.num_programs(0)
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N_dim, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in tl.range(start_tile, num_tiles, stride):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        
        m_offset = pid_m * BLOCK_M
        n_offset = pid_n * BLOCK_N
        
        if m_offset >= M:
            break
            
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k0 in range(0, triton.cdiv(K_dim, BLOCK_K)):
            k_offset = k0 * BLOCK_K
            
            a = a_desc.load([m_offset, k_offset])
            b = b_desc.load([n_offset, k_offset])
            
            acc = tl.dot(a, b.T, acc)
        
        c_desc.store([m_offset, n_offset], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N_dim = B.shape[0]
    K_dim = B.shape[1]
    
    block_m = 256
    block_n = 256
    block_k = 128
    
    grid = (128,) 
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    
    triton.set_allocator(alloc_fn)
    
    _gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N_dim,
        K_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8,
        num_stages=3,
    )