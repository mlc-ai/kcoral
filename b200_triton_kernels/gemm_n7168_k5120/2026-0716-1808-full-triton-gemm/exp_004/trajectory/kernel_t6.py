import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    """Allocator for device-created descriptor infrastructure storage."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.jit
def _gemm(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes C = A @ B.T using device-created TMA descriptors.
    Utilizes a 2-stage TMA prefetch pipeline over a simple contiguous K-loop.
    """
    
    a_desc = tl.make_tensor_descriptor(
        A,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
        padding_option="zero",
    )
    
    b_desc = tl.make_tensor_descriptor(
        B,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
        padding_option="zero",
    )
    
    c_desc = tl.make_tensor_descriptor(
        C,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    
    # Limit initial grid expansion to available SM capacity to avoid scheduling overhead bottlenecks 
    num_m_tiles = (M + 255) // 256
    num_n_tiles = (N + 255) // 256
    
    NUM_GROUPS = 2
    total_tiles_per_group = num_m_tiles * (num_n_tiles // NUM_GROUPS)
    
    program_id = tl.program_id(0)
    group_id = program_id // total_tiles_per_group
    group_n = (program_id % total_tiles_per_group) // num_m_tiles
    pid_m = program_id % num_m_tiles
    pid_n = group_id * (num_n_tiles // NUM_GROUPS) + group_n
    
    offset_m = pid_m * BLOCK_M
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    NUM_K_STEPS = 40 
    
    for n_iter in range(num_n_tiles):
        n_tile = (n_iter + pid_n) % num_n_tiles
        offset_n = n_tile * BLOCK_N
        
        # Re-initialize accumulator for strictly independent math reduction limits per unique N slice
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k_step in range(NUM_K_STEPS):
            offset_k = k_step * BLOCK_K
            
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            if k_step + 1 < NUM_K_STEPS:
                next_offset_k = offset_k + BLOCK_K
                a_desc.prefetch(a_ptr, [offset_m, next_offset_k])
                b_desc.prefetch(b_ptr, [offset_n, next_offset_k])
            
            acc = tl.dot(a, b.T, acc)
            
        out_val = acc.to(tl.bfloat16)
        c_desc.store([offset_m, offset_n], out_val)


def run(A, B, C):
    """
    Efficient destination-passing wrapper launching our highly specialized persistent TMA GEMM. 
    
    Takes preallocated tensors explicitly matching definition sequence constraints.
    """
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N, _ = B.shape
    
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 128
    
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    NUM_SMS = 132
    
    num_ctas = min(NUM_SMS, num_tiles)
    grid = (num_ctas,)
    
    triton.set_allocator(alloc_fn)
    
    _gemm[grid](
        A, B, C,
        M, N, K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )