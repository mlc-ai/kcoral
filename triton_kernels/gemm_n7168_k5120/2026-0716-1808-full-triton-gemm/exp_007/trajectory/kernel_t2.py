import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.jit
def gemm_kernel(A_ptr, B_ptr, C_ptr, M, N, K, 
                stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, 
                NUM_K_TILES: tl.constexpr, GROUP_M: tl.constexpr, NUM_SMS: tl.constexpr):
    
    A_desc = tl.make_tensor_descriptor(A_ptr, shape=[M, K], strides=[stride_am, 1], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero")
    B_desc = tl.make_tensor_descriptor(B_ptr, shape=[N, K], strides=[stride_bn, 1], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero")
    C_desc = tl.make_tensor_descriptor(C_ptr, shape=[M, N], strides=[stride_cm, 1], block_shape=[BLOCK_M, BLOCK_N])
    
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False, warp_specialize=True):
        group_id = tile_id // (GROUP_M * num_pid_n)
        first_m = group_id * GROUP_M
        m_range = min(num_pid_m - first_m, GROUP_M)
        
        idx = tile_id % (GROUP_M * num_pid_n)
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
    
    NUM_SMS = 132
    
    triton.set_allocator(alloc_fn)
    
    num_pid_m = triton.cdiv(M, 128)
    num_pid_n = triton.cdiv(N, 128)
    num_programs = min(num_pid_m * num_pid_n, NUM_SMS)
    
    grid = (num_programs,)
    gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, 
        NUM_K_TILES=triton.cdiv(K, 128), GROUP_M=8, NUM_SMS=NUM_SMS,
        num_warps=8
    )