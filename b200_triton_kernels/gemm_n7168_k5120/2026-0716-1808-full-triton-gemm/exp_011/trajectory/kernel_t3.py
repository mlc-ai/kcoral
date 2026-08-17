import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(A, B, C, M, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Persistent GEMM pass for computing C = A @ B.T.
    Uses Hardware TMA via runtime TensorDescriptors to effectively stage data transfers 
    dictated by intrinsic swizzling patterns required for underlying Hopper WGMMA execution paths.
    """
    N = B.shape[0]
    K = B.shape[1]
    
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
    
    start_pid = tl.program_id(0)
    tile_stride = tl.num_programs(0)
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in range(start_pid, num_tiles, tile_stride):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_K):
            a = a_desc.load([offset_m, k])
            global_row = offset_m + tl.arange(0, BLOCK_M)
            a = tl.where(global_row[:, None] < M, a, 0.0)
            
            b = b_desc.load([offset_n, k])
            
            acc = tl.dot(a, b.T, acc)
        
        row = tl.arange(0, BLOCK_M)
        col = tl.arange(0, BLOCK_N)
        out_ptr = C + (offset_m + row)[:, None] * N + (offset_n + col)[None, :]
        tl.store(out_ptr, acc.to(tl.bfloat16), mask=((offset_m + row)[:, None] < M) & ((offset_n + col)[None, :] < N))


def run(A, B, C):
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    
    NUM_SMS = 132
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 128
    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(7168, BLOCK_N)
    grid = (min(NUM_SMS, num_tiles),)
    
    gemm_kernel[grid](
        A, B, C, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_ctas=1, num_stages=2
    )