import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    ptr_a,
    ptr_b, 
    ptr_c,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    dtype = ptr_c.dtype.element_ty
    
    a_desc = tl.make_tensor_descriptor(
        ptr_a, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero")
    
    b_desc = tl.make_tensor_descriptor(
        ptr_b, shape=[N, K], strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero")
    
    c_desc = tl.make_tensor_descriptor(
        ptr_c, shape=[M, N], strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N])

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offset_m = pid_m * BLOCK_M
        offset_n = pid_n * BLOCK_N
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        num_k_tiles = tl.cdiv(K, BLOCK_K)
        for k_tile in range(num_k_tiles):
            offset_k = k_tile * BLOCK_K
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            acc = tl.dot(a, b, acc)

        c_desc.store([offset_m, offset_n], acc.to(dtype))


def run(A, B, C):
    torch.cuda.set_device(A.device)
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256
    
    NUM_SMS = 132
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    grid = (min(NUM_SMS, num_tiles),)
    
    _gemm_kernel[grid](
        A, B, C, M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
        WARP_SPECIALIZE=True,
        num_warps=8,
        num_stages=3,
    )