import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.jit
def _gemm_kernel_persistent(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    a_desc = tl.make_tensor_descriptor(
        A, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K], padding_option="zero")
    
    b_desc = tl.make_tensor_descriptor(
        B, shape=[N, K], strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K], padding_option="zero")
    
    c_desc = tl.make_tensor_descriptor(
        C, shape=[M, N], strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N])

    start_pid_m = tl.program_id(0)
    tile_stride_m = tl.num_programs(0)
    pid_n = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)

    offset_n = pid_n * BLOCK_N
    if offset_n >= N:
        return
    
    num_k_steps = K // BLOCK_K

    for m_tile in tl.range(
        start_pid_m,
        num_pid_m,
        tile_stride_m,
        flatten=False,
        warp_specialize=WARP_SPECIALIZE,
    ):
        offset_m = m_tile * BLOCK_M
        if offset_m >= M:
            return
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_step in range(num_k_steps):
            a = a_desc.load([offset_m, k_step * BLOCK_K])
            b = b_desc.load([offset_n, k_step * BLOCK_K])
            acc = tl.dot(a, b.T, acc)
        
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    grid = (min(num_sms, num_pid_m), num_pid_n)
    
    _gemm_kernel_persistent[grid](
        A, B, C, M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        WARP_SPECIALIZE=True,
        num_warps=8,
        num_stages=3,
    )