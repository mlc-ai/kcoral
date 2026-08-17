import torch
import triton
import triton.language as tl


# Naive fallback ensuring correctness and avoiding unnecessary kernel launches.
@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
):
    dtype = c_ptr.dtype.element_ty
    
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[128, 128],
        padding_option="zero",
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[64, 128],
        padding_option="zero",
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[128, 64],
    )
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset_m = pid_m * 128
    offset_n = pid_n * 64
    
    acc = tl.zeros((128, 64), tl.float32)
    
    num_k_tiles = tl.cdiv(K, 128)
    for k_tile in range(num_k_tiles):
        offset_k = k_tile * 128
        
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        acc = tl.dot(a, b.T, acc)
    
    c_desc.store([offset_m, offset_n], acc.to(dtype))


def run(A, B, C):
    """Compute C = A @ B.T using optimized CUTLASS-style logic."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]  # Expecting 7168
    K = A.shape[1]  # Expecting 5120
    
    if M == 0:
        return

    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    
    grid = (
        triton.cdiv(M, 128),
        triton.cdiv(N, 64)
    )
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        num_warps=8,
        num_stages=2,
    )