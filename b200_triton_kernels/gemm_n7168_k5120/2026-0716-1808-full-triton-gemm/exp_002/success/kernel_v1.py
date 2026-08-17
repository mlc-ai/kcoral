import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# Optimized configuration leveraging Hopper TMA and Tensor Cores
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    dtype = c_ptr.dtype.element_ty
    
    # Utilize a 2D grid mapped primarily over N-tiles first to enable 
    # massive hardware zero-copy sharing of B slices across CTAs via L2.
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offset_n = pid_n * BLOCK_N
    offset_m = pid_m * BLOCK_M
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k_tile in range(num_k_tiles):
        offset_k = k_tile * BLOCK_K
        
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        acc = tl.dot(a, b.T, acc)
    
    c_desc.store([offset_m, offset_n], acc.to(dtype))


def run(A, B, C):
    """Compute C = A @ B.T using optimized Hopper GEMM logic."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    if M == 0:
        return

    # Fix highly optimized tile sizes matching the strategy
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 128
    
    # Create optimized TMA descriptors on host
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    # Grid ordered by N first to enable zero-copy sharing of B slices via L2
    grid = (
        triton.cdiv(N, BLOCK_N),
        triton.cdiv(M, BLOCK_M),
    )
    
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        A, B, C,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )