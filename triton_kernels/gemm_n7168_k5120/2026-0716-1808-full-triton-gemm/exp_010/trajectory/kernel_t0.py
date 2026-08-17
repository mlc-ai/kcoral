import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Iterate over K dimension tiles to perform the full reduction
    for k_tile in range(0, 5120 // BLOCK_K):
        offset_k = k_tile * BLOCK_K
        
        # Load tiles mapped to [0 .. BLOCK_DIM - 1] logically via descriptors
        a = a_desc.load([0, offset_k])
        b = b_desc.load([0, offset_k])
        
        # Initialize accumulator
        if k_tile == 0:
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        # Compute partial dot product contribution
        acc = tl.dot(a, b.T, acc)

    # Final store utilizing native TMA bounds checking 
    acc_bf16 = acc.to(tl.bfloat16)
    c_desc.store([0, 0], acc_bf16)


def run(A, B, C):
    """Perform destination-passing GEMM computation C = A @ B.T."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    # Utilize Hopper Tensor Descriptors for optimized TMA movement
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    # Order grid primarily along N blocks so adjacent CTAs overlap heavily in L2 cache for A
    grid = (7168 // BLOCK_N, triton.cdiv(M, BLOCK_M))
    
    _gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )