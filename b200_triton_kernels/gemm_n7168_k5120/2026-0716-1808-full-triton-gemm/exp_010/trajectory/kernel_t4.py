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
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Use a straightforward 2D mapping where each program owns exactly one output tile.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # Explicit scalar iteration over the contiguous dimension allows us to avoid 
    # `tl.arange` runtime errors and manually pipeline the descriptor loads.
    for k_tile in range(K // BLOCK_K):
        offset_k = k_tile * BLOCK_K
        
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        acc = tl.dot(a, b.T, acc)

    acc_bf16 = acc.to(tl.bfloat16)
    c_desc.store([offset_m, offset_n], acc_bf16)


def run(A, B, C):
    """Perform destination-passing GEMM computation C = A @ B.T."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0] # Expected 7168
    K = A.shape[1] # Expected 5120
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    # Utilize Hopper Tensor Descriptors for optimized TMA movement
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Zero-fill prevents uninitialized memory reads in boundary scenarios
    torch.ops.triton.memory.rewind_and_zero_fill(C)
    
    _gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )