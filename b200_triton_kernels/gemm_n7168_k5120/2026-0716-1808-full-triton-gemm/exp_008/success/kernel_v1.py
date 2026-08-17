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
    N_out,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Implements C = A @ B^T using Hopper TMA descriptors and a 2D grid mapping.
    Accumulates internally in FP32 utilizing tensor cores.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = k_tile * BLOCK_K
        
        a = a_desc.load([offset_m, offset_k])
        b = b_desc.load([offset_n, offset_k])
        
        acc = tl.dot(a, b.T, acc)
            
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into the preallocated tensor C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N_out = B.shape[0]
    K = B.shape[1]
    
    block_m = 128
    block_n = 128
    block_k = 128
    
    a_desc = TensorDescriptor.from_tensor(A, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(B, [block_n, block_k])
    c_desc = TensorDescriptor.from_tensor(C, [block_m, block_n])
    
    grid = (triton.cdiv(M, block_m), triton.cdiv(N_out, block_n))
    
    _gemm_kernel[grid](
        a_desc, 
        b_desc, 
        c_desc, 
        M, N_out, K,
        BLOCK_M=block_m, 
        BLOCK_N=block_n, 
        BLOCK_K=block_k,
        num_warps=4, 
        num_stages=3
    )