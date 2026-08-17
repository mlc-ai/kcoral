import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel_desc(
    A_desc, 
    B_desc, 
    C_desc, 
    M, 
    N, 
    K, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offset_k = k0 * BLOCK_K
        
        a = A_desc.load([offset_m, offset_k])
        b = B_desc.load([offset_n, offset_k])
        
        acc = tl.dot(a, b.T, acc)
    
    C_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    block_m = 64
    block_n = 256
    block_k = 64
    
    A_desc = TensorDescriptor.from_tensor(A, [block_m, block_k])
    B_desc = TensorDescriptor.from_tensor(B, [block_n, block_k])
    C_desc = TensorDescriptor.from_tensor(C, [block_m, block_n])
    
    num_pid_m = triton.cdiv(M, block_m)
    num_pid_n = 28  
    grid = (num_pid_m, num_pid_n)
    
    _gemm_kernel_desc[grid](
        A_desc,
        B_desc,
        C_desc,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8,
        num_stages=3,
    )