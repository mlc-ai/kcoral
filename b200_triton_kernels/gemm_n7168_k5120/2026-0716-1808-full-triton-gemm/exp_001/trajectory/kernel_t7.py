import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    c_desc,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    if offset_m < M:
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        num_k_tiles = K // BLOCK_K
        
        for k_tile in tl.range(num_k_tiles, num_stages=3):
            offset_k = k_tile * BLOCK_K
            
            a = a_desc.load([offset_m, offset_k])
            b = b_desc.load([offset_n, offset_k])
            
            acc = tl.dot(a, b.T, acc)
            
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16), boundary_check=(0,))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], device=True)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K], device=True)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], device=True)
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=3,
    )