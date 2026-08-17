import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N_out,
    K,
    stride_C_m,
    stride_C_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Implements C = A @ B^T using Hopper TMA descriptors and a 2D grid mapping.
    Accumulates internally in FP32 utilizing tensor cores and emits static loop unrolling.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Since the task defines a fixed constant N = 7168, we leverage it explicitly.
    num_pid_n = 56 
    
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # Exactly 40 iterations for K=5120 and BLOCK_K=128
    for k_tile in tl.static_range(0, 40, unroll_factor=8):
        offset_k = k_tile * BLOCK_K
        
        a = a_desc.load([row_idx[0], offset_k])
        b = b_desc.load([col_idx[0], offset_k])
        
        acc = tl.dot(a, b.T, acc)
            
    mask_m = row_idx < M
    mask_n = col_idx < N_out
    
    ptr_C = c_ptr + row_idx[:, None] * stride_C_m + col_idx[None, :] * stride_C_n
    tl.store(ptr_C, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


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
    
    grid = (triton.cdiv(M, block_m), triton.cdiv(N_out, block_n))
    
    _gemm_kernel[grid](
        a_desc, 
        b_desc, 
        C, 
        M, N_out, K,
        C.stride(0), C.stride(1),
        BLOCK_M=block_m, 
        BLOCK_N=block_n, 
        BLOCK_K=block_k,
        num_warps=4, 
        num_stages=3
    )