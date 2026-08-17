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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    if offset_m < M and offset_n < N:
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        num_k_tiles = tl.cdiv(K, BLOCK_K)
        
        for k0 in range(num_k_tiles):
            offset_k = k0 * BLOCK_K
            
            a_st = a_desc.load([offset_m, offset_k])
            b_st = b_desc.load([offset_n, offset_k])
            
            acc = tl.dot(a_st, b_st.T, acc)
            
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 128
    
    assert A.stride(0) % 8 == 0, f"A row stride must be 16-byte aligned for TMA, got {A.stride(0)}"
    assert B.stride(0) % 8 == 0, f"B row stride must be 16-byte aligned for TMA, got {B.stride(0)}"
    assert C.stride(0) % 8 == 0, f"C row stride must be 16-byte aligned for TMA, got {C.stride(0)}"
    
    assert K % 128 == 0, f"K ({K}) must be divisible by 128 for static shape padding."
    assert N % 256 == 0, f"N ({N}) must be divisible by 256 for static shape padding."
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=4,
    )