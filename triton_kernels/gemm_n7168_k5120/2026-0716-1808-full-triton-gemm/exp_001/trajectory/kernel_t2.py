import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc,
    b_desc,
    C,
    M,
    N,
    K,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # Safety check to avoid running loops on completely OOB tiles when M < BLOCK_M
    if offset_m < M:
        num_k_tiles = K // BLOCK_K
        
        for k0 in range(num_k_tiles):
            offset_k = k0 * BLOCK_K
            
            a_st = a_desc.load([offset_m, offset_k])
            b_st = b_desc.load([offset_n, offset_k])
            
            acc = tl.dot(a_st, b_st.T, acc)
            
        M_left = M - offset_m
        
        C_ptr = C + offset_m * stride_cm + offset_n * stride_cn
        
        arange_m = tl.arange(0, BLOCK_M)
        arange_n = tl.arange(0, BLOCK_N)
        
        if M_left >= BLOCK_M:
            tl.store(C_ptr + arange_m[:, None] * stride_cm + arange_n[None, :] * stride_cn, 
                     acc.to(C.dtype))
        else:
            mask = (arange_m < M_left)[:, None] & (arange_n < N)[None, :]
            tl.store(C_ptr + arange_m[:, None] * stride_cm + arange_n[None, :] * stride_cn, 
                     acc.to(C.dtype), mask=mask)


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 128
    
    with torch.compile.disable():
        assert A.stride(0) % 8 == 0, f"A row stride must be 16-byte aligned for TMA, got {A.stride(0)}"
        assert B.stride(0) % 8 == 0, f"B row stride must be 16-byte aligned for TMA, got {B.stride(0)}"
        
        # Padding logic expects consistent multiples of 128 across fully utilized layouts.
        assert K % 128 == 0, f"K ({K}) must be divisible by 128 for static shape padding."
        assert N % 256 == 0, f"N ({N}) must be divisible by 256 for static shape padding."
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    
    grid = (triton.cdiv(M, BLOCK_M), N // BLOCK_N)
    _gemm_kernel[grid](
        a_desc,
        b_desc,
        C,
        M, N, K,
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=4,
    )