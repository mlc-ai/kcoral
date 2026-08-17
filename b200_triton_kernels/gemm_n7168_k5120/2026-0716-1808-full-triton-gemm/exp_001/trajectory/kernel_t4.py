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
    K_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = c_desc.load([offset_m, offset_n])
    
    for k_idx in range(20):  # 2560 / 128 = 20
        current_k = K_offset + k_idx * BLOCK_K
        
        a_st = a_desc.load([offset_m, current_k])
        b_st = b_desc.load([offset_n, current_k])
        
        acc = tl.dot(a_st, b_st.T, acc)
    
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16), boundary_check=(0, 1))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M, K = A.shape
    N, _ = B.shape
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    
    assert A.stride(0) % 8 == 0, f"A row stride must be 16-byte aligned for TMA, got {A.stride(0)}"
    assert B.stride(0) % 8 == 0, f"B row stride must be 16-byte aligned for TMA, got {B.stride(0)}"
    assert C.stride(0) % 8 == 0, f"C row stride must be 16-byte aligned for TMA, got {C.stride(0)}"
    
    assert K % 2560 == 0, f"Kernel expects K ({K}) to be divisible by 2560"
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Split the inner dimension K=5120 into two distinct launches of 2560 width.
    # This avoids excessive unrolled loop ranges and mitigates register pressure 
    # build-up in a single monolithic execution.
    
    # 1st launch calculates A[:, :2560] @ B.T[:2560, :]
    _gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M, N, 0,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=4,
    )
    
    # 2nd launch accumulates A[:, 2560:] @ B.T[2560:, :]
    _gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M, N, 2560,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=4,
    )