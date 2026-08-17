import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc_A,
    a_desc_B,
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
        
        # Pipeling loop across K dimension blocks
        a = a_desc_A.load([offset_m, 0])
        b = a_desc_B.load([offset_n, 0])
        
        for k_idx in range(1, K // BLOCK_K):
            offset_k = k_idx * BLOCK_K
            a_next = a_desc_A.load([offset_m, offset_k])
            b_next = a_desc_B.load([offset_n, offset_k])
            acc = tl.dot(a, b.T, acc)
            a = a_next
            b = b_next
        
        # Process the final chunk
        acc = tl.dot(a, b.T, acc)
        
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
    
    a_desc_A = TensorDescriptor.from_tensor(A, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    a_desc_B = TensorDescriptor.from_tensor(B, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _gemm_kernel[grid](
        a_desc_A, a_desc_B, c_desc,
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=2,
    )