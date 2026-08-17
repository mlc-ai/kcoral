import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel_persistent(
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
    """
    Persistent tiled GEMM. Computes C = A @ B.T
    Uses tensor descriptors for efficient TMA loads and stores.
    Grid is 1D across N-tiles; each program loops over all M-tiles.
    """
    pid_n = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    
    offset_n = pid_n * BLOCK_N
    if offset_n >= N:
        return
    
    for m_tile in range(num_pid_m):
        offset_m = m_tile * BLOCK_M
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_step in range(tl.cdiv(K, BLOCK_K)):
            a = a_desc.load([offset_m, k_step * BLOCK_K])
            b = b_desc.load([offset_n, k_step * BLOCK_K])
            acc = tl.dot(a, b.T, acc)
        
        c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_sms = torch.cuda.get_device_properties(A.device).multi_processor_count
    
    grid = (min(num_sms, num_pid_n),)
    
    _gemm_kernel_persistent[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )