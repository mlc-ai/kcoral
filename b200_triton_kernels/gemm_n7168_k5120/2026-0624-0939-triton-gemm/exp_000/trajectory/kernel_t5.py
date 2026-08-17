import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _gemm_kernel(
    a_desc, b_desc, c_desc,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_steps = tl.cdiv(K, BLOCK_K)
    
    for k_idx in tl.range(0, num_k_steps, num_stages=4):
        offset_k = k_idx * BLOCK_K
        a_tile = a_desc.load([offset_m, offset_k])
        b_tile = b_desc.load([offset_n, offset_k])
        acc = tl.dot(a_tile, b_tile.T, acc)
    
    c_desc.store([offset_m, offset_n], acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 128
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m, num_pid_n)
    
    _gemm_kernel[grid](a_desc, b_desc, c_desc, M, N, K,
                       BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                       num_warps=8, num_stages=4)