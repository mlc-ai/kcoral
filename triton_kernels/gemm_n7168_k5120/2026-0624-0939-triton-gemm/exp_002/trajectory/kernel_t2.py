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
    STRIDE_AM: tl.constexpr,
    STRIDE_BK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    
    b_buf_0 = b_desc.load([n_offset, 0])
    b_buf_1 = b_desc.load([n_offset, 0])  # dummy load to init reg
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    phase = 0
    
    for k_step in range(0, K, BLOCK_K):
        next_phase = phase + 1
        next_k = k_step + BLOCK_K
        
        if next_k < K:
            if next_phase % 2 == 0:
                b_buf_1 = b_desc.load([n_offset, next_k])
            else:
                b_buf_0 = b_desc.load([n_offset, next_k])
                
        a = a_desc.load([m_offset, k_step])
        
        if phase % 2 == 0:
            b = b_buf_0
        else:
            b = b_buf_1
            
        acc = tl.dot(a, b.T, acc)
        phase = next_phase
        
    c_desc.store([m_offset, n_offset], acc.to(tl.bfloat16))


def run(A, B, C):
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m, num_pid_n)
    
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc, M, N, K,
        STRIDE_AM=M,
        STRIDE_BK=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )