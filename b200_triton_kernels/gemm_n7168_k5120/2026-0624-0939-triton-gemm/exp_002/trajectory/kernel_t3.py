import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from typing import List


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
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        m_offset = pid_m * BLOCK_M
        n_offset = pid_n * BLOCK_N
        
        b_0 = b_desc.load([n_offset, 0])
        b_1 = tl.zeros((BLOCK_N, BLOCK_K), tl.bfloat16)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_step in range(0, K, BLOCK_K):
            next_k = k_step + BLOCK_K
            stage = (k_step // BLOCK_K) & 1
            next_stage = (stage + 1) & 1
            
            if next_k < K:
                if next_stage == 0:
                    b_0 = b_desc.load([n_offset, next_k])
                else:
                    b_1 = b_desc.load([n_offset, next_k])
            
            a = a_desc.load([m_offset, k_step])
            
            if stage == 0:
                b = b_0
            else:
                b = b_1
                
            acc = tl.dot(a, b.T, acc)
            
        c_desc.store([m_offset, n_offset], acc.to(tl.bfloat16))


def run(A, B, C):
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 256
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    NUM_SMS = 132
    grid = (min(NUM_SMS, num_tiles),)
    
    _gemm_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
        num_warps=4,
        num_stages=2,
    )