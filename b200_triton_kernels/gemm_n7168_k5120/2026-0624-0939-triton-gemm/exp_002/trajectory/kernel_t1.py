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
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    
    b_buf_0 = tl.empty((BLOCK_N, BLOCK_K), tl.bfloat16)
    b_buf_1 = tl.empty((BLOCK_N, BLOCK_K), tl.bfloat16)
    b_buf = [b_buf_0, b_buf_1]
    
    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        m_offset = pid_m * BLOCK_M
        n_offset = pid_n * BLOCK_N
        
        next_idx = 0
        cur_idx = 0
        next_k = 0
        
        if next_k < K:
            b_buf[next_idx] = b_desc.load([n_offset, next_k])
            next_k += BLOCK_K
            
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for k_step in range(0, K, BLOCK_K):
            if next_k < K:
                b_buf[next_idx] = b_desc.load([n_offset, next_k])
                next_k += BLOCK_K
                
            a = a_desc.load([m_offset, k_step])
            acc = tl.dot(a, b_buf[cur_idx].T, acc)
            
            cur_idx ^= 1
            next_idx ^= 1
            
        c_desc.store([m_offset, n_offset], acc.to(tl.bfloat16))


def run(A, B, C):
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 128
    BLOCK_N = 256
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
        num_warps=8,
        num_stages=3,
    )