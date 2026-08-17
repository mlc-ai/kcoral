import torch
import triton
import triton.language as tl


@triton.jit
def _batched_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    return


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    STAGES: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    num_steps = ((M + BLOCK_M - 1) // BLOCK_M) * num_n_tiles
    
    for step in tl.range(start_pid, num_steps, NUM_SMS, flatten=False):
        pid_n = step // num_n_tiles
        pid_m = step % num_n_tiles
        
        offset_n = pid_n * BLOCK_N
        offset_m = pid_m * BLOCK_M
        
        off_n = offset_n + tl.arange(0, BLOCK_N)
        off_m = offset_m + tl.arange(0, BLOCK_M)
        
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        num_k_tiles = (K + BLOCK_K - 1) // BLOCK_K
        for k_step in tl.range(0, num_k_tiles, 1, num_stages=STAGES):
            off_k = k_step * BLOCK_K + tl.arange(0, BLOCK_K)
            
            a = tl.load(a_ptr + off_m[:, None] * K + off_k[None, :], mask=(off_m < M)[:, None] & (off_k < K)[None, :], other=0.0)
            b = tl.load(b_ptr + off_n[:, None] * K + off_k[None, :], mask=(off_n < N)[:, None] & (off_k < K)[None, :], other=0.0)
            
            acc = tl.dot(a, b.T, acc)
        
        c_ptr_base = c_ptr + off_m[:, None] * N + off_n[None, :]
        tl.store(c_ptr_base, acc.to(tl.bfloat16), mask=(off_m < M)[:, None] & (off_n < N)[None, :])


def run(A, B, C):
    """Compute C = A @ B.T using optimized Hopper GEMM logic."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    if M == 0:
        return

    BLOCK_M = 1024
    BLOCK_N = 256
    BLOCK_K = 128
    NUM_SMS = 132
    STAGES = 3
    
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    num_steps = ((M + BLOCK_M - 1) // BLOCK_M) * num_n_tiles
    grid_size = min(NUM_SMS, num_steps)
    grid = (grid_size,)
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
        STAGES=STAGES,
        num_warps=8,
    )