import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=["M", "N", "K"],
)
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
    tile = c_desc
    
    __shared__ a_shared: [2, BLOCK_M, BLOCK_K]
    __shared__ b_shared: [2, BLOCK_N, BLOCK_K]
    
    extern __shared__ _smem[]
    a_shared = _smem[0 : 2 * BLOCK_M * BLOCK_K].reshape([2, BLOCK_M, BLOCK_K])
    b_shared = _smem[(2 * BLOCK_M * BLOCK_K) : (2 * BLOCK_M * BLOCK_K + 2 * BLOCK_N * BLOCK_K)].reshape([2, BLOCK_N, BLOCK_K])
    
    start_pid = tl.program_id(0)
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    num_steps = ((M + BLOCK_M - 1) // BLOCK_M) * num_n_tiles
    
    start_pid_m = start_pid // num_n_tiles
    start_pid_n = start_pid % num_n_tiles
    offset_m = start_pid_m * BLOCK_M
    offset_n = start_pid_n * BLOCK_N
    
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    off_k = tl.arange(0, BLOCK_K)
    off_m_base = (start_pid_m * BLOCK_M + off_m).to(tl.int64)
    off_n_base = (start_pid_n * BLOCK_N + off_n).to(tl.int64)
    
    thread_id = tl.program_id(0) * 128 + tl.arange(0, 128)
    num_parts = 4
    part_id = thread_id // (128 // num_parts)
    part_m = BLOCK_M // num_parts
    part_n = BLOCK_N // num_parts
    part_off_m = part_id * part_m
    part_off_n = part_id * part_n
    
    buf_idx = 0
    
    # Prefetch initial tiles
    if tl.program_id(0) == 0:
        tma_load(
            a_shared[0, part_off_m + off_m, off_k],
            a_desc, [0, 0, part_off_m + off_m_base, 0],
            shape=[1, 1, part_m, BLOCK_K],
        )
        tma_load(
            b_shared[0, part_off_n + off_n, off_k],
            b_desc, [0, 0, part_off_n + off_n_base, 0],
            shape=[1, 1, part_n, BLOCK_K],
        )
        tma_commit()
        if BLOCK_K <= K:
            tma_load(
                a_shared[1, part_off_m + off_m, off_k],
                a_desc, [0, 0, part_off_m + off_m_base, BLOCK_K],
                shape=[1, 1, part_m, BLOCK_K],
            )
            tma_load(
                b_shared[1, part_off_n + off_n, off_k],
                b_desc, [0, 0, part_off_n + off_n_base, BLOCK_K],
                shape=[1, 1, part_n, BLOCK_K],
            )
            tma_commit()
            
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    tma_wait(-1)
    
    for k_step in range(0, (K + BLOCK_K - 1) // BLOCK_K - 1):
        next_next_buf = 1 - buf_idx
        if (k_step + 2) * BLOCK_K <= K:
            if tl.program_id(0) == 0:
                off_k_base = (off_k * (k_step + 2) * BLOCK_K).to(tl.int64)
                
                tma_load(
                    a_shared[next_next_buf, part_off_m + off_m, off_k],
                    a_desc, [0, 0, part_off_m + off_m_base, off_k_base],
                    shape=[1, 1, part_m, BLOCK_K],
                )
                tma_load(
                    b_shared[next_next_buf, part_off_n + off_n, off_k],
                    b_desc, [0, 0, part_off_n + off_n_base, off_k_base],
                    shape=[1, 1, part_n, BLOCK_K],
                )
                tma_commit()
        
        tma_wait(-1)
        
        a = a_shared[buf_idx]
        b = b_shared[buf_idx]
        
        acc = tl.dot(a, b.T, acc, input_precision="ieee")
        
        buf_idx = 1 - buf_idx
        
    # Handle the last step outside the loop
    if (K + BLOCK_K - 1) // BLOCK_K > 0:
        tma_wait(-1)
        a = a_shared[buf_idx]
        b = b_shared[buf_idx]
        acc = tl.dot(a, b.T, acc, input_precision="ieee")
    
    c_desc.store(
        [offset_m, offset_n],
        acc.to(tile.dtype)
    )


def run(A, B, C):
    """Compute C = A @ B.T using optimized Hopper GEMM logic."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    if M == 0:
        return

    BLOCK_M = 128
    BLOCK_N = 512
    BLOCK_K = 128
    NUM_SMS = 132
    
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N])
    
    num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    num_steps = ((M + BLOCK_M - 1) // BLOCK_M) * num_n_tiles
    grid_size = min(NUM_SMS, num_steps)
    grid = (grid_size,)
    
    _gemm_kernel[grid](
        a_desc, b_desc, c_desc,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_SMS=NUM_SMS,
    )