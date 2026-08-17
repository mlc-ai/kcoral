import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    A, B, C, M, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Fused GEMM pass utilizing precise layout tuning and coalesced vector loads.
    1. Uses direct pointer arithmetic with fully unrolled strides mapping.
    2. Iterates linearly across chunks of the static 5120 hidden feature space.
    3. Computes C = A @ B.T chunked effectively over the underlying feature space mapping.
    """
    
    # Initialize offsets mapped optimally over Hopper SM resources utilizing a 2D grid mapping
    start_pid_m = tl.program_id(0)
    start_pid_n = tl.program_id(1)
    
    pid_m = start_pid_m
    pid_n = start_pid_n
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterating linearly across chunks of the static 5120 hidden feature space
    K_step = 5120 // BLOCK_K
    
    # Strides configured specifically for A[M, 5120], B[7168, 5120], C[M, 7168]
    stride_A_row = 5120
    stride_A_col = 1
    stride_B_row = 5120
    stride_B_col = 1
    stride_CM = 7168
    stride_CN = 1
    
    row_offsets = offset_m + tl.arange(0, BLOCK_M)
    col_offsets = offset_n + tl.arange(0, BLOCK_N)
    col_offsets_k = tl.arange(0, BLOCK_K)
    
    for k in range(0, K_step, 1):
        k_offset = k * BLOCK_K
        
        a = tl.load(A + row_offsets[:, None] * stride_A_row + (k_offset + col_offsets_k)[None, :] * stride_A_col, mask=(row_offsets[:, None] < M), other=0.0)
        
        b = tl.load(B + col_offsets[:, None] * stride_B_row + (k_offset + col_offsets_k)[None, :] * stride_B_col, mask=((k_offset + col_offsets_k)[None, :] < 5120) & (col_offsets[:, None] < 7168), other=0.0)
        
        # Fundamental WGMMA reduction step accumulating over K blocks in native FP32 precision internally
        acc = tl.dot(a, b, acc)
    
    # Commit the final aggregated outputs directly to HBM using highly-vectorized coalesced stores
    out_ptr = C + row_offsets[:, None] * stride_CM + col_offsets[None, :] * stride_CN
    tl.store(out_ptr, acc.to(tl.bfloat16), mask=((row_offsets[:, None] < M) & (col_offsets[None, :] < 7168)))


def run(A, B, C):
    """
    Destination-passing wrapper for the GEMM. 
    Properly initializes topologies and launches an L2 optimized grid.
    """
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
    
    NUM_SMS = 132
    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(7168, BLOCK_N)
    
    grid = (min(num_tiles_m, NUM_SMS), num_tiles_n)
    
    gemm_kernel[grid](
        A, B, C, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_ctas=1, num_stages=2, maxnreg=255
    )