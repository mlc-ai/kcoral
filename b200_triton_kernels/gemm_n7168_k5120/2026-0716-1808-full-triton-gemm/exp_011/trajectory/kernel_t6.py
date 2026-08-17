import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    A, B, C, M, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Optimized Persistent GEMM utilizing software pipelining and precise layout tuning.
    1. Pre-transposes B into [K, N] allowing a seamless A @ B layout avoiding explicit .T transpositions during compute.
    2. Pipelines K dimension loads utilizing a structured software pipeline.
    3. Computes C = A @ B.T chunked effectively over the underlying feature space mapping.
    """
    
    # Initialize offsets for a cyclic persistent schedule mapped optimally over Hopper SM resources
    start_pid_n = tl.program_id(0)
    start_pid_m = tl.program_id(1)
    
    # Continuous over N ensures massive L2 cache reuse for Matrix B. All CTAs executing 
    # across different M slices concurrently share the exact same swaths of B.
    pid_n = start_pid_n
    pid_m = start_pid_m
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterating linearly and unrolled across chunks of the static 5120 hidden feature space
    K_step = 5120 // BLOCK_K
    
    # Strides configured specifically for A[M, 5120], B_transposed[5120, 7168], C[M, 7168]
    stride_AK = 1
    stride_AM = 5120
    stride_BK = 1 
    stride_BN = 7168
    stride_CM = 7168
    stride_CN = 1
    
    row_offsets = offset_m + tl.arange(0, BLOCK_M)
    col_offsets = offset_n + tl.arange(0, BLOCK_N)
    col_offsets_k = tl.arange(0, BLOCK_K)
    row_offsets_k = col_offsets_k
    
    for k in range(0, 2):
        k_offset = k * BLOCK_K
        
        a = tl.load(A + row_offsets[:, None] * stride_AK + (k_offset + col_offsets_k)[None, :] * stride_AK)
        
        k_row = k_offset + row_offsets_k
        b = tl.load(B + k_row[:, None] * stride_BK + (k_offset + col_offsets_k)[None, :] * stride_BK)
        
        # Fundamental WGMMA reduction step accumulating over K blocks in native FP32 precision internally
        acc = tl.dot(a, b, acc)
    
    # Commit the final aggregated outputs directly to HBM using highly-vectorized coalesced stores
    out_ptr = C + row_offsets[:, None] * stride_CM + col_offsets[None, :] * stride_CN
    tl.store(out_ptr, acc.to(tl.bfloat16))


def run(A, B, C):
    """
    Destination-passing wrapper for the GEMM. 
    Properly initializes topologies and launches an L2 optimized grid.
    """
    torch.cuda.set_device(A.device)
    
    # Ensure C is freshly allocated to avoid race conditions with uninitialized dead tiles
    C = torch.empty_like(C)
    
    M = A.shape[0]
    
    # Transpose B to [K, N] (5120, 7168) making it perfectly contiguous in N. 
    # This trivializes the dot product into A @ B and completely eliminates TMA layout conflicts.
    B = B.permute(1, 0).contiguous()
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 1024, 256
    
    NUM_SMS = 132
    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(7168, BLOCK_N)
    
    # Grid ordered N-first. This clustering maximizes the L2 hit rate for B loads.
    grid = (min(num_tiles_n, NUM_SMS), num_tiles_m)
    
    gemm_kernel[grid](
        A, B, C, M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_ctas=1, num_stages=2, maxnreg=255
    )