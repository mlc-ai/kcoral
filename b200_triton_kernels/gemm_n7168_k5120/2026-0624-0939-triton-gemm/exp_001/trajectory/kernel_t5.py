import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K, N_HALF,
    stride_a_m, stride_a_k, stride_b_n, stride_b_k, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N_HALF: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Utilize a 3D grid explicitly iterating across M coordinates, condensed N-blocks, and the N-split factor
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    half_idx = tl.program_id(2)
    
    offset_m = pid_m * BLOCK_M
    n_base = half_idx * N_HALF
    offset_n = n_base + pid_n * BLOCK_N_HALF
    
    row_idx = offset_m + tl.arange(0, BLOCK_M)
    col_idx = offset_n + tl.arange(0, BLOCK_N_HALF)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N_HALF), tl.float32)
    
    num_k_full = K // BLOCK_K
    for k_step in range(num_k_full):
        k_idx = k_step * BLOCK_K + tl.arange(0, BLOCK_K)
        ptrs_a = A_ptr + row_idx[:, None] * stride_a_m + k_idx[None, :] * stride_a_k
        a_tile = tl.load(ptrs_a, mask=(row_idx[:, None] < M), other=0.0)
        
        ptrs_b = B_ptr + col_idx[:, None] * stride_b_n + k_idx[None, :] * stride_b_k
        b_tile = tl.load(ptrs_b, mask=(col_idx[:, None] < N_HALF), other=0.0)
        
        acc = tl.dot(a_tile, b_tile.T, acc)
    
    k_offset = num_k_full * BLOCK_K
    if k_offset < K:
        rem = K - k_offset
        a_tile = tl.full((BLOCK_M, BLOCK_K), 0.0, dtype=tl.bfloat16)
        valid_k = k_offset + tl.arange(0, BLOCK_K) < K
        valid_k_idx = k_offset + tl.arange(0, BLOCK_K)
        
        a_val = tl.load(A_ptr + row_idx[:, None] * stride_a_m + valid_k_idx[None, :] * stride_a_k)
        a_tile = tl.where(valid_k_idx[None, :], a_val, a_tile)
        
        b_tile = tl.full((BLOCK_N_HALF, BLOCK_K), 0.0, dtype=tl.bfloat16)
        b_val = tl.load(B_ptr + col_idx[:, None] * stride_b_n + valid_k_idx[None, :] * stride_b_k)
        b_tile = tl.where(valid_k_idx[None, :], b_val, b_tile)
        
        acc = tl.dot(a_tile, b_tile.T, acc)
        
    ptrs_c = C_ptr + row_idx[:, None] * stride_c_m + col_idx[None, :] * stride_c_n
    mask_c = (row_idx[:, None] < M) & (col_idx[None, :] < n_base + N_HALF)
    tl.store(ptrs_c, acc.to(tl.bfloat16), mask=mask_c)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    # Partition the N dimension cleanly into two independent processing chunks
    N_HALF = N // 2  
    
    BLOCK_M = 128
    BLOCK_N_HALF = 256
    BLOCK_K = 64
    
    # Rely strictly on standard pointer indexing avoiding descriptor shape limits entirely
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N_HALF, BLOCK_N_HALF),
        2
    )
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K, N_HALF,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N_HALF=BLOCK_N_HALF, BLOCK_K=BLOCK_K,
        num_warps=8, num_stages=3,
    )