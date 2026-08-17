import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_a_m, stride_a_k, stride_b_n, stride_b_k, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Utilize a standard 2D grid iterating across M and N output coordinate tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    offset_n = pid_n * BLOCK_N
    
    row_idx = offset_m + tl.arange(0, BLOCK_M)
    col_idx = offset_n + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_full = K // BLOCK_K
    for k_step in range(num_k_full):
        k_idx = k_step * BLOCK_K + tl.arange(0, BLOCK_K)
        
        ptrs_a = A_ptr + row_idx[:, None] * stride_a_m + k_idx[None, :] * stride_a_k
        a_tile = tl.load(ptrs_a, mask=(row_idx[:, None] < M) & (k_idx[None, :] < K), other=0.0)
        
        ptrs_b = B_ptr + col_idx[:, None] * stride_b_n + k_idx[None, :] * stride_b_k
        b_tile = tl.load(ptrs_b, mask=(col_idx[:, None] < N) & (k_idx[None, :] < K), other=0.0)
        
        acc = tl.dot(a_tile, b_tile.T, acc)
    
    # Process the final K chunk containing the potential boundary remainders safely via masking
    k_offset = num_k_full * BLOCK_K
    if k_offset < K:
        rem = K - k_offset
        valid_k_idx = k_offset + tl.arange(0, BLOCK_K)
        
        ptrs_a = A_ptr + row_idx[:, None] * stride_a_m + valid_k_idx[None, :] * stride_a_k
        a_tile = tl.load(ptrs_a, mask=(row_idx[:, None] < M) & (valid_k_idx[None, :] < K), other=0.0)
        
        ptrs_b = B_ptr + col_idx[:, None] * stride_b_n + valid_k_idx[None, :] * stride_b_k
        b_tile = tl.load(ptrs_b, mask=(col_idx[:, None] < N) & (valid_k_idx[None, :] < K), other=0.0)
        
        acc = tl.dot(a_tile, b_tile.T, acc)
        
    ptrs_c = C_ptr + row_idx[:, None] * stride_c_m + col_idx[None, :] * stride_c_n
    mask_c = (row_idx[:, None] < M) & (col_idx[None, :] < N)
    tl.store(ptrs_c, acc.to(tl.bfloat16), mask=mask_c)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N)
    )
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )