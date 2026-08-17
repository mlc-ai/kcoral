import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A,
    B,
    C,
    M,
    N_out,
    K,
    stride_A_m,
    stride_A_k,
    stride_B_n,
    stride_B_k,
    stride_C_m,
    stride_C_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Implements C = A @ B^T on a 2D grid. Features double-buffered loads pipelined 
    across K iterations and Hopper tensor core instructions.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # Double buffered arrays for A and B along the K dimension
    a = tl.empty((2, BLOCK_M, BLOCK_K), dtype=tl.bfloat16)
    b = tl.empty((2, BLOCK_K, BLOCK_N), dtype=tl.bfloat16)
    
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    
    # Prologue: load the first chunk of A and B asynchronously
    if num_k_tiles > 0:
        k_idx = tl.arange(0, BLOCK_K)
        
        mask_A = (row_idx[:, None] < M) & (k_idx[None, :] < K)
        ptr_A = A + row_idx[:, None] * stride_A_m + k_idx[None, :] * stride_A_k
        a[0] = tl.load(ptr_A, mask=mask_A, other=0.0)
        torch.dlcp_async_copy(a[0], ptr_A, mask_A)
        
        mask_B = (k_idx[:, None] < K) & (col_idx[None, :] < N_out)
        ptr_B = B + k_idx[:, None] * stride_B_k + col_idx[None, :] * stride_B_n
        b[0] = tl.load(ptr_B, mask=mask_B, other=0.0)
        torch.dlcp_async_copy(b[0], ptr_B, mask_B)
        
    for k in range(0, num_k_tiles):
        idx = k % 2
        next_idx = (k + 1) % 2
        
        if k + 2 < num_k_tiles:
            next_next_idx = (k + 2) % 2
            
            # Software pipeline: prefetch the loads for iteration k+2 whilst processing k
            next_k_idx = (k + 1) * BLOCK_K + tl.arange(0, BLOCK_K)
            
            next_mask_A = (row_idx[:, None] < M) & (next_k_idx[None, :] < K)
            next_ptr_A = A + row_idx[:, None] * stride_A_m + next_k_idx[None, :] * stride_A_k
            a[next_next_idx] = tl.load(next_ptr_A, mask=next_mask_A, other=0.0)
            torch.dlcp_async_copy(a[next_next_idx], next_ptr_A, next_mask_A)
            
            next_mask_B = (next_k_idx[:, None] < K) & (col_idx[None, :] < N_out)
            next_ptr_B = B + next_k_idx[:, None] * stride_B_k + col_idx[None, :] * stride_B_n
            b[next_next_idx] = tl.load(next_ptr_B, mask=next_mask_B, other=0.0)
            torch.dlcp_async_copy(b[next_next_idx], next_ptr_B, next_mask_B)
            
        # Wait for current buffer to be ready and execute dot product accumulation
        # This effectively acts as a sync point between the two buffers in the pipeline loop
        acc = tl.dot(a[idx], b[idx], acc)
            
    
    mask_m = row_idx < M
    mask_n = col_idx < N_out
    ptr_C = C + row_idx[:, None] * stride_C_m + col_idx[None, :] * stride_C_n
    tl.store(ptr_C, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def run(A, B, C):
    """Compute ``C = A @ B.T`` into the preallocated tensor C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N_out = B.shape[0]
    K = B.shape[1]
    
    grid = (triton.cdiv(M, 128), triton.cdiv(N_out, 128))
    _gemm_kernel[grid](
        A, 
        B, 
        C, 
        M, N_out, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=128, 
        BLOCK_N=128, 
        BLOCK_K=128,
        num_warps=8, 
        num_stages=3
    )