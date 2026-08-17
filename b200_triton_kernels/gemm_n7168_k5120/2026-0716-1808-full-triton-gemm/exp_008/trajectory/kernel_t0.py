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
    Implements C = A @ B^T on a 2D grid. Each program instance computes one 
    [BLOCK_M, BLOCK_N] tile of the output and completely reduces across the 
    K dimension internally, accumulating in FP32.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Linear indices for the rows (M dimension) and columns (N dimension)
    row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = row_idx < M
    
    col_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # Iterating sequentially through the chunks of 64 elements along the K dimension.
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_idx = k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Load A tile
        ptr_A = A + row_idx[:, None] * stride_A_m + k_idx[None, :] * stride_A_k
        a = tl.load(ptr_A, mask=mask_m[:, None], other=0.0)
        
        # Load B tile
        ptr_B = B + k_idx[:, None] * stride_B_k + col_idx[None, :] * stride_B_n
        b = tl.load(ptr_B, mask=None, other=0.0)
        
        # Pass `b.T` so that Triton correctly executes the underlying tensor core instructions 
        # for multiplying [BLOCK_M, BLOCK_K] by [BLOCK_K, BLOCK_N] to get [BLOCK_M, BLOCK_N].
        acc = tl.dot(a, b.T, acc)
    
    ptr_C = C + row_idx[:, None] * stride_C_m + col_idx[None, :] * stride_C_n
    tl.store(ptr_C, acc.to(tl.bfloat16), mask=mask_m[:, None])


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
        BLOCK_K=64,
        num_warps=4, 
        num_stages=3
    )