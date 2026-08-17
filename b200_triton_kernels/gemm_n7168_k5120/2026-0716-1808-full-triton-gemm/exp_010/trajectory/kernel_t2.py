import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
):
    # Explicit Contiguous Element Strides 
    stride_am = 5120
    stride_ak = 1
    stride_bk = 1
    stride_bn = 5120
    stride_cm = 7168
    stride_cn = 1
    
    # Optimized Block Sizes
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    # Define Program Tile Mapping
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Define Row and Column Indices within the Tile
    row_indices = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_indices = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Accumulator Initialized to Zero
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # Iterate Over K Tiles Dimension
    for k in range(0, K, BLOCK_K):
        k_indices = k + tl.arange(0, BLOCK_K)
        
        # Mask Out of Bounds Memory Access Safely with Proper Transpose Patterns
        valid_row = row_indices[:, None] < M
        valid_col = col_indices[None, :] < N
        
        # Load Tile A [BLOCK_M, BLOCK_K] Elements Contiguously
        a = tl.load(a_ptr + row_indices[:, None] * stride_am + k_indices[None, :] * stride_ak,
                    mask=valid_row & (k_indices[None, :] < K),
                    other=0.0)
        
        # Load Transposed Tile B.T [BLOCK_K, BLOCK_N] Elements Contiguously
        b = tl.load(b_ptr + k_indices[:, None] * stride_bk + col_indices[None, :] * stride_bn,
                    mask=valid_col & (k_indices[:, None] < K),
                    other=0.0)
        
        # Perform Batched Dot Product Accumulation
        acc = tl.dot(a, b, acc)
    
    # Final Explicit Conversion to Output Precision
    out = acc.to(tl.bfloat16)
    ptr = c_ptr + row_indices[:, None] * stride_cm + col_indices[None, :] * stride_cn
    tl.store(ptr, out, mask=valid_row & valid_col)


def run(A, B, C):
    """Perform destination-passing GEMM computation C = A @ B.T."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0] # Expected 7168
    K = A.shape[1] # Expected 5120
    
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        num_warps=4,
        num_stages=2,
    )