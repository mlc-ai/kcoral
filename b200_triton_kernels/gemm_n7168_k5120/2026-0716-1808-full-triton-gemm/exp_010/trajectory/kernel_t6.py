import torch
import triton
import triton.language as tl


@triton.jit(reduction_loop_unrolls=2)
def _gemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    __launch_bounds__(128, 1)
    
    # Ensure inner loops iterate using constant steps matching tile sizing
    stride_A_m = A.shape[1]
    stride_A_k = 1
    stride_B_k = 1
    stride_B_n = B.shape[1]
    
    stride_C_m = C.shape[1]
    stride_C_n = 1
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Vectors containing absolute offsets for this CTA's chunk of work
    row_indices = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_indices = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # Linear scan across the contiguous reduction dimension 
    for k_tile in range(K // BLOCK_K):
        k_offset = k_tile * BLOCK_K
        k_indices = k_offset + tl.arange(0, BLOCK_K)
        
        valid_row = row_indices[:, None] < M
        valid_k = k_indices[None, :] < K
        
        # Load Tile A ([BLOCK_M, BLOCK_K]) Contiguously along K
        a = tl.load(A + row_indices[:, None] * stride_A_m + k_indices[None, :] * stride_A_k,
                    mask=valid_row & valid_k, other=0.0)
        
        valid_col = col_indices[None, :] < N
        valid_k_T = k_indices[None, :] < K
        
        # Load Tile B_T ([BLOCK_K, BLOCK_N]) Contiguously along Strided Dimension N
        b = tl.load(B + k_indices[:, None] * stride_B_k + col_indices[None, :] * stride_B_n,
                    mask=valid_col & valid_k_T, other=0.0)
        
        acc = tl.dot(a, b, acc)
    
    # Final Explicit Conversion to Output Precision
    out = acc.to(tl.bfloat16)
    ptr = C + row_indices[:, None] * stride_C_m + col_indices[None, :] * stride_C_n
    tl.store(ptr, out, mask=(valid_row & valid_col))


def run(A, B, C):
    """Perform destination-passing GEMM computation C = A @ B.T."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0] 
    K = A.shape[1]
    
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    
    _gemm_kernel[grid](A, B, C, M, N, K, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_warps=4)