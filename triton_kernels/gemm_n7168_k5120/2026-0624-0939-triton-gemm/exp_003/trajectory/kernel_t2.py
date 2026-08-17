import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Standard row-major tiled GEMM computing C = A @ B.T.
    Utilizes direct global memory pointers with explicit masks and 
    automatic loop staging to maximize execution density.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Establish base coordinate offsets for the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_iters = K // BLOCK_K
    
    # Iterate over the K dimension. Staged to overlap load and dot instructions.
    for k_idx in tl.range(0, num_k_iters, num_stages=3):
        k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Load A tile shaped [BLOCK_M, BLOCK_K] corresponding to [M, K]
        a_tile = tl.load(
            A_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0
        )
        
        # Load B tile shaped [BLOCK_K, BLOCK_N] corresponding to B^T[K, N]
        b_tile = tl.load(
            B_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )
        
        acc = tl.dot(a_tile, b_tile, acc)
    
    # Convert the FP32 accumulation into the required bfloat16 format
    c = acc.to(tl.bfloat16)
    
    # Persist the result to global memory safely
    out_ptr = C_ptr + offs_m[:, None] * stride_bk + offs_n[None, :] * stride_bn
    tl.store(out_ptr, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    # 64x64x64 provides excellent occupancy and arithmetic intensity balance
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Pass physical element strides explicitly
    _gemm_kernel[grid](
        A, B, C, M, N, K,
        stride_am=K, stride_ak=1,
        stride_bk=1, stride_bn=K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )