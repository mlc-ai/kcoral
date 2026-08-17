import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_Am, stride_Ak,
    stride_Bn, stride_Bk,
    stride_Cm, stride_Cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Standard 2D-tiled GEMM kernel. Computes C[M, N] = A[M, K] @ B.T[K, N]
    where B is physically stored as [N, K].
    Accumulates in float32 and converts to bfloat16 for the output.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Base offsets for the M and N output tiles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    # Iterate over the K dimension
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Load A tile of shape [BLOCK_M, BLOCK_K]
        a_ptr = A_ptr + offs_m[:, None] * stride_Am + k[None, :] * stride_Ak
        a = tl.load(a_ptr, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        
        # Load B tile of shape [BLOCK_N, BLOCK_K] (stored as [N, K])
        b_ptr = B_ptr + offs_n[:, None] * stride_Bn + k[None, :] * stride_Bk
        b = tl.load(b_ptr, mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0.0)
        
        # Compute dot product. b.T converts [BLOCK_N, BLOCK_K] to [BLOCK_K, BLOCK_N]
        acc = tl.dot(a, b.T, acc)
        
    # Convert accumulator to output dtype and store the result
    out_ptr = C_ptr + offs_m[:, None] * stride_Cm + offs_n[None, :] * stride_Cn
    out = acc.to(tl.bfloat16)
    tl.store(out_ptr, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    block_m = 64
    block_n = 64
    block_k = 64
    
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    
    _gemm_kernel[grid](
        A, B, C, M, N, K,
        K, 1,  # stride_Am, stride_Ak
        K, 1,  # stride_Bn, stride_Bk
        N, 1,  # stride_Cm, stride_Cn
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4,
    )