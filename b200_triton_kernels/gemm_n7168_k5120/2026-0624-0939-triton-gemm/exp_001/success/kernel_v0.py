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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate base offsets for this program's output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_k_steps = tl.cdiv(K, BLOCK_K)
    for k_step in range(num_k_steps):
        offs_k = k_step * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Load A tile [BLOCK_M, BLOCK_K]
        ptrs_a = A_ptr + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_tile = tl.load(ptrs_a, mask=mask_a, other=0.0)
        
        # Load B tile [BLOCK_N, BLOCK_K]
        ptrs_b = B_ptr + offs_n[:, None] * stride_b_n + offs_k[None, :] * stride_b_k
        mask_b = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b_tile = tl.load(ptrs_b, mask=mask_b, other=0.0)
        
        # Compute partial dot product, accumulating into `acc`
        # B is physically [N, K], so b_tile.T gives the required [K, N] operand
        acc = tl.dot(a_tile, b_tile.T, acc)
    
    # Convert the fp32 accumulator to bfloat16 and store the result
    ptrs_c = C_ptr + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(ptrs_c, acc.to(tl.bfloat16), mask=mask_c)


def run(A, B, C):
    """Compute C = A @ B.T into the preallocated output tensor C."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N = B.shape[0]
    K = A.shape[1]
    
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _gemm_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=128,
        num_warps=4, num_stages=3,
    )