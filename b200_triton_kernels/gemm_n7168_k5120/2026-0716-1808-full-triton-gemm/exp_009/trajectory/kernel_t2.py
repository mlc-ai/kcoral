import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N_dim,
    K_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k0 in range(0, triton.cdiv(K_dim, BLOCK_K)):
        k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = (offs_m[:, None] < M) & (k[None, :] < K_dim)
        a_tile = tl.load(A_ptr + offs_m[:, None] * K_dim + k[None, :] * 1, mask=mask_m, other=0.0f)
        
        # Load B.T directly by constructing the pointer as [BLOCK_K, BLOCK_N]
        mask_n = (offs_n[None, :] < N_dim) & (k[:, None] < K_dim)
        b_tile = tl.load(B_ptr + offs_n[None, :] * K_dim + k[:, None] * 1, mask=mask_n, other=0.0f)
        
        acc = tl.dot(a_tile, b_tile, acc)

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N_dim)
    out_ptr = C_ptr + offs_m[:, None] * N_dim + offs_n[None, :] * 1
    tl.store(out_ptr, acc.to(tl.bfloat16), mask=mask_out)


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N_dim = B.shape[0] 
    K_dim = B.shape[1] 
    
    block_m = 128
    block_n = 128
    block_k = 64
    
    grid = (triton.cdiv(M, block_m), triton.cdiv(N_dim, block_n))
    
    _gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N_dim,
        K_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )