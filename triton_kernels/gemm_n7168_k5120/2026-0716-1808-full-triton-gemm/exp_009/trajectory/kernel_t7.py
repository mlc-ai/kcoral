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
    
    m_offset = pid_m * BLOCK_M
    n_offset = pid_n * BLOCK_N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    m_vec = m_offset + tl.arange(0, BLOCK_M)
    n_vec = n_offset + tl.arange(0, BLOCK_N)
    
    for k_idx in range(0, triton.cdiv(K_dim, BLOCK_K)):
        k_vec = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        
        offsets_a = A_ptr + m_vec[:, None] * K_dim + k_vec[None, :] * 1
        mask_a = (m_vec[:, None] < M) & (k_vec[None, :] < K_dim)
        a_tile = tl.load(offsets_a, mask=mask_a, other=0.0)
        
        offsets_b = B_ptr + n_vec[None, :] * K_dim + k_vec[:, None] * 1
        mask_b = (n_vec[None, :] < N_dim) & (k_vec[:, None] < K_dim)
        b_tile = tl.load(offsets_b, mask=mask_b, other=0.0)
        
        acc = tl.dot(a_tile, b_tile, acc)
    
    offsets_c = C_ptr + m_vec[:, None] * N_dim + n_vec[None, :] * 1
    mask_c = (m_vec[:, None] < M) & (n_vec[None, :] < N_dim)
    tl.store(offsets_c, acc.to(tl.bfloat16), mask=mask_c)


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    M = A.shape[0]
    N_dim = B.shape[0] 
    K_dim = B.shape[1]
    
    block_m = 256
    block_n = 512
    block_k = 128
    
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
        num_warps=8,
        num_stages=4,
    )