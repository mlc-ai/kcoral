import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _persistent_gemm_row_major(
    a_desc, b_desc, C_ptr,
    M, N, K,
    stride_Cm, stride_Cn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    start_pid = tl.program_id(0)
    m_offset = start_pid * BLOCK_M
    
    # Explicit shared-memory staging preserves A blocks across cheaper inner loops
    a_s = tl.shared.empty((2, BLOCK_M, BLOCK_K), dtype=tl.bfloat16)
    b_s = tl.shared.empty((BLOCK_N, BLOCK_K), dtype=tl.bfloat16)
    
    # We encode thread-local symbols matching our block footprint
    m_idx = start_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    k_idx = tl.arange(0, BLOCK_K)
    
    # Initially stage the first K-slice of A
    a_s[0, m_idx, k_idx] = a_desc.load([m_offset, 0])
    
    num_N_tiles = tl.cdiv(N, BLOCK_N)
    
    for n_idx in range(num_N_tiles):
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        group_id = n_idx // GROUP_M
        first_n = group_id * GROUP_M
        n_in_group = n_idx % GROUP_M
        pid_n = first_n + n_in_group
        n_offset = pid_n * BLOCK_N
        
        a_stage = 0
        for k_idx_iter in range(0, K, BLOCK_K):
            k_next = k_idx_iter + BLOCK_K
            k_idx_curr = tl.arange(0, BLOCK_K)
            
            # Pipelined descriptor load targeting the opposite shared-memory stratum
            if k_next < K:
                a_s[1 - a_stage, m_idx, k_idx_curr] = a_desc.load([m_offset, k_next])
            
            n_idx = tl.arange(0, BLOCK_N)
            b_s[n_idx, k_idx_curr] = b_desc.load([n_offset, k_idx_iter])
            
            a = a_s[a_stage, m_idx, k_idx_curr]
            b = b_s[n_idx, k_idx_curr]
            acc = tl.dot(a, b.T, acc)
            
            a_stage = 1 - a_stage
        
        last_n_idx = n_idx

    offs_n = (last_n_idx * BLOCK_N) + tl.arange(0, BLOCK_N)
    out_ptr = C_ptr + m_idx[:, None] * stride_Cm + offs_n[None, :] * stride_Cn
    tl.store(out_ptr, acc.to(tl.bfloat16))


def run(A, B, C):
    """Compute ``C = A @ B.T`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(A.device)
    
    M, K = A.shape
    N = B.shape[0]
    
    assert K == 5120, f"Unsupported K={K}"
    assert N == 7168, f"Unsupported N={N}"
    
    block_m = 64
    block_n = 128
    block_k = 64
    
    NUM_SMS = 132
    GROUP_M = 8
    
    num_pid_m = triton.cdiv(M, block_m)
    
    grid = (num_pid_m,)
    
    a_desc = TensorDescriptor.from_tensor(A, [block_m, block_k])
    b_desc = TensorDescriptor.from_tensor(B, [block_n, block_k])
    
    _persistent_gemm_row_major[grid](
        a_desc, b_desc, C, M, N, K,
        N, 1,  
        NUM_SMS=NUM_SMS,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        GROUP_M=GROUP_M,
        num_warps=8,
        num_stages=4,
    )