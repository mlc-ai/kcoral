import math
import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    S,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H: tl.constexpr,
):
    """
    Optimized FlashAttention-style MHA forward kernel.
    
    Computes O = softmax(Q @ K^T / sqrt(D)) @ V and LSE = log-sum-exp(P)
    where P = Q @ K^T / sqrt(D).
    
    Each warp processes a unique (batch, head) pair and a block of BLOCK_Q query rows,
    iterating sequentially over all BLOCK_KV-sized key/value blocks.
    """
    # Identify our (batch, head) slice and our chunk of Q rows
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    
    q_start = pid_q * BLOCK_Q
    batch_head_offset = pid_bh * S
    
    # Precompute transposed Index tensors for coalesced loads across the D-mode span
    q_idx = q_start + tl.arange(0, BLOCK_Q)
    d_idx = tl.arange(0, BLOCK_D)
    
    # Load our Q slice exactly once. Shape is conceptually [BLOCK_Q, BLOCK_D].
    Q_tile = tl.load(
        Q_ptr + (batch_head_offset + q_idx[:, None]) * HEAD_DIM + d_idx[None, :],
        mask=(q_idx[:, None] < S) & (d_idx[None, :] < HEAD_DIM),
        other=0.0
    )
    
    # Maintain precise FP32 online-softmax state over the full sequence
    m = tl.full((BLOCK_Q, 1), -1e20, dtype=tl.float32)
    l = tl.zeros((BLOCK_Q, 1), dtype=tl.float32)
    O_acc_0 = tl.zeros((BLOCK_Q, BLOCK_Q), dtype=tl.float32)
    O_acc_1 = tl.zeros((BLOCK_Q, BLOCK_Q), dtype=tl.float32)
    
    # Sequentially pipeline all K/V blocks. 
    # Because this is non-causal MHA, every Q block attends to every K/V block.
    for kv_idx_base in range(0, S, BLOCK_KV):
        kv_idx = kv_idx_base + tl.arange(0, BLOCK_KV)
        
        K_tile = tl.load(
            K_ptr + (batch_head_offset + kv_idx[:, None]) * HEAD_DIM + d_idx[None, :],
            mask=(kv_idx[:, None] < S) & (d_idx[None, :] < HEAD_DIM),
            other=0.0
        )
        
        # Utilize inner 64x64 sub-block iterations to satisfy WGMMA rank expectations
        S_block = tl.zeros((BLOCK_Q, BLOCK_KV), dtype=tl.float32)
        
        Q0 = Q_tile[:, :64]
        Q1 = Q_tile[:, 64:]
        K0 = K_tile[:, :64]
        K1 = K_tile[:, 64:]
        
        S_block = tl.dot(Q0, K0.T, S_block)
        S_block = tl.dot(Q1, K1.T, S_block)
        
        # Standard FlashAttention maximum tracking & exponentiation logic
        S_block = S_block * scale
        cur_m = tl.max(S_block, axis=1, keep_dims=True)
        m_new = tl.maximum(m, cur_m)
        
        P = tl.exp(S_block - m_new)
        cur_l = tl.sum(P, axis=1, keep_dims=True)
        
        # Accurately preserve numerics when updating cumulative statistics mid-loop
        l_new = l * tl.exp(m - m_new) + cur_l
        
        O_acc_0 = O_acc_0 * tl.exp(m - m_new)
        O_acc_1 = O_acc_1 * tl.exp(m - m_new)
        
        m = m_new
        l = l_new
        
        V_tile = tl.load(
            V_ptr + (batch_head_offset + kv_idx[:, None]) * HEAD_DIM + d_idx[None, :],
            mask=(kv_idx[:, None] < S) & (d_idx[None, :] < HEAD_DIM),
            other=0.0
        )
        
        V0 = V_tile[:, :64]
        V1 = V_tile[:, 64:]
        
        # Accumulate directly against previously rescaled carry-over state
        O_acc_0 = tl.dot(P, V0, O_acc_0)
        O_acc_1 = tl.dot(P, V1, O_acc_1)
        
    # Final division by cumulative scalar sums to resolve correct probabilities
    O_acc_0 = O_acc_0 / l
    O_acc_1 = O_acc_1 / l
    
    out_ptr_0 = O_ptr + (batch_head_offset + q_idx[:, None]) * HEAD_DIM + d_idx[None, :64]
    out_ptr_1 = O_ptr + (batch_head_offset + q_idx[:, None]) * HEAD_DIM + d_idx[None, 64:]
    
    o_0 = O_acc_0.to(tl.bfloat16)
    o_1 = O_acc_1.to(tl.bfloat16)
    
    mask_o = (q_idx[:, None] < S)
    tl.store(out_ptr_0, o_0, mask=mask_o)
    tl.store(out_ptr_1, o_1, mask=mask_o)
    
    # Emit natural logarithm directly matching standard PyTorch SDPA conventions
    lse_ptr = LSE_ptr + batch_head_offset + q_idx
    lse = m[:, 0] + tl.log(l[:, 0])
    tl.store(lse_ptr, lse, mask=(q_idx < S))


def run(Q, K, V, O, LSE):
    """Compute Multi-Head Attention O and Log-Sum-Exp LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H_dim, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    # Use exactly 64 rows per block and 4 active warps (matching 64x64 WGMMA architecture)
    grid = (B * H_dim, triton.cdiv(S, 64))
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        S,
        scale,
        BLOCK_Q=64,
        BLOCK_KV=64,
        BLOCK_D=128,
        HEAD_DIM=128,
        H=H_dim,
        num_warps=4,
        num_stages=3,
    )