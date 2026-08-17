import torch
import triton
import triton.language as tl
import math


@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    S_val,
    scale,
    BLOCK_K: tl.constexpr,
):
    q_blk = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    # Linearize batch and head indices to navigate the [B, H, S, D] layout
    bh_idx = b_idx * 48 + h_idx
    
    q_offs = tl.arange(0, BLOCK_K)
    k_offs = tl.arange(0, BLOCK_K)
    
    # Split the head dimension D=128 into two strictly 64-element chunks. 
    # This eliminates the 64 vs 128 shape mismatch and aligns perfectly with WGMMA tiles.
    d_offs0 = tl.arange(0, 64)
    d_offs1 = tl.arange(64, 128)
    
    out_acc0 = tl.zeros((BLOCK_K, 64), tl.float32)
    out_acc1 = tl.zeros((BLOCK_K, 64), tl.float32)
    
    m_prev = tl.full((BLOCK_K,), -float('inf'), tl.float32)
    sum_prev = tl.full((BLOCK_K,), 0.0, tl.float32)
    
    base_ptr_Q = Q
    base_ptr_K = K
    base_ptr_V = V
    
    num_blocks = (S_val + BLOCK_K - 1) // BLOCK_K
    
    # Load Q only once. It does not change across K blocks.
    ptr_q0 = base_ptr_Q + bh_idx * (S_val * 128) + q_blk * BLOCK_K * 128 + q_offs[:, None] * 128 + d_offs0[None, :]
    q0 = tl.load(ptr_q0, mask=(q_blk * BLOCK_K + q_offs[:, None]) < S_val, other=0.0)
    
    ptr_q1 = base_ptr_Q + bh_idx * (S_val * 128) + q_blk * BLOCK_K * 128 + 64 + q_offs[:, None] * 128 + d_offs1[None, :]
    q1 = tl.load(ptr_q1, mask=(q_blk * BLOCK_K + q_offs[:, None]) < S_val, other=0.0)
    
    q_mask = (q_blk * BLOCK_K + q_offs) < S_val
    
    for k_blk in range(num_blocks):
        # Load contiguous [BLOCK_K, 64] chunks of K and V.
        ptr_k0 = base_ptr_K + bh_idx * (S_val * 128) + k_blk * BLOCK_K * 128 + k_offs[:, None] * 128 + d_offs0[None, :]
        k0 = tl.load(ptr_k0, mask=(k_blk * BLOCK_K + k_offs[:, None]) < S_val, other=0.0)
        
        ptr_k1 = base_ptr_K + bh_idx * (S_val * 128) + k_blk * BLOCK_K * 128 + 64 + k_offs[:, None] * 128 + d_offs1[None, :]
        k1 = tl.load(ptr_k1, mask=(k_blk * BLOCK_K + k_offs[:, None]) < S_val, other=0.0)
        
        ptr_v0 = base_ptr_V + bh_idx * (S_val * 128) + k_blk * BLOCK_K * 128 + k_offs[:, None] * 128 + d_offs0[None, :]
        v0 = tl.load(ptr_v0, mask=(k_blk * BLOCK_K + k_offs[:, None]) < S_val, other=0.0)
        
        ptr_v1 = base_ptr_V + bh_idx * (S_val * 128) + k_blk * BLOCK_K * 128 + 64 + k_offs[:, None] * 128 + d_offs1[None, :]
        v1 = tl.load(ptr_v1, mask=(k_blk * BLOCK_K + k_offs[:, None]) < S_val, other=0.0)
        
        # Compute P = Q @ K^T / sqrt(D).
        p = tl.zeros((BLOCK_K, BLOCK_K), tl.float32)
        
        # Explicit contraction loops over the split D dimension halves ensures layout compatibility and prevents dynamic indexing issues.
        for i in range(64):
            p += q0[:, None] * k0[None, :, i]
            p += q1[:, None] * k1[None, :, i]
            
        p = p / scale
        
        # Mask out-of-bounds sequence positions in P
        k_mask = (k_blk * BLOCK_K + k_offs) < S_val
        seq_mask = q_mask[:, None] & k_mask[None, :]
        p = tl.where(seq_mask, p, -float('inf'))
        
        # Numerically stable online softmax (FlashAttention style)
        m_i = tl.max(p, axis=1, keep_dims=True)
        m_new = tl.maximum(m_prev[:, None], m_i)
        
        exp_block = tl.exp(m_prev[:, None] - m_new)
        
        # Rescale previously aggregated sum and weights with the new global maximum block
        sum_prev = sum_prev * exp_block[:, 0]
        
        p = p - m_new
        exp_p = tl.exp(p)
        
        new_sum = tl.sum(exp_p, axis=1, keep_dims=True)
        sum_prev += new_sum[:, 0]
        
        # Scale accumulated context by the change in max
        out_acc0 *= exp_block[None, :]
        out_acc1 *= exp_block[None, :]
        
        # Multiply by V and accumulate into context.
        # Mapping V directly along its feature rows bypasses the need for transpositions that conflict with TMA layout.
        s = exp_p
        
        for j in range(64):
            out_acc0 += s[:, j] * v0[j, :]
            out_acc1 += s[:, j] * v1[j, :]
            
        m_prev = m_new[:, 0]
        
    # Output division and cast to bf16
    out_acc0 = out_acc0 / sum_prev[None, :]
    out_acc1 = out_acc1 / sum_prev[None, :]
    
    ptr_o0 = O + bh_idx * (S_val * 128) + q_blk * BLOCK_K * 128 + q_offs[:, None] * 128 + d_offs0[None, :]
    tl.store(ptr_o0, out_acc0.to(tl.bfloat16), mask=q_mask[:, None])
    
    ptr_o1 = O + bh_idx * (S_val * 128) + q_blk * BLOCK_K * 128 + 64 + q_offs[:, None] * 128 + d_offs1[None, :]
    tl.store(ptr_o1, out_acc1.to(tl.bfloat16), mask=q_mask[:, None])
    
    # Log-Sum-Exp
    lse = m_prev + tl.log(sum_prev)
    ptr_lse = LSE + bh_idx * S_val + q_blk * BLOCK_K + q_offs
    tl.store(ptr_lse, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    """Compute non-causal Multi-Head Attention and output Log-Sum-Exp."""
    torch.cuda.set_device(Q.device)
    B, H, S_val, D = Q.shape
    scale = 1.0 / math.sqrt(128)

    grid = (triton.cdiv(S_val, 64), H, B)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S_val, scale,
        BLOCK_K=64,
        num_warps=4, num_stages=3
    )