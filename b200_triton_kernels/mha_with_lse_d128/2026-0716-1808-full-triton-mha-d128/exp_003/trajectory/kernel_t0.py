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
    BLOCK_D: tl.constexpr,
):
    q_blk = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    # Linearize batch and head indices to navigate the [B, H, S, D] layout
    bh_idx = b_idx * 48 + h_idx
    
    q_offs_0 = tl.arange(0, BLOCK_K)
    d_offs_0 = tl.arange(0, BLOCK_D)
    
    # Accumulators
    out_acc = tl.zeros((BLOCK_D, BLOCK_K), tl.float32)
    m_prev = tl.full((BLOCK_K,), -float('inf'), tl.float32)
    sum_prev = tl.full((BLOCK_K,), 0.0, tl.float32)
    
    base_ptr_Q = Q
    base_ptr_K = K
    base_ptr_V = V
    
    num_blocks = (S_val + BLOCK_K - 1) // BLOCK_K
    
    # Load Q only once. It does not change across K blocks.
    ptr_q = base_ptr_Q + bh_idx * (S_val * 128) + q_blk * BLOCK_K * 128 + q_offs_0[:, None] * 128 + d_offs_0[None, :]
    q = tl.load(ptr_q, mask=(q_blk * BLOCK_K + q_offs_0[:, None]) < S_val, other=0.0)
    
    for k_blk in range(num_blocks):
        k_offs_0 = tl.arange(0, BLOCK_K)
        
        # Load contiguous [BLOCK_K, BLOCK_D] chunks of K and V.
        ptr_k = base_ptr_K + bh_idx * (S_val * 128) + k_blk * BLOCK_K * 128 + k_offs_0[:, None] * 128 + d_offs_0[None, :]
        k = tl.load(ptr_k, mask=(k_blk * BLOCK_K + k_offs_0[:, None]) < S_val, other=0.0)
        
        ptr_v = base_ptr_V + bh_idx * (S_val * 128) + k_blk * BLOCK_K * 128 + k_offs_0[:, None] * 128 + d_offs_0[None, :]
        v = tl.load(ptr_v, mask=(k_blk * BLOCK_K + k_offs_0[:, None]) < S_val, other=0.0)
        
        # Compute P = Q @ K^T / sqrt(D).
        # Shapes: q [BLOCK_D, BLOCK_K], k [BLOCK_D, BLOCK_K]
        p = q[:, None] * k[None, :] 
        p = tl.sum(p, axis=0) / scale
        
        # Mask out-of-bounds sequence positions in P
        seq_mask = ((k_blk * BLOCK_K + k_offs_0[None, :]) < S_val) & \
                   ((q_blk * BLOCK_K + q_offs_0[:, None]) < S_val)
        p = tl.where(seq_mask, p, -float('inf'))
        
        # Numerically stable online softmax (FlashAttention style)
        m_i = tl.max(p, axis=0, keep_dims=True)
        m_new = tl.maximum(m_prev[:, None], m_i)
        
        exp_block = tl.exp(m_prev[:, None] - m_new)
        
        p = p - m_prev[:, None]
        local_sum = tl.sum(tl.exp(p), axis=0, keep_dims=True)
        new_sum = local_sum * exp_block
        
        q = p  # reuse q for exp(P_i - M_{i-1})
        
        # Scale previously accumulated context by the change in max
        out_acc *= exp_block[None, :]
        
        # Multiply by V^T and accumulate into context.
        # v is [BLOCK_K, BLOCK_D], so v.T is [BLOCK_D, BLOCK_K].
        v_T = k  # reuse dead register k for holding V transposed
        v_T = tl.load(ptr_v, mask=(k_blk * BLOCK_K + k_offs_0[:, None]) < S_val, other=0.0)
        v_T = v_T.T
        
        out_acc += tl.sum((q * v_T.T), axis=1)
        
        m_prev = m_new[:, 0]
        sum_prev = new_sum[:, 0]
        
    # Output division and cast to bf16
    out_acc = out_acc / sum_prev[None, :]
    
    ptr_o = O + bh_idx * (S_val * 128) + q_blk * BLOCK_K * 128 + q_offs_0[:, None] * 128 + d_offs_0[None, :]
    tl.store(ptr_o, out_acc.T.to(tl.bfloat16), mask=(q_blk * BLOCK_K + q_offs_0[:, None]) < S_val)
    
    # Log-Sum-Exp
    lse = m_prev + tl.log(sum_prev)
    ptr_lse = LSE + bh_idx * S_val + q_blk * BLOCK_K + q_offs_0
    tl.store(ptr_lse, lse, mask=(q_blk * BLOCK_K + q_offs_0) < S_val)


def run(Q, K, V, O, LSE):
    """Compute non-causal Multi-Head Attention and output Log-Sum-Exp."""
    torch.cuda.set_device(Q.device)
    B, H, S_val, D = Q.shape
    scale = 1.0 / math.sqrt(128)

    grid = (triton.cdiv(S_val, 64), H, B)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S_val, scale,
        BLOCK_K=64, BLOCK_D=128,
        num_warps=4, num_stages=3
    )