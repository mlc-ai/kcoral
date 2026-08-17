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
    HEAD_DIM: tl.constexpr,
):
    """
    Optimized FlashAttention-style MHA forward kernel.
    
    Computes O = softmax(Q @ K^T / sqrt(D)) @ V and LSE = log-sum-exp(P)
    where P = Q @ K^T / sqrt(D).
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    
    q_start = pid_q * BLOCK_Q
    bh_offset = pid_bh * S
    
    q_idx = q_start + tl.arange(0, BLOCK_Q)
    d_idx_0 = tl.arange(0, 64)
    d_idx_1 = 64 + tl.arange(0, 64)
    
    # Load our Q slice exactly once, casting to fp32 immediately to avoid dot type mismatches
    Q0 = tl.load(
        Q_ptr + (bh_offset + q_idx[:, None]) * HEAD_DIM + d_idx_0[None, :],
        mask=(q_idx[:, None] < S), other=0.0f
    ).to(tl.float32)
    Q1 = tl.load(
        Q_ptr + (bh_offset + q_idx[:, None]) * HEAD_DIM + d_idx_1[None, :],
        mask=(q_idx[:, None] < S), other=0.0f
    ).to(tl.float32)
    
    # Maintain precise FP32 online-softmax state over the full sequence
    m = tl.full((BLOCK_Q,), -1e20, dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    O_acc_0 = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    O_acc_1 = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    
    # Pipeline all K/V blocks sequentially along the Sequence dimension 
    num_kv_blks = tl.cdiv(S, 64)
    for k_blk in range(num_kv_blks):
        
        K0 = tl.load(
            K_ptr + (bh_offset + (k_blk * 64 + tl.arange(0, 64))[:, None]) * HEAD_DIM + d_idx_0[None, :],
            mask=((k_blk * 64 + tl.arange(0, 64))[:, None]) < S, other=0.0f
        ).to(tl.float32)
        K1 = tl.load(
            K_ptr + (bh_offset + (k_blk * 64 + tl.arange(0, 64))[:, None]) * HEAD_DIM + d_idx_1[None, :],
            mask=((k_blk * 64 + tl.arange(0, 64))[:, None]) < S, other=0.0f
        ).to(tl.float32)
        
        # Accumulate Q @ K^T dot products
        S_block = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
        S_block = tl.dot(Q0, K0.T, acc=S_block)
        S_block = tl.dot(Q1, K1.T, acc=S_block)
        
        # Apply causal scaling
        S_block = S_block * scale
        
        # Masking for partially filled tiles to ensure -inf contribution (zero after exp)
        kv_idx = k_blk * 64 + tl.arange(0, 64)
        mask_kv = (kv_idx < S)
        S_block = tl.where(mask_kv[None, :], S_block, float('-inf'))
        
        cur_m = tl.max(S_block, axis=1)
        m_new = tl.maximum(m, cur_m)
        
        P = tl.exp(S_block - m_new[:, None])
        cur_l = tl.sum(P, axis=1)
        
        # Accurately preserve numerics when updating cumulative statistics mid-loop
        l_new = l * tl.exp(m - m_new) + cur_l
        
        O_acc_0 = O_acc_0 * tl.exp(m - m_new)[:, None]
        O_acc_1 = O_acc_1 * tl.exp(m - m_new)[:, None]
        
        m = m_new
        l = l_new
        
        V0 = tl.load(
            V_ptr + (bh_offset + kv_idx[:, None]) * HEAD_DIM + d_idx_0[None, :],
            mask=(kv_idx[:, None] < S), other=0.0f
        ).to(tl.float32)
        V1 = tl.load(
            V_ptr + (bh_offset + kv_idx[:, None]) * HEAD_DIM + d_idx_1[None, :],
            mask=(kv_idx[:, None] < S), other=0.0f
        ).to(tl.float32)
        
        # Accumulate P @ V dot products
        O_acc_0 = tl.dot(P, V0, acc=O_acc_0)
        O_acc_1 = tl.dot(P, V1, acc=O_acc_1)
        
    # Final division by cumulative scalar sums to resolve correct probabilities
    O_acc_0 = O_acc_0 / l[:, None]
    O_acc_1 = O_acc_1 / l[:, None]
    
    out_ptr_0 = O_ptr + (bh_offset + q_idx[:, None]) * HEAD_DIM + d_idx_0[None, :]
    out_ptr_1 = O_ptr + (bh_offset + q_idx[:, None]) * HEAD_DIM + d_idx_1[None, :]
    
    mask_o = (q_idx[:, None] < S)
    tl.store(out_ptr_0, O_acc_0.to(tl.bfloat16), mask=mask_o)
    tl.store(out_ptr_1, O_acc_1.to(tl.bfloat16), mask=mask_o)
    
    # Emit natural logarithm directly matching standard PyTorch SDPA conventions
    lse_ptr = LSE_ptr + bh_offset + q_idx
    lse = m + tl.log(l)
    tl.store(lse_ptr, lse, mask=(q_idx < S))


def run(Q, K, V, O, LSE):
    """Compute Multi-Head Attention O and Log-Sum-Exp LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H_dim, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    # Use exactly 128 rows per block and 4 active warps 
    grid = (B * H_dim, triton.cdiv(S, 128))
    
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        S,
        scale,
        BLOCK_Q=128,
        HEAD_DIM=128,
        num_warps=4,
        num_stages=4,
    )