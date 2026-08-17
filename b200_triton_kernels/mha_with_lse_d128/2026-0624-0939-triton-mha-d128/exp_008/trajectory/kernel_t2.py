import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q, K, V, O, LSE,
    B, H, S,
    scale,
    stride_Q_b, stride_Q_h, stride_Q_s,
    stride_K_b, stride_K_h, stride_K_s,
    stride_V_b, stride_V_h, stride_V_s,
    stride_O_b, stride_O_h, stride_O_s,
    stride_LSE_b, stride_LSE_h, stride_LSE_s,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    FlashAttention-style MHA kernel for (B, H, S, D) tensors.
    Each program processes one (b, h, q_idx) tile of size BLOCK_Q x D.
    Iterates over all N-blocks of K and V to accumulate the output.
    """
    pid = tl.program_id(0)
    num_blocks_q = tl.cdiv(S, BLOCK_Q)
    head_idx = pid // num_blocks_q
    b = head_idx // H
    h = head_idx % H
    q_block = pid % num_blocks_q
    q_idx = q_block * BLOCK_Q
    
    num_blocks_n = tl.cdiv(S, BLOCK_N)
    
    batch_offset = b * stride_Q_b
    head_offset = h * stride_Q_h
    
    # Load Q tile [BLOCK_Q, D] (split into two column-halves)
    q_row_offs = (q_idx + tl.arange(0, BLOCK_Q))[:, None]
    q_offs_l = batch_offset + head_offset + q_row_offs * stride_Q_s + tl.arange(0, 64)[None, :]
    q_mask_l = (q_row_offs.squeeze() < S)[:, None]
    q_tile_l = tl.load(Q + q_offs_l, mask=q_mask_l, other=0.0)
    
    q_offs_r = batch_offset + head_offset + q_row_offs * stride_Q_s + 64 + tl.arange(0, 64)[None, :]
    q_tile_r = tl.load(Q + q_offs_r, mask=q_mask_l, other=0.0)
    
    o_l = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    o_r = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    
    for n_block in range(num_blocks_n):
        n_idx = n_block * BLOCK_N
        n_row_offs = (n_idx + tl.arange(0, BLOCK_N))[:, None]
        
        # Load K tile and split into two column-halves
        k_offs_l = batch_offset + head_offset + n_row_offs * stride_K_s + tl.arange(0, 64)[None, :]
        k_mask = (n_row_offs.squeeze() < S)[:, None]
        k_tile_l = tl.load(K + k_offs_l, mask=k_mask, other=0.0)
        
        k_offs_r = batch_offset + head_offset + n_row_offs * stride_K_s + 64 + tl.arange(0, 64)[None, :]
        k_tile_r = tl.load(K + k_offs_r, mask=k_mask, other=0.0)
        
        # SS-GEMM: Q @ K^T
        s = tl.zeros((BLOCK_Q, BLOCK_N), dtype=tl.float32)
        s = tl.dot(q_tile_l, k_tile_l.T, s)
        s = tl.dot(q_tile_r, k_tile_r.T, s)
        
        s = s * scale
        
        m_old = m
        m = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m)
        
        l_old = l
        l = l_old * tl.exp(m_old - m) + tl.sum(p, axis=1)
        
        o_l = o_l * tl.exp(m_old - m)[:, None]
        o_r = o_r * tl.exp(m_old - m)[:, None]
        
        # Load full V tile [BLOCK_N, D] (split into two column-halves)
        v_offs_l = batch_offset + head_offset + n_row_offs * stride_V_s + tl.arange(0, 64)[None, :]
        v_tile_l = tl.load(V + v_offs_l, mask=k_mask, other=0.0)
        
        v_offs_r = batch_offset + head_offset + n_row_offs * stride_V_s + 64 + tl.arange(0, 64)[None, :]
        v_tile_r = tl.load(V + v_offs_r, mask=k_mask, other=0.0)
        
        # RS-GEMM: P @ V
        o_l = tl.dot(p, v_tile_l, o_l)
        o_r = tl.dot(p, v_tile_r, o_r)
        
    o_l = o_l / l[:, None]
    o_r = o_r / l[:, None]
    
    o_offs_l = batch_offset + head_offset + q_row_offs * stride_O_s + tl.arange(0, 64)[None, :]
    tl.store(O + o_offs_l, o_l, mask=q_mask_l)
    
    o_offs_r = batch_offset + head_offset + q_row_offs * stride_O_s + 64 + tl.arange(0, 64)[None, :]
    tl.store(O + o_offs_r, o_r, mask=q_mask_l)
    
    lse = m + tl.log(l)
    lse_offs = b * stride_LSE_b + h * stride_LSE_h + (q_idx + tl.arange(0, BLOCK_Q))
    lse_mask = (q_idx + tl.arange(0, BLOCK_Q)) < S
    tl.store(LSE + lse_offs, lse, mask=lse_mask)


def run(Q, K, V, O, LSE):
    """Compute Non-causal Multi-Head Attention output O and LogSumExp LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_Q = 64
    BLOCK_N = 64
    
    num_blocks_q = triton.cdiv(S, BLOCK_Q)
    grid = (B * H * num_blocks_q,)
    
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        B, H, S,
        scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_Q=BLOCK_Q, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3,
    )