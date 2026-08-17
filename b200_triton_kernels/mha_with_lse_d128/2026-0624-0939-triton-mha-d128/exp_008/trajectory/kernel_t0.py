import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q, K, V, O, LSE,
    B, H, S, D,
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
    
    # Load Q tile [BLOCK_Q, D]
    q_row_offs = (q_idx + tl.arange(0, BLOCK_Q))[:, None]
    d_offs = tl.arange(0, D)[None, :]
    q_offs = batch_offset + head_offset + q_row_offs * stride_Q_s + d_offs
    q_mask = (q_row_offs.squeeze() < S)[:, None]
    q_tile = tl.load(Q + q_offs, mask=q_mask, other=0.0)
    
    # Split Q into two independent column-halves to distribute work
    q_left, q_right = tl.split(q_tile, 2)
    
    o = tl.zeros((BLOCK_Q, D), dtype=tl.float32)
    m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    
    for n_block in range(num_blocks_n):
        n_idx = n_block * BLOCK_N
        n_row_offs = (n_idx + tl.arange(0, BLOCK_N))[:, None]
        
        # Load K tile and split into two column-halves
        k_offs = batch_offset + head_offset + n_row_offs * stride_K_s + d_offs
        k_mask = (n_row_offs.squeeze() < S)[:, None]
        k_tile = tl.load(K + k_offs, mask=k_mask, other=0.0)
        k_left, k_right = tl.split(k_tile, 2)
        
        # SS-GEMM: Q @ K^T
        s = tl.zeros((BLOCK_Q, BLOCK_N), dtype=tl.float32)
        s = tl.dot(q_left, k_left.T, s)
        s = tl.dot(q_right, k_right.T, s)
        
        s = s * scale
        
        m_old = m
        m = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m)
        
        l_old = l
        l = l_old * tl.exp(m_old - m) + tl.sum(p, axis=1)
        
        o = o * tl.exp(m_old - m)[:, None]
        
        # Load full V tile [BLOCK_N, D]
        v_offs = batch_offset + head_offset + n_row_offs * stride_V_s + d_offs
        v_tile = tl.load(V + v_offs, mask=k_mask, other=0.0)
        
        # RS-GEMM: P @ V
        o = tl.dot(p, v_tile, o)
        
    o = o / l[:, None]
    
    o_offs = batch_offset + head_offset + q_row_offs * stride_O_s + d_offs
    tl.store(O + o_offs, o, mask=q_mask)
    
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
        B, H, S, D,
        scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_Q=BLOCK_Q, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3,
    )