import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q, K, V, O, LSE,
    H, S,
    scale,
    stride_Q_b, stride_Q_h, stride_Q_s, stride_Q_d,
    stride_K_b, stride_K_h, stride_K_s, stride_K_d,
    stride_V_b, stride_V_h, stride_V_s, stride_V_d,
    stride_O_b, stride_O_h, stride_O_s, stride_O_d,
    stride_LSE_b, stride_LSE_h, stride_LSE_s,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    FlashAttention-style MHA kernel for (B, H, S, D) tensors.
    Each program processes one (b, h, q_idx) tile of size BLOCK_Q x D.
    Iterates over all N-blocks of K and V to accumulate the output.
    """
    constexpr_D = 128
    assert constexpr_D == 128, "Hardcoded D dimension support"
    
    pid = tl.program_id(0)
    num_blocks_q = tl.cdiv(S, BLOCK_Q)
    head_idx = pid // num_blocks_q
    b = head_idx // H
    h = head_idx % H
    q_block = pid % num_blocks_q
    q_idx = q_block * BLOCK_Q
    
    num_blocks_n = tl.cdiv(S, BLOCK_N)
    
    # Explicitly cast strides to int32 to avoid FP32 promotion bugs when adding to tensor offsets
    s_Q_b = tl.int32(stride_Q_b); s_Q_h = tl.int32(stride_Q_h); s_Q_s = tl.int32(stride_Q_s); s_Q_d = tl.int32(stride_Q_d)
    s_K_b = tl.int32(stride_K_b); s_K_h = tl.int32(stride_K_h); s_K_s = tl.int32(stride_K_s); s_K_d = tl.int32(stride_K_d)
    s_V_b = tl.int32(stride_V_b); s_V_h = tl.int32(stride_V_h); s_V_s = tl.int32(stride_V_s); s_V_d = tl.int32(stride_V_d)
    s_O_b = tl.int32(stride_O_b); s_O_h = tl.int32(stride_O_h); s_O_s = tl.int32(stride_O_s); s_O_d = tl.int32(stride_O_d)
    s_LSE_b = tl.int32(stride_LSE_b); s_LSE_h = tl.int32(stride_LSE_h); s_LSE_s = tl.int32(stride_LSE_s)

    b_offs_Q = b * s_Q_b; h_offs_Q = h * s_Q_h
    b_offs_K = b * s_K_b; h_offs_K = h * s_K_h
    b_offs_V = b * s_V_b; h_offs_V = h * s_V_h
    b_offs_O = b * s_O_b; h_offs_O = h * s_O_h
    
    q_row_offs = q_idx + tl.arange(0, BLOCK_Q)
    q_mask = (q_row_offs < S)[:, None]
    
    # Static constexpr D offsets avoiding runtime 'D' parameter constraints
    d_offs_l = tl.arange(0, 64)
    d_offs_r = 64 + tl.arange(0, 64)
    
    q_offs_l = b_offs_Q + h_offs_Q + q_row_offs[:, None] * s_Q_s + d_offs_l[None, :] * s_Q_d
    q_tile_l = tl.load(Q + q_offs_l, mask=q_mask, other=0.0)
    
    q_offs_r = b_offs_Q + h_offs_Q + q_row_offs[:, None] * s_Q_s + d_offs_r[None, :] * s_Q_d
    q_tile_r = tl.load(Q + q_offs_r, mask=q_mask, other=0.0)
    
    o_l = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    o_r = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    
    for n_block in range(num_blocks_n):
        n_idx = n_block * BLOCK_N
        n_row_offs = n_idx + tl.arange(0, BLOCK_N)
        k_mask = (n_row_offs < S)[:, None]
        
        k_offs_l = b_offs_K + h_offs_K + n_row_offs[:, None] * s_K_s + d_offs_l[None, :] * s_K_d
        k_tile_l = tl.load(K + k_offs_l, mask=k_mask, other=0.0)
        
        k_offs_r = b_offs_K + h_offs_K + n_row_offs[:, None] * s_K_s + d_offs_r[None, :] * s_K_d
        k_tile_r = tl.load(K + k_offs_r, mask=k_mask, other=0.0)
        
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
        
        v_offs_l = b_offs_V + h_offs_V + n_row_offs[:, None] * s_V_s + d_offs_l[None, :] * s_V_d
        v_tile_l = tl.load(V + v_offs_l, mask=k_mask, other=0.0)
        
        v_offs_r = b_offs_V + h_offs_V + n_row_offs[:, None] * s_V_s + d_offs_r[None, :] * s_V_d
        v_tile_r = tl.load(V + v_offs_r, mask=k_mask, other=0.0)
        
        o_l = tl.dot(p, v_tile_l, o_l)
        o_r = tl.dot(p, v_tile_r, o_r)
        
    o_l = o_l / l[:, None]
    o_r = o_r / l[:, None]
    
    o_offs_l = b_offs_O + h_offs_O + q_row_offs[:, None] * s_O_s + d_offs_l[None, :] * s_O_d
    tl.store(O + o_offs_l, o_l, mask=q_mask)
    
    o_offs_r = b_offs_O + h_offs_O + q_row_offs[:, None] * s_O_s + d_offs_r[None, :] * s_O_d
    tl.store(O + o_offs_r, o_r, mask=q_mask)
    
    lse = m + tl.log(l)
    lse_row_offs = q_idx + tl.arange(0, BLOCK_Q)
    lse_mask = lse_row_offs < S
    lse_offs = b * s_LSE_b + h * s_LSE_h + lse_row_offs * s_LSE_s
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
        H, S,
        scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_Q=BLOCK_Q, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3,
    )