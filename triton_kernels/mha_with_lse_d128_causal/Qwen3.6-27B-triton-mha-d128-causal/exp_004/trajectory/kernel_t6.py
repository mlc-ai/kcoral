import torch
import triton
import triton.language as tl


@triton.jit
def _causal_mha_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, D,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(S, BLOCK_M)
    
    bh = pid // num_pid_m
    bm = pid % num_pid_m
    
    bid = bh // NUM_HEADS
    hid = bh % NUM_HEADS
    
    q_offs_s = bm * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offs_s = tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, BLOCK_D)
    
    # Base offsets for this batch, head
    q_base = Q + bid * stride_qb + hid * stride_qh
    k_base = K + bid * stride_kb + hid * stride_kh
    v_base = V + bid * stride_vb + hid * stride_vh
    o_base = O + bid * stride_ob + hid * stride_oh
    lse_base = LSE + bid * stride_lseb + hid * stride_lseh
    
    # Q pointers shape [BLOCK_M, BLOCK_D]
    q_ptrs = q_base + q_offs_s[:, None] * stride_qs + d_offs[None, :] * stride_qd
    q_mask = (q_offs_s[:, None] < S) & (d_offs[None, :] < D)
    Q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    scale = 1.0 / tl.sqrt(D, dtype=tl.float32)
    
    off_k_init = 0
    while off_k_init < S:
        k_offs_cur = off_k_init + k_offs_s
        
        kv_mask = (k_offs_cur[:, None] < S) & (d_offs[None, :] < D)
        
        # Load K and V tiles
        k_ptrs = k_base + k_offs_cur[:, None] * stride_ks + d_offs[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        
        v_ptrs = v_base + k_offs_cur[:, None] * stride_vs + d_offs[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        
        # Compute scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_tile, K_tile.T)
        scores = scores * scale
        
        # Causal mask: q_pos >= k_pos, plus key boundary
        causal = (q_offs_s[:, None] >= k_offs_cur[None, :]) & (k_offs_cur[None, :] < S)
        scores = tl.where(causal, scores, float("-inf"))
        
        # Online softmax step
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        old_scale = tl.exp(m_i - m_new)
        p_smooth = tl.exp(scores - m_new[:, None])
        
        # Update accumulator: old_scale * acc + P @ V
        acc_scaled = old_scale[:, None] * acc
        Pv = tl.dot(p_smooth.to(tl.bfloat16), V_tile)
        acc = acc_scaled + Pv
        
        # Update denominator
        l_i = old_scale * l_i + tl.sum(p_smooth, axis=1)
        
        m_i = m_new
        off_k_init += BLOCK_N
    
    # Finalize: normalize
    denom = tl.where(l_i > 0.0, l_i, 1.0)
    O_tile = (acc / denom[:, None]).to(tl.bfloat16)
    
    # Store output O
    o_ptrs = o_base + q_offs_s[:, None] * stride_os + d_offs[None, :] * stride_od
    tl.store(o_ptrs, O_tile, mask=q_mask)
    
    # Store LSE
    lse_out = tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf"))
    lse_ptrs = lse_base + q_offs_s * stride_lses
    tl.store(lse_ptrs, lse_out, mask=(q_offs_s < S))


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward pass."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    if S == 0 or B == 0 or H == 0:
        return
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128
    
    num_pid_m = triton.cdiv(S, BLOCK_M)
    grid_size = B * H * num_pid_m
    grid = (grid_size,)
    
    _causal_mha_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, D,
        NUM_HEADS=H,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )