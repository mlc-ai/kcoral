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
    NEG_INF_FP32: tl.constexpr,
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
    
    q_offset_s = bm * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offset_s = tl.arange(0, BLOCK_N)
    d_offset = tl.arange(0, BLOCK_D)
    
    # Base pointers for this batch, head
    q_base = Q + bid * stride_qb + hid * stride_qh
    k_base = K + bid * stride_kb + hid * stride_kh
    v_base = V + bid * stride_vb + hid * stride_vh
    o_base = O + bid * stride_ob + hid * stride_oh
    lse_base = LSE + bid * stride_lseb + hid * stride_lseh
    
    # Load Q tile [BLOCK_M, BLOCK_D]
    q_ptrs = q_base + q_offset_s[:, None] * stride_qs + d_offset[None, :] * stride_qd
    q_mask = (q_offset_s[:, None] < S) & (d_offset[None, :] < D)
    Q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Online softmax state
    m_i = tl.full((BLOCK_M,), NEG_INF_FP32, dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    inv_sqrt_d = tl.cast(1.0 / tl.sqrt(D, dtype=tl.float32), tl.float32)
    
    off_n = 0
    while off_n < S:
        cur_k_s = off_n + k_offset_s
        
        kv_valid = (cur_k_s[:, None] < S) & (d_offset[None, :] < D)
        
        # Load K
        k_ptrs = k_base + cur_k_s[:, None] * stride_ks + d_offset[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=kv_valid, other=0.0)
        
        # Load V  
        v_ptrs = v_base + cur_k_s[:, None] * stride_vs + d_offset[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=kv_valid, other=0.0)
        
        # Dot product: bf16 inputs -> fp32 accumulation
        scores_fp32 = tl.dot(Q_tile, K_tile.T)
        scores_fp32 = tl.fma(scores_fp32, inv_sqrt_d, 0.0)
        
        # Causal mask + boundary
        attn_mask = (q_offset_s[:, None] >= cur_k_s[None, :]) & (cur_k_s[None, :] < S)
        scores_fp32 = tl.where(attn_mask, scores_fp32, NEG_INF_FP32)
        
        # Softmax numerics
        row_max_fp32 = tl.max(scores_fp32, axis=1)
        m_new_fp32 = tl.maximum(m_i, row_max_fp32)
        
        old_scale_fp32 = tl.exp(m_i - m_new_fp32)
        p_fp32 = tl.exp(scores_fp32 - m_new_fp32[:, None])
        
        # Update accumulator
        acc_scaled = tl.fma(old_scale_fp32[:, None], acc, 0.0)
        pv_fp32 = tl.dot(p_fp32.to(tl.bfloat16), V_tile.to(tl.bfloat16))
        acc = acc_scaled + pv_fp32
        
        # Update denominator  
        l_i = tl.fma(old_scale_fp32, l_i, 0.0) + tl.sum(p_fp32, axis=1)
        
        m_i = m_new_fp32
        off_n += BLOCK_N
    
    # Normalize output
    denom_fp32 = tl.where(l_i > 0.0, l_i, 1.0)
    O_tile = (acc / denom_fp32[:, None]).to(tl.bfloat16)
    
    # Store O
    o_ptrs = o_base + q_offset_s[:, None] * stride_os + d_offset[None, :] * stride_od
    tl.store(o_ptrs, O_tile, mask=q_mask)
    
    # Store LSE
    lse_fp32 = tl.where(l_i > 0.0, m_i + tl.log(l_i), NEG_INF_FP32)
    lse_ptrs = lse_base + q_offset_s * stride_lses
    tl.store(lse_ptrs, lse_fp32, mask=(q_offset_s < S))


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
        NEG_INF_FP32=float("-inf"),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )