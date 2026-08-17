import torch
import triton
import triton.language as tl
import math

BLOCK = 64


@triton.jit
def _bwd_dk_dv_kernel(
    Q, K, V, O, dO, L,
    dK, dV,
    S, scale,
    y_s1, l_s1,
    BLOCK: tl.constexpr,
):
    n_blk = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    valid_n = (bh_id * S + n_blk * BLOCK + tl.arange(0, BLOCK)) < S
    
    k_ptr = K + (n_blk * BLOCK * y_s1) + bh_id * (S * y_s1)
    k0 = tl.load(k_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + tl.arange(0, 64)[None, :] * 1, mask=valid_n[:, None], other=0.0)
    k1 = tl.load(k_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + (tl.arange(0, 64)[None, :] + 64) * 1, mask=valid_n[:, None], other=0.0)
    
    v_ptr = V + (n_blk * BLOCK * y_s1) + bh_id * (S * y_s1)
    v0 = tl.load(v_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + tl.arange(0, 64)[None, :] * 1, mask=valid_n[:, None], other=0.0)
    v1 = tl.load(v_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + (tl.arange(0, 64)[None, :] + 64) * 1, mask=valid_n[:, None], other=0.0)
    
    dk0 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dk1 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dv0 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dv1 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    
    total_blks = tl.cdiv(S, BLOCK)
    
    for i_blk in range(n_blk, total_blks):
        q_ptr = Q + (i_blk * BLOCK * y_s1) + bh_id * (S * y_s1)
        q0 = tl.load(q_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + tl.arange(0, 64)[None, :] * 1, mask=((bh_id * S + i_blk * BLOCK + tl.arange(0, BLOCK)) < S)[:, None], other=0.0)
        q1 = tl.load(q_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + (tl.arange(0, 64)[None, :] + 64) * 1, mask=((bh_id * S + i_blk * BLOCK + tl.arange(0, BLOCK)) < S)[:, None], other=0.0)
        
        do_ptr = dO + (i_blk * BLOCK * y_s1) + bh_id * (S * y_s1)
        do0 = tl.load(do_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + tl.arange(0, 64)[None, :] * 1, mask=((bh_id * S + i_blk * BLOCK + tl.arange(0, BLOCK)) < S)[:, None], other=0.0)
        do1 = tl.load(do_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + (tl.arange(0, 64)[None, :] + 64) * 1, mask=((bh_id * S + i_blk * BLOCK + tl.arange(0, BLOCK)) < S)[:, None], other=0.0)
        
        o_ptr = O + (i_blk * BLOCK * y_s1) + bh_id * (S * y_s1)
        o0 = tl.load(o_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + tl.arange(0, 64)[None, :] * 1, mask=((bh_id * S + i_blk * BLOCK + tl.arange(0, BLOCK)) < S)[:, None], other=0.0)
        o1 = tl.load(o_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + (tl.arange(0, 64)[None, :] + 64) * 1, mask=((bh_id * S + i_blk * BLOCK + tl.arange(0, BLOCK)) < S)[:, None], other=0.0)
        
        l_ptr = L + bh_id * l_s1 + i_blk * BLOCK
        l = tl.load(l_ptr + tl.arange(0, BLOCK), mask=(bh_id * S + i_blk * BLOCK + tl.arange(0, BLOCK)) < S, other=-float('inf'))
        
        d0 = tl.sum(do0 * o0, axis=1)
        d1 = tl.sum(do1 * o1, axis=1)
        d = d0 + d1
        d_exp = d[:, None] * tl.ones((BLOCK, BLOCK), dtype=tl.bfloat16)
        
        s_gem = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        s_gem = tl.dot(q0, k0.T, s_gem)
        s_gem = tl.dot(q1, k1.T, s_gem)
        s = s_gem * scale
        
        apply_mask = (n_blk <= i_blk)
        if apply_mask:
            off_m = tl.arange(0, BLOCK)
            off_n = tl.arange(0, BLOCK)
            valid = (n_blk * BLOCK + off_n[None, :]) <= (i_blk * BLOCK + off_m[:, None])
            s = tl.where(valid, s, -float('inf'))
            
        p = tl.exp(s - l)
        
        if apply_mask:
            p = tl.where(valid, p, 0.0)
            
        dp_gem = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        dp_gem = tl.dot(do0, v0.T, dp_gem)
        dp_gem = tl.dot(do1, v1.T, dp_gem)
        
        ds = p * (dp_gem - d_exp) * scale
        
        dv0 = tl.dot(p.T, do0, dv0)
        dv1 = tl.dot(p.T, do1, dv1)
        
        dk0 = tl.dot(ds.T, q0, dk0)
        dk1 = tl.dot(ds.T, q1, dk1)
        
    dK_ptr = dK + (n_blk * BLOCK * y_s1) + bh_id * (S * y_s1)
    tl.store(dK_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + tl.arange(0, 64)[None, :] * 1, dk0.to(tl.bfloat16), mask=valid_n[:, None])
    tl.store(dK_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + (tl.arange(0, 64)[None, :] + 64) * 1, dk1.to(tl.bfloat16), mask=valid_n[:, None])
    
    dV_ptr = dV + (n_blk * BLOCK * y_s1) + bh_id * (S * y_s1)
    tl.store(dV_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + tl.arange(0, 64)[None, :] * 1, dv0.to(tl.bfloat16), mask=valid_n[:, None])
    tl.store(dV_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + (tl.arange(0, 64)[None, :] + 64) * 1, dv1.to(tl.bfloat16), mask=valid_n[:, None])


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, O, dO, L,
    dQ,
    S, scale,
    y_s1, l_s1,
    BLOCK: tl.constexpr,
):
    m_blk = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    valid_m = (bh_id * S + m_blk * BLOCK + tl.arange(0, BLOCK)) < S
    
    q_ptr = Q + (m_blk * BLOCK * y_s1) + bh_id * (S * y_s1)
    q0 = tl.load(q_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + tl.arange(0, 64)[None, :] * 1, mask=valid_m[:, None], other=0.0)
    q1 = tl.load(q_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + (tl.arange(0, 64)[None, :] + 64) * 1, mask=valid_m[:, None], other=0.0)
    
    do_ptr = dO + (m_blk * BLOCK * y_s1) + bh_id * (S * y_s1)
    do0 = tl.load(do_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + tl.arange(0, 64)[None, :] * 1, mask=valid_m[:, None], other=0.0)
    do1 = tl.load(do_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + (tl.arange(0, 64)[None, :] + 64) * 1, mask=valid_m[:, None], other=0.0)
    
    o_ptr = O + (m_blk * BLOCK * y_s1) + bh_id * (S * y_s1)
    o0 = tl.load(o_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + tl.arange(0, 64)[None, :] * 1, mask=valid_m[:, None], other=0.0)
    o1 = tl.load(o_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + (tl.arange(0, 64)[None, :] + 64) * 1, mask=valid_m[:, None], other=0.0)
    
    l_ptr = L + bh_id * l_s1 + m_blk * BLOCK
    l = tl.load(l_ptr + tl.arange(0, BLOCK), mask=valid_m, other=-float('inf'))
    
    dq0 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    dq1 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    
    d0 = tl.sum(do0 * o0, axis=1)
    d1 = tl.sum(do1 * o1, axis=1)
    d = d0 + d1
    d_exp = d[:, None] * tl.ones((BLOCK, BLOCK), dtype=tl.bfloat16)
    
    for n_blk in range(0, m_blk + 1):
        k_ptr = K + (n_blk * BLOCK * y_s1) + bh_id * (S * y_s1)
        k0 = tl.load(k_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + tl.arange(0, 64)[None, :] * 1, mask=((bh_id * S + n_blk * BLOCK + tl.arange(0, BLOCK)) < S)[:, None], other=0.0)
        k1 = tl.load(k_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + (tl.arange(0, 64)[None, :] + 64) * 1, mask=((bh_id * S + n_blk * BLOCK + tl.arange(0, BLOCK)) < S)[:, None], other=0.0)
        
        v_ptr = V + (n_blk * BLOCK * y_s1) + bh_id * (S * y_s1)
        v0 = tl.load(v_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + tl.arange(0, 64)[None, :] * 1, mask=((bh_id * S + n_blk * BLOCK + tl.arange(0, BLOCK)) < S)[:, None], other=0.0)
        v1 = tl.load(v_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + (tl.arange(0, 64)[None, :] + 64) * 1, mask=((bh_id * S + n_blk * BLOCK + tl.arange(0, BLOCK)) < S)[:, None], other=0.0)
        
        s_gem = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        s_gem = tl.dot(q0, k0.T, s_gem)
        s_gem = tl.dot(q1, k1.T, s_gem)
        s = s_gem * scale
        
        apply_mask = (n_blk <= m_blk)
        if apply_mask:
            off_m = tl.arange(0, BLOCK)
            off_n = tl.arange(0, BLOCK)
            valid = (n_blk * BLOCK + off_n[None, :]) <= (m_blk * BLOCK + off_m[:, None])
            s = tl.where(valid, s, -float('inf'))
            
        p = tl.exp(s - l)
        
        if apply_mask:
            p = tl.where(valid, p, 0.0)
            
        dp_gem = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
        dp_gem = tl.dot(do0, v0.T, dp_gem)
        dp_gem = tl.dot(do1, v1.T, dp_gem)
        
        ds = p * (dp_gem - d_exp) * scale
        
        dq0 = tl.dot(ds, k0, dq0)
        dq1 = tl.dot(ds, k1, dq1)
        
    dQ_ptr = dQ + (m_blk * BLOCK * y_s1) + bh_id * (S * y_s1)
    tl.store(dQ_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + tl.arange(0, 64)[None, :] * 1, dq0.to(tl.bfloat16), mask=valid_m[:, None])
    tl.store(dQ_ptr + tl.arange(0, BLOCK)[:, None] * y_s1 + (tl.arange(0, 64)[None, :] + 64) * 1, dq1.to(tl.bfloat16), mask=valid_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute FlashAttention backward pass for causal multi-head attention."""
    torch.cuda.set_device(Q.device)
    
    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    
    scale = 1.0 / math.sqrt(128)
    
    y_s1 = Q.stride()[1]
    l_s1 = L.stride()[1]
    
    config = triton.Config(
        {"BLOCK": BLOCK},
        num_warps=4,
        num_stages=2,
    )
    
    grid = (
        triton.cdiv(S, BLOCK),
        B * H,
    )
    
    _bwd_dk_dv_kernel[grid](Q, K, V, O, dO, L, dK, dV, S, scale, y_s1, l_s1, BLOCK=BLOCK)
    _bwd_dq_kernel[grid](Q, K, V, O, dO, L, dQ, S, scale, y_s1, l_s1, BLOCK=BLOCK)