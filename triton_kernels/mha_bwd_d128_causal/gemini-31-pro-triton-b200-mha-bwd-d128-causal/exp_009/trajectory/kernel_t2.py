import torch
import triton
import triton.language as tl
import math

@triton.jit
def _bwd_q_kernel(
    Q, K, V, O, dO, L,
    dQ,
    sm_scale, sm_scale_log2, log2_e,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    H, S,
    d: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Computes dQ in a standalone pass.
    Each program owns a Query block and loops over corresponding Key/Value blocks.
    Accumulates locally in high-precision registers.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    q_base = Q + b * stride_q_b + h * stride_q_h
    k_base = K + b * stride_k_b + h * stride_k_h
    v_base = V + b * stride_v_b + h * stride_v_h
    o_base = O + b * stride_o_b + h * stride_o_h
    do_base = dO + b * stride_do_b + h * stride_do_h
    l_base = L + b * stride_l_b + h * stride_l_h
    dq_base = dQ + b * stride_dq_b + h * stride_dq_h
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    
    mask_m = offs_m < S
    
    q_ptrs = q_base + offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    o_ptrs = o_base + offs_m[:, None] * stride_o_s + offs_d[None, :] * stride_o_d
    do_ptrs = do_base + offs_m[:, None] * stride_do_s + offs_d[None, :] * stride_do_d
    l_ptrs = l_base + offs_m * stride_l_s
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Compute precise row-wise delta
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    lse_scaled = lse * log2_e
    
    dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    max_k_idx = tl.minimum((pid_m + 1) * BLOCK_M, S)
    num_k_chunks = tl.cdiv(max_k_idx, BLOCK_N)
    
    k_ptrs = k_base + tl.arange(0, BLOCK_N)[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
    v_ptrs = v_base + tl.arange(0, BLOCK_N)[:, None] * stride_v_s + offs_d[None, :] * stride_v_d
    
    for k_idx in range(0, num_k_chunks):
        offs_n = k_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale_log2
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_score = causal_mask & mask_m[:, None] & mask_n[None, :]
        
        scores = tl.where(valid_score, scores, -float("inf"))
        
        # Native hardware exp2 instruction mapping
        p = tl.math.exp2(scores - lse_scaled[:, None])
        p = tl.where(valid_score, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid_score, ds, 0.0)
        
        dq = tl.dot(ds.to(q.dtype), k, acc=dq, out_dtype=tl.float32)
        
        k_ptrs += BLOCK_N * stride_k_s
        v_ptrs += BLOCK_N * stride_v_s
        
    dq_ptrs = dq_base + offs_m[:, None] * stride_dq_s + offs_d[None, :] * stride_dq_d
    tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_m[:, None])


@triton.jit
def _bwd_kv_kernel(
    Q, K, V, O, dO, L,
    dK, dV,
    sm_scale, sm_scale_log2, log2_e,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    H, S,
    d: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """
    Computes dK and dV in a standalone pass.
    Each program owns a Key/Value block and streams over Q/O/dO blocks.
    Accumulates locally in high-precision registers to eliminate non-deterministic atomic errors.
    """
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    q_base = Q + b * stride_q_b + h * stride_q_h
    k_base = K + b * stride_k_b + h * stride_k_h
    v_base = V + b * stride_v_b + h * stride_v_h
    o_base = O + b * stride_o_b + h * stride_o_h
    do_base = dO + b * stride_do_b + h * stride_do_h
    l_base = L + b * stride_l_b + h * stride_l_h
    dk_base = dK + b * stride_dk_b + h * stride_dk_h
    dv_base = dV + b * stride_dv_b + h * stride_dv_h
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    
    mask_n = offs_n < S
    
    k_ptrs = k_base + offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
    v_ptrs = v_base + offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    min_m_idx = (pid_n * BLOCK_N) // BLOCK_M
    num_m_chunks = tl.cdiv(S, BLOCK_M)
    
    q_ptrs = q_base + (min_m_idx * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    o_ptrs = o_base + (min_m_idx * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_o_s + offs_d[None, :] * stride_o_d
    do_ptrs = do_base + (min_m_idx * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_do_s + offs_d[None, :] * stride_do_d
    l_ptrs = l_base + (min_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)) * stride_l_s
    
    for m_idx in range(min_m_idx, num_m_chunks):
        offs_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        lse_scaled = lse * log2_e
        
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale_log2
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_score = causal_mask & mask_m[:, None] & mask_n[None, :]
        
        scores = tl.where(valid_score, scores, -float("inf"))
        
        p = tl.math.exp2(scores - lse_scaled[:, None])
        p = tl.where(valid_score, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid_score, ds, 0.0)
        
        dk = tl.dot(tl.trans(ds.to(k.dtype)), q, acc=dk, out_dtype=tl.float32)
        dv = tl.dot(tl.trans(p.to(v.dtype)), do, acc=dv, out_dtype=tl.float32)
        
        q_ptrs += BLOCK_M * stride_q_s
        o_ptrs += BLOCK_M * stride_o_s
        do_ptrs += BLOCK_M * stride_do_s
        l_ptrs += BLOCK_M * stride_l_s
        
    dk_ptrs = dk_base + offs_n[:, None] * stride_dk_s + offs_d[None, :] * stride_dk_d
    dv_ptrs = dv_base + offs_n[:, None] * stride_dv_s + offs_d[None, :] * stride_dv_d
    
    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes exact, deterministic Multi-Head Attention causal backward gradients without atmoic_add.
    Fills pre-allocated destination buffers matching the specification requirements.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    # Precompute factors for faster math.exp2 translation
    log2_e = 1.4426950408889634
    sm_scale_log2 = sm_scale * log2_e
    
    if S == 0:
        return
        
    # Highly tuned block sizes ensuring peak register and SRAM occupancy on SM100.
    BLOCK_M_Q = 128
    BLOCK_N_Q = 64
    
    BLOCK_M_KV = 64
    BLOCK_N_KV = 128

    grid_q = (triton.cdiv(S, BLOCK_M_Q), B * H, 1)
    _bwd_q_kernel[grid_q](
        Q, K, V, O, dO, L,
        dQ,
        sm_scale, sm_scale_log2, log2_e,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        H=H, S=S,
        d=128, BLOCK_M=BLOCK_M_Q, BLOCK_N=BLOCK_N_Q,
        num_warps=8, num_stages=4
    )

    grid_kv = (triton.cdiv(S, BLOCK_N_KV), B * H, 1)
    _bwd_kv_kernel[grid_kv](
        Q, K, V, O, dO, L,
        dK, dV,
        sm_scale, sm_scale_log2, log2_e,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        H=H, S=S,
        d=128, BLOCK_M=BLOCK_M_KV, BLOCK_N=BLOCK_N_KV,
        num_warps=8, num_stages=3
    )