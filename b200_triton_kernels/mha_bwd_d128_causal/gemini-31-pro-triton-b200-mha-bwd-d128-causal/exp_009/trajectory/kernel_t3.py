import torch
import triton
import triton.language as tl
import math


def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)


@triton.jit
def _bwd_q_kernel(
    Q, K, V, O, dO, L, dQ,
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
    
    # TMA descriptors provide robust out-of-bounds safety and hardware pipelining
    desc_q = tl.make_tensor_descriptor(q_base, shape=[S, d], strides=[stride_q_s, stride_q_d], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_o = tl.make_tensor_descriptor(o_base, shape=[S, d], strides=[stride_o_s, stride_o_d], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_do = tl.make_tensor_descriptor(do_base, shape=[S, d], strides=[stride_do_s, stride_do_d], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_dq = tl.make_tensor_descriptor(dq_base, shape=[S, d], strides=[stride_dq_s, stride_dq_d], block_shape=[BLOCK_M, d], padding_option="zero")
    
    desc_k = tl.make_tensor_descriptor(k_base, shape=[S, d], strides=[stride_k_s, stride_k_d], block_shape=[BLOCK_N, d], padding_option="zero")
    desc_v = tl.make_tensor_descriptor(v_base, shape=[S, d], strides=[stride_v_s, stride_v_d], block_shape=[BLOCK_N, d], padding_option="zero")
    
    m_offset = pid_m * BLOCK_M
    q = tl.load(desc_q, [m_offset, 0])
    o = tl.load(desc_o, [m_offset, 0])
    do = tl.load(desc_do, [m_offset, 0])
    
    offs_m = m_offset + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    # L is 1D per head, standard pointers are safe and sufficient here
    l_ptrs = l_base + offs_m * stride_l_s
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    lse_scaled = lse * log2_e
    
    dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    max_k_idx = tl.minimum(m_offset + BLOCK_M, S)
    num_k_chunks = tl.cdiv(max_k_idx, BLOCK_N)
    
    for k_idx in range(0, num_k_chunks):
        n_offset = k_idx * BLOCK_N
        k = tl.load(desc_k, [n_offset, 0])
        v = tl.load(desc_v, [n_offset, 0])
        
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale_log2
        
        offs_n = n_offset + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_score = causal_mask & mask_m[:, None] & (offs_n[None, :] < S)
        
        scores = tl.where(valid_score, scores, -float("inf"))
        
        p = tl.math.exp2(scores - lse_scaled[:, None])
        p = tl.where(valid_score, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid_score, ds, 0.0)
        
        dq = tl.dot(ds.to(q.dtype), k, acc=dq, out_dtype=tl.float32)
        
    tl.store(desc_dq, [m_offset, 0], dq.to(q.dtype))


@triton.jit
def _bwd_kv_kernel(
    Q, K, V, O, dO, L, dK, dV,
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
    
    desc_k = tl.make_tensor_descriptor(k_base, shape=[S, d], strides=[stride_k_s, stride_k_d], block_shape=[BLOCK_N, d], padding_option="zero")
    desc_v = tl.make_tensor_descriptor(v_base, shape=[S, d], strides=[stride_v_s, stride_v_d], block_shape=[BLOCK_N, d], padding_option="zero")
    desc_dk = tl.make_tensor_descriptor(dk_base, shape=[S, d], strides=[stride_dk_s, stride_dk_d], block_shape=[BLOCK_N, d], padding_option="zero")
    desc_dv = tl.make_tensor_descriptor(dv_base, shape=[S, d], strides=[stride_dv_s, stride_dv_d], block_shape=[BLOCK_N, d], padding_option="zero")
    
    desc_q = tl.make_tensor_descriptor(q_base, shape=[S, d], strides=[stride_q_s, stride_q_d], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_o = tl.make_tensor_descriptor(o_base, shape=[S, d], strides=[stride_o_s, stride_o_d], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_do = tl.make_tensor_descriptor(do_base, shape=[S, d], strides=[stride_do_s, stride_do_d], block_shape=[BLOCK_M, d], padding_option="zero")
    
    n_offset = pid_n * BLOCK_N
    k = tl.load(desc_k, [n_offset, 0])
    v = tl.load(desc_v, [n_offset, 0])
    
    dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    min_m_idx = n_offset // BLOCK_M
    num_m_chunks = tl.cdiv(S, BLOCK_M)
    
    offs_n = n_offset + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    for m_idx in range(min_m_idx, num_m_chunks):
        m_offset = m_idx * BLOCK_M
        
        q = tl.load(desc_q, [m_offset, 0])
        o = tl.load(desc_o, [m_offset, 0])
        do = tl.load(desc_do, [m_offset, 0])
        
        offs_m = m_offset + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        l_ptrs = l_base + offs_m * stride_l_s
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
        
    tl.store(desc_dk, [n_offset, 0], dk.to(k.dtype))
    tl.store(desc_dv, [n_offset, 0], dv.to(v.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes exact, deterministic Multi-Head Attention causal backward gradients.
    Fills pre-allocated destination buffers without atomic_add.
    Uses Blackwell TMA descriptors for ultimate safety and high performance pipelining.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    # Precompute log base factor to deploy optimized tl.math.exp2 mapped instructions
    log2_e = 1.4426950408889634
    sm_scale_log2 = sm_scale * log2_e
    
    if S == 0:
        return
        
    BLOCK_M_Q = 128
    BLOCK_N_Q = 64
    
    BLOCK_M_KV = 64
    BLOCK_N_KV = 128

    grid_q = (triton.cdiv(S, BLOCK_M_Q), B * H, 1)
    _bwd_q_kernel[grid_q](
        Q, K, V, O, dO, L, dQ,
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
        Q, K, V, O, dO, L, dK, dV,
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
        num_warps=8, num_stages=4
    )