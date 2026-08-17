import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _bwd_q_kernel(
    desc_q, desc_k, desc_v, desc_o, desc_do, desc_dq,
    L,
    sm_scale, sm_scale_log2, log2_e,
    stride_l_b, stride_l_h, stride_l_s,
    H, S,
    d: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    m_offset = pid_m * BLOCK_M
    
    # TMA natively guarantees bounds safety by hardware padding zero.
    q_4d = desc_q.load([b, h, m_offset, 0])
    o_4d = desc_o.load([b, h, m_offset, 0])
    do_4d = desc_do.load([b, h, m_offset, 0])
    
    q = tl.reshape(q_4d, (BLOCK_M, d))
    o = tl.reshape(o_4d, (BLOCK_M, d))
    do = tl.reshape(do_4d, (BLOCK_M, d))
    
    offs_m = m_offset + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    l_base = L + b * stride_l_b + h * stride_l_h
    l_ptrs = l_base + offs_m * stride_l_s
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    lse_scaled = lse * log2_e
    
    dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    max_k_idx = tl.minimum(m_offset + BLOCK_M, S)
    num_k_chunks = tl.cdiv(max_k_idx, BLOCK_N)
    
    for k_idx in range(0, num_k_chunks):
        n_offset = k_idx * BLOCK_N
        
        k_4d = desc_k.load([b, h, n_offset, 0])
        v_4d = desc_v.load([b, h, n_offset, 0])
        
        k = tl.reshape(k_4d, (BLOCK_N, d))
        v = tl.reshape(v_4d, (BLOCK_N, d))
        
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale_log2
        
        offs_n = n_offset + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_score = causal_mask & mask_m[:, None] & (offs_n[None, :] < S)
        
        scores = tl.where(valid_score, scores, -float("inf"))
        
        # Deploy optimized tl.math.exp2 mapped instructions
        p = tl.math.exp2(scores - lse_scaled[:, None])
        p = tl.where(valid_score, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid_score, ds, 0.0)
        
        dq = tl.dot(ds.to(q.dtype), k, acc=dq, out_dtype=tl.float32)
        
    desc_dq.store([b, h, m_offset, 0], tl.reshape(dq.to(q.dtype), (1, 1, BLOCK_M, d)))


@triton.jit
def _bwd_kv_kernel(
    desc_q, desc_k, desc_v, desc_o, desc_do, desc_dk, desc_dv,
    L,
    sm_scale, sm_scale_log2, log2_e,
    stride_l_b, stride_l_h, stride_l_s,
    H, S,
    d: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    n_offset = pid_n * BLOCK_N
    
    k_4d = desc_k.load([b, h, n_offset, 0])
    v_4d = desc_v.load([b, h, n_offset, 0])
    
    k = tl.reshape(k_4d, (BLOCK_N, d))
    v = tl.reshape(v_4d, (BLOCK_N, d))
    
    dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    min_m_idx = n_offset // BLOCK_M
    num_m_chunks = tl.cdiv(S, BLOCK_M)
    
    offs_n = n_offset + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    l_base = L + b * stride_l_b + h * stride_l_h
    
    for m_idx in range(min_m_idx, num_m_chunks):
        m_offset = m_idx * BLOCK_M
        
        q_4d = desc_q.load([b, h, m_offset, 0])
        o_4d = desc_o.load([b, h, m_offset, 0])
        do_4d = desc_do.load([b, h, m_offset, 0])
        
        q = tl.reshape(q_4d, (BLOCK_M, d))
        o = tl.reshape(o_4d, (BLOCK_M, d))
        do = tl.reshape(do_4d, (BLOCK_M, d))
        
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
        
    desc_dk.store([b, h, n_offset, 0], tl.reshape(dk.to(k.dtype), (1, 1, BLOCK_N, d)))
    desc_dv.store([b, h, n_offset, 0], tl.reshape(dv.to(v.dtype), (1, 1, BLOCK_N, d)))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes exact, deterministic Multi-Head Attention causal backward gradients without atomic_add.
    Fills pre-allocated destination buffers matching the specification requirements.
    Uses Blackwell-native TMA host descriptors for uncompromising pipelining bounds safety.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    log2_e = 1.4426950408889634
    sm_scale_log2 = sm_scale * log2_e
    
    if S == 0:
        return
        
    # Standard highly-tuned FP16/BF16 memory and TMA configurations
    BLOCK_M_Q = 128
    BLOCK_N_Q = 64
    
    BLOCK_M_KV = 64
    BLOCK_N_KV = 128

    # TMA definitions for Pass 1 (dQ local collection)
    desc_q_qpass = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M_Q, d])
    desc_k_qpass = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N_Q, d])
    desc_v_qpass = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N_Q, d])
    desc_o_qpass = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M_Q, d])
    desc_do_qpass = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M_Q, d])
    desc_dq_qpass = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M_Q, d])

    grid_q = (triton.cdiv(S, BLOCK_M_Q), B * H, 1)
    _bwd_q_kernel[grid_q](
        desc_q_qpass, desc_k_qpass, desc_v_qpass, desc_o_qpass, desc_do_qpass, desc_dq_qpass,
        L,
        sm_scale, sm_scale_log2, log2_e,
        L.stride(0), L.stride(1), L.stride(2),
        H=H, S=S,
        d=128, BLOCK_M=BLOCK_M_Q, BLOCK_N=BLOCK_N_Q,
        num_warps=8, num_stages=3
    )

    # TMA definitions for Pass 2 (dK, dV local collection)
    desc_q_kvpass = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M_KV, d])
    desc_k_kvpass = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N_KV, d])
    desc_v_kvpass = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N_KV, d])
    desc_o_kvpass = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M_KV, d])
    desc_do_kvpass = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M_KV, d])
    desc_dk_kvpass = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_N_KV, d])
    desc_dv_kvpass = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_N_KV, d])

    grid_kv = (triton.cdiv(S, BLOCK_N_KV), B * H, 1)
    _bwd_kv_kernel[grid_kv](
        desc_q_kvpass, desc_k_kvpass, desc_v_kvpass, desc_o_kvpass, desc_do_kvpass, desc_dk_kvpass, desc_dv_kvpass,
        L,
        sm_scale, sm_scale_log2, log2_e,
        L.stride(0), L.stride(1), L.stride(2),
        H=H, S=S,
        d=128, BLOCK_M=BLOCK_M_KV, BLOCK_N=BLOCK_N_KV,
        num_warps=8, num_stages=3
    )