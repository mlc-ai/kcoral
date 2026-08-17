import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
    ],
    key=['seq_len'],
)
@triton.jit
def _bwd_kernel_dk_dv(
    Q, K, V, sm_scale, Out, DO,
    DK, DV,
    L,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    stride_l_b, stride_l_h, stride_l_s,
    num_heads, seq_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    start_n = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    b_id = bh_id // num_heads
    h_id = bh_id % num_heads

    offset_q = b_id * stride_q_b + h_id * stride_q_h
    offset_k = b_id * stride_k_b + h_id * stride_k_h
    offset_v = b_id * stride_v_b + h_id * stride_v_h
    offset_o = b_id * stride_o_b + h_id * stride_o_h
    offset_do = b_id * stride_do_b + h_id * stride_do_h
    offset_dk = b_id * stride_dk_b + h_id * stride_dk_h
    offset_dv = b_id * stride_dv_b + h_id * stride_dv_h
    offset_l = b_id * stride_l_b + h_id * stride_l_h

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    mask_n = offs_n < seq_len
    
    k_ptrs = K + offset_k + offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
    v_ptrs = V + offset_v + offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    
    num_m_blocks = tl.cdiv(seq_len, BLOCK_M)
    
    for start_m in range(0, num_m_blocks):
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len
        
        q_ptrs = Q + offset_q + offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
        do_ptrs = DO + offset_do + offs_m[:, None] * stride_do_s + offs_d[None, :] * stride_do_d
        o_ptrs = Out + offset_o + offs_m[:, None] * stride_o_s + offs_d[None, :] * stride_o_d
        l_ptrs = L + offset_l + offs_m * stride_l_s
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        # Precompute D = rowsum(dO * O) on the fly for the current Q block
        d_i = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        
        p = tl.exp(qk - l_i[:, None])
        p = tl.where((mask_m[:, None] & mask_n[None, :]), p, 0.0)
        
        dp = tl.dot(do, tl.trans(v))
        
        ds = p * (dp - d_i[:, None])
        ds = ds * sm_scale
        
        p_bf16 = p.to(q.dtype)
        ds_bf16 = ds.to(q.dtype)
        
        dv = tl.dot(tl.trans(p_bf16), do, acc=dv)
        dk = tl.dot(tl.trans(ds_bf16), q, acc=dk)
        
    dk_ptrs = DK + offset_dk + offs_n[:, None] * stride_dk_s + offs_d[None, :] * stride_dk_d
    dv_ptrs = DV + offset_dv + offs_n[:, None] * stride_dv_s + offs_d[None, :] * stride_dv_d
    
    tl.store(dk_ptrs, dk.to(q.dtype), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(q.dtype), mask=mask_n[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
    ],
    key=['seq_len'],
)
@triton.jit
def _bwd_kernel_dq(
    Q, K, V, sm_scale, Out, DO,
    DQ,
    L,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    stride_l_b, stride_l_h, stride_l_s,
    num_heads, seq_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    start_m = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    b_id = bh_id // num_heads
    h_id = bh_id % num_heads
    
    offset_q = b_id * stride_q_b + h_id * stride_q_h
    offset_k = b_id * stride_k_b + h_id * stride_k_h
    offset_v = b_id * stride_v_b + h_id * stride_v_h
    offset_o = b_id * stride_o_b + h_id * stride_o_h
    offset_do = b_id * stride_do_b + h_id * stride_do_h
    offset_dq = b_id * stride_dq_b + h_id * stride_dq_h
    offset_l = b_id * stride_l_b + h_id * stride_l_h

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    mask_m = offs_m < seq_len
    
    q_ptrs = Q + offset_q + offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    do_ptrs = DO + offset_do + offs_m[:, None] * stride_do_s + offs_d[None, :] * stride_do_d
    o_ptrs = Out + offset_o + offs_m[:, None] * stride_o_s + offs_d[None, :] * stride_o_d
    l_ptrs = L + offset_l + offs_m * stride_l_s
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute D = rowsum(dO * O) on the fly for the current Q block
    d_i = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    num_n_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    for start_n in range(0, num_n_blocks):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        k_ptrs = K + offset_k + offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
        v_ptrs = V + offset_v + offs_n[:, None] * stride_v_s + offs_d[None, :] * stride_v_d
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        qk = tl.dot(q, tl.trans(k))
        qk = qk * sm_scale
        
        p = tl.exp(qk - l_i[:, None])
        p = tl.where((mask_m[:, None] & mask_n[None, :]), p, 0.0)
        
        dp = tl.dot(do, tl.trans(v))
        
        ds = p * (dp - d_i[:, None])
        ds = ds * sm_scale
        
        ds_bf16 = ds.to(q.dtype)
        dq = tl.dot(ds_bf16, k, acc=dq)
        
    dq_ptrs = DQ + offset_dq + offs_m[:, None] * stride_dq_s + offs_d[None, :] * stride_dq_d
    tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sm_scale = 1.0 / (d ** 0.5)
    
    if S == 0:
        return
    
    grid_dkdv = lambda META: (
        triton.cdiv(S, META['BLOCK_N']),
        B * H
    )
    _bwd_kernel_dk_dv[grid_dkdv](
        Q, K, V, sm_scale, O, dO,
        dK, dV,
        L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        H, S,
        BLOCK_DMODEL=d
    )
    
    grid_dq = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H
    )
    _bwd_kernel_dq[grid_dq](
        Q, K, V, sm_scale, O, dO,
        dQ,
        L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        H, S,
        BLOCK_DMODEL=d
    )