import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    scale, S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    
    offset_q_bh = b_idx * stride_qb + h_idx * stride_qh
    offset_k_bh = b_idx * stride_kb + h_idx * stride_kh
    offset_v_bh = b_idx * stride_vb + h_idx * stride_vh
    offset_o_bh = b_idx * stride_ob + h_idx * stride_oh
    offset_do_bh = b_idx * stride_dob + h_idx * stride_doh
    offset_l_bh = b_idx * stride_lb + h_idx * stride_lh
    
    q_ptrs = Q + offset_q_bh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + offset_o_bh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + offset_do_bh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + offset_l_bh + offs_m * stride_ls
    
    mask_m = offs_m < S
    
    q_i = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o_i = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    Di = tl.sum((do_i.to(tl.float32)) * (o_i.to(tl.float32)), axis=1)
    
    dq_i = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    max_j = tl.minimum(S, (pid_m + 1) * BLOCK_M)
    num_j_blocks = tl.cdiv(max_j, BLOCK_N)
    
    k_base = K + offset_k_bh + offs_d[None, :] * stride_kd
    v_base = V + offset_v_bh + offs_d[None, :] * stride_vd
    
    for j_block in range(num_j_blocks):
        offs_n = j_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_ptrs_j = k_base + offs_n[:, None] * stride_ks
        v_ptrs_j = v_base + offs_n[:, None] * stride_vs
        
        k_j = tl.load(k_ptrs_j, mask=mask_n[:, None], other=0.0)
        v_j = tl.load(v_ptrs_j, mask=mask_n[:, None], other=0.0)
        
        s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid = causal_mask & mask_m[:, None] & mask_n[None, :]
        
        p_ij = tl.exp(s_ij - l_i[:, None])
        p_ij = tl.where(valid, p_ij, 0.0)
        
        dp_ij = tl.dot(do_i, tl.trans(v_j))
        ds_ij = p_ij * (dp_ij - Di[:, None])
        
        dq_i += tl.dot(ds_ij.to(q_i.dtype), k_j)
        
    dq_i = dq_i * scale
    
    offset_dq_bh = b_idx * stride_dqb + h_idx * stride_dqh
    dq_ptrs = dQ + offset_dq_bh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_i.to(dQ.dtype.element_ty), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    scale, S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    
    offset_k_bh = b_idx * stride_kb + h_idx * stride_kh
    offset_v_bh = b_idx * stride_vb + h_idx * stride_vh
    
    k_ptrs = K + offset_k_bh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + offset_v_bh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    mask_n = offs_n < S
    
    k_j = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v_j = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk_j = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv_j = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    start_i_block = (pid_n * BLOCK_N) // BLOCK_M
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    
    offset_q_bh = b_idx * stride_qb + h_idx * stride_qh
    offset_o_bh = b_idx * stride_ob + h_idx * stride_oh
    offset_do_bh = b_idx * stride_dob + h_idx * stride_doh
    offset_l_bh = b_idx * stride_lb + h_idx * stride_lh
    
    q_base = Q + offset_q_bh + offs_d[None, :] * stride_qd
    o_base = O + offset_o_bh + offs_d[None, :] * stride_od
    do_base = dO + offset_do_bh + offs_d[None, :] * stride_dod
    l_base = L + offset_l_bh
    
    for i_block in range(start_i_block, num_m_blocks):
        offs_m = i_block * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_ptrs_i = q_base + offs_m[:, None] * stride_qs
        o_ptrs_i = o_base + offs_m[:, None] * stride_os
        do_ptrs_i = do_base + offs_m[:, None] * stride_dos
        l_ptrs_i = l_base + offs_m * stride_ls
        
        q_i = tl.load(q_ptrs_i, mask=mask_m[:, None], other=0.0)
        o_i = tl.load(o_ptrs_i, mask=mask_m[:, None], other=0.0)
        do_i = tl.load(do_ptrs_i, mask=mask_m[:, None], other=0.0)
        l_i = tl.load(l_ptrs_i, mask=mask_m, other=0.0)
        
        Di = tl.sum((do_i.to(tl.float32)) * (o_i.to(tl.float32)), axis=1)
        
        s_ij = tl.dot(q_i, tl.trans(k_j)) * scale
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid = causal_mask & mask_m[:, None] & mask_n[None, :]
        
        p_ij = tl.exp(s_ij - l_i[:, None])
        p_ij = tl.where(valid, p_ij, 0.0)
        
        dv_j += tl.dot(tl.trans(p_ij.to(do_i.dtype)), do_i)
        
        dp_ij = tl.dot(do_i, tl.trans(v_j))
        ds_ij = p_ij * (dp_ij - Di[:, None])
        
        dk_j += tl.dot(tl.trans(ds_ij.to(q_i.dtype)), q_i)
        
    dk_j = dk_j * scale
    
    offset_dk_bh = b_idx * stride_dkb + h_idx * stride_dkh
    offset_dv_bh = b_idx * stride_dvb + h_idx * stride_dvh
    
    dk_ptrs = dK + offset_dk_bh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + offset_dv_bh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk_j.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv_j.to(dV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        
        grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
        bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            scale, S, H,
            d=128
        )
        
        grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
        bwd_dk_dv_kernel[grid_dk_dv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            scale, S, H,
            d=128
        )