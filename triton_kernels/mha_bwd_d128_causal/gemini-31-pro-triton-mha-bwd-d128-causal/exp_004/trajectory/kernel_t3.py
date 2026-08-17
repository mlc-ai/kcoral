import torch
import triton
import triton.language as tl

_configs_dq = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
]

@triton.autotune(
    configs=_configs_dq,
    key=["S"],
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L,
    dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, H: tl.constexpr, d: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    if pid_m * BLOCK_M >= S:
        return
        
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = off_m < S
    
    off_b_h_q = pid_b * stride_qb + pid_h * stride_qh
    off_b_h_do = pid_b * stride_dob + pid_h * stride_doh
    off_b_h_o = pid_b * stride_ob + pid_h * stride_oh
    
    Q_ptr = Q + off_b_h_q + off_m[:, None] * stride_qs + tl.arange(0, d)[None, :] * stride_qd
    dO_ptr = dO + off_b_h_do + off_m[:, None] * stride_dos + tl.arange(0, d)[None, :] * stride_dod
    O_ptr = O + off_b_h_o + off_m[:, None] * stride_os + tl.arange(0, d)[None, :] * stride_od
    
    q = tl.load(Q_ptr, mask=mask_m[:, None], other=0.0)
    do = tl.load(dO_ptr, mask=mask_m[:, None], other=0.0)
    o = tl.load(O_ptr, mask=mask_m[:, None], other=0.0)
    
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    off_b_h_l = pid_b * stride_lb + pid_h * stride_lh
    L_ptr = L + off_b_h_l + off_m * stride_ls
    l = tl.load(L_ptr, mask=mask_m, other=0.0)
    
    dq_acc = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    off_b_h_k = pid_b * stride_kb + pid_h * stride_kh
    off_b_h_v = pid_b * stride_vb + pid_h * stride_vh
    
    K_ptr_base = K + off_b_h_k + tl.arange(0, BLOCK_N)[:, None] * stride_ks + tl.arange(0, d)[None, :] * stride_kd
    V_ptr_base = V + off_b_h_v + tl.arange(0, BLOCK_N)[:, None] * stride_vs + tl.arange(0, d)[None, :] * stride_vd
    
    end_n_full = (pid_m * BLOCK_M) // BLOCK_N * BLOCK_N
    end_n = tl.minimum((pid_m + 1) * BLOCK_M, S)
    end_n_aligned = ((end_n + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    
    for start_n in tl.range(0, end_n_full, BLOCK_N, num_stages=3):
        K_ptr = K_ptr_base + start_n * stride_ks
        V_ptr = V_ptr_base + start_n * stride_vs
        
        k = tl.load(K_ptr)
        v = tl.load(V_ptr)
        
        s_mat = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        p = tl.exp(s_mat - l[:, None])
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = (dp - d_val[:, None]) * p * scale
        
        dq_acc += tl.dot(ds.to(tl.bfloat16), k, out_dtype=tl.float32)
        
    for start_n in range(end_n_full, end_n_aligned, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < S
        
        K_ptr = K_ptr_base + start_n * stride_ks
        V_ptr = V_ptr_base + start_n * stride_vs
        
        k = tl.load(K_ptr, mask=mask_n[:, None], other=0.0)
        v = tl.load(V_ptr, mask=mask_n[:, None], other=0.0)
        
        s_mat = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        causal_mask = (off_m[:, None] >= off_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        s_mat = tl.where(causal_mask, s_mat, float("-inf"))
        
        p = tl.exp(s_mat - l[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = (dp - d_val[:, None]) * p * scale
        
        dq_acc += tl.dot(ds.to(tl.bfloat16), k, out_dtype=tl.float32)
        
    off_b_h_dq = pid_b * stride_dqb + pid_h * stride_dqh
    dQ_ptr = dQ + off_b_h_dq + off_m[:, None] * stride_dqs + tl.arange(0, d)[None, :] * stride_dqd
    tl.store(dQ_ptr, dq_acc.to(dQ.dtype.element_ty), mask=mask_m[:, None])

_configs_dkdv = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
]

@triton.autotune(
    configs=_configs_dkdv,
    key=["S"],
)
@triton.jit
def bwd_dkdv_kernel(
    Q, K, V, O, dO, L,
    dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, H: tl.constexpr, d: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    if pid_n * BLOCK_N >= S:
        return
        
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = off_n < S
    
    off_b_h_k = pid_b * stride_kb + pid_h * stride_kh
    off_b_h_v = pid_b * stride_vb + pid_h * stride_vh
    
    K_ptr = K + off_b_h_k + off_n[:, None] * stride_ks + tl.arange(0, d)[None, :] * stride_kd
    V_ptr = V + off_b_h_v + off_n[:, None] * stride_vs + tl.arange(0, d)[None, :] * stride_vd
    
    k = tl.load(K_ptr, mask=mask_n[:, None], other=0.0)
    v = tl.load(V_ptr, mask=mask_n[:, None], other=0.0)
    
    dk_acc = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    off_b_h_q = pid_b * stride_qb + pid_h * stride_qh
    off_b_h_do = pid_b * stride_dob + pid_h * stride_doh
    off_b_h_o = pid_b * stride_ob + pid_h * stride_oh
    off_b_h_l = pid_b * stride_lb + pid_h * stride_lh
    
    Q_ptr_base = Q + off_b_h_q + tl.arange(0, BLOCK_M)[:, None] * stride_qs + tl.arange(0, d)[None, :] * stride_qd
    dO_ptr_base = dO + off_b_h_do + tl.arange(0, BLOCK_M)[:, None] * stride_dos + tl.arange(0, d)[None, :] * stride_dod
    O_ptr_base = O + off_b_h_o + tl.arange(0, BLOCK_M)[:, None] * stride_os + tl.arange(0, d)[None, :] * stride_od
    L_ptr_base = L + off_b_h_l + tl.arange(0, BLOCK_M) * stride_ls
    
    start_m_initial = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M
    end_m_diagonal = tl.minimum(pid_n * BLOCK_N + BLOCK_N, S)
    end_m_diagonal_aligned = ((end_m_diagonal + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    
    for start_m in range(start_m_initial, end_m_diagonal_aligned, BLOCK_M):
        off_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = off_m < S
        
        Q_ptr = Q_ptr_base + start_m * stride_qs
        dO_ptr = dO_ptr_base + start_m * stride_dos
        O_ptr = O_ptr_base + start_m * stride_os
        
        q = tl.load(Q_ptr, mask=mask_m[:, None], other=0.0)
        do = tl.load(dO_ptr, mask=mask_m[:, None], other=0.0)
        o = tl.load(O_ptr, mask=mask_m[:, None], other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        s_mat = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        causal_mask = (off_m[:, None] >= off_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        s_mat = tl.where(causal_mask, s_mat, float("-inf"))
        
        L_ptr = L_ptr_base + start_m * stride_ls
        l = tl.load(L_ptr, mask=mask_m, other=0.0)
        
        p = tl.exp(s_mat - l[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = (dp - d_val[:, None]) * p * scale
        
        dv_acc += tl.dot(tl.trans(p.to(tl.bfloat16)), do, out_dtype=tl.float32)
        dk_acc += tl.dot(tl.trans(ds.to(tl.bfloat16)), q, out_dtype=tl.float32)
        
    start_m_full = end_m_diagonal_aligned
    end_m_full = (S // BLOCK_M) * BLOCK_M
    loop_end = tl.maximum(start_m_full, end_m_full)
    
    for start_m in tl.range(start_m_full, loop_end, BLOCK_M, num_stages=3):
        Q_ptr = Q_ptr_base + start_m * stride_qs
        dO_ptr = dO_ptr_base + start_m * stride_dos
        O_ptr = O_ptr_base + start_m * stride_os
        
        q = tl.load(Q_ptr)
        do = tl.load(dO_ptr)
        o = tl.load(O_ptr)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        s_mat = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        L_ptr = L_ptr_base + start_m * stride_ls
        l = tl.load(L_ptr)
        
        p = tl.exp(s_mat - l[:, None])
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = (dp - d_val[:, None]) * p * scale
        
        dv_acc += tl.dot(tl.trans(p.to(tl.bfloat16)), do, out_dtype=tl.float32)
        dk_acc += tl.dot(tl.trans(ds.to(tl.bfloat16)), q, out_dtype=tl.float32)
        
    if loop_end < S and loop_end >= start_m_full:
        start_m = loop_end
        off_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = off_m < S
        
        Q_ptr = Q_ptr_base + start_m * stride_qs
        dO_ptr = dO_ptr_base + start_m * stride_dos
        O_ptr = O_ptr_base + start_m * stride_os
        
        q = tl.load(Q_ptr, mask=mask_m[:, None], other=0.0)
        do = tl.load(dO_ptr, mask=mask_m[:, None], other=0.0)
        o = tl.load(O_ptr, mask=mask_m[:, None], other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        s_mat = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        L_ptr = L_ptr_base + start_m * stride_ls
        l = tl.load(L_ptr, mask=mask_m, other=0.0)
        
        p = tl.exp(s_mat - l[:, None])
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = (dp - d_val[:, None]) * p * scale
        
        dv_acc += tl.dot(tl.trans(p.to(tl.bfloat16)), do, out_dtype=tl.float32)
        dk_acc += tl.dot(tl.trans(ds.to(tl.bfloat16)), q, out_dtype=tl.float32)
        
    off_b_h_dk = pid_b * stride_dkb + pid_h * stride_dkh
    off_b_h_dv = pid_b * stride_dvb + pid_h * stride_dvh
    
    dK_ptr = dK + off_b_h_dk + off_n[:, None] * stride_dks + tl.arange(0, d)[None, :] * stride_dkd
    dV_ptr = dV + off_b_h_dv + off_n[:, None] * stride_dvs + tl.arange(0, d)[None, :] * stride_dvd
    
    tl.store(dK_ptr, dk_acc.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dV_ptr, dv_acc.to(dV.dtype.element_ty), mask=mask_n[:, None])

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L,
        dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, H=H, d=d,
        scale=scale,
    )
    
    grid_dkdv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    
    bwd_dkdv_kernel[grid_dkdv](
        Q, K, V, O, dO, L,
        dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, H=H, d=d,
        scale=scale,
    )