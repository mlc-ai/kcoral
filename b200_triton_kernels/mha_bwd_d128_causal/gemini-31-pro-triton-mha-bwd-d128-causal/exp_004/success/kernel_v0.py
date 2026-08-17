import torch
import triton
import triton.language as tl

_configs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
]

@triton.autotune(
    configs=_configs,
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
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
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
    
    end_n = (pid_m * BLOCK_M + BLOCK_M)
    end_n = tl.minimum(end_n, S)
    
    off_b_h_k = pid_b * stride_kb + pid_h * stride_kh
    off_b_h_v = pid_b * stride_vb + pid_h * stride_vh
    
    for start_n in range(0, end_n, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < S
        
        K_ptr = K + off_b_h_k + off_n[:, None] * stride_ks + tl.arange(0, d)[None, :] * stride_kd
        V_ptr = V + off_b_h_v + off_n[:, None] * stride_vs + tl.arange(0, d)[None, :] * stride_vd
        
        k = tl.load(K_ptr, mask=mask_n[:, None], other=0.0)
        v = tl.load(V_ptr, mask=mask_n[:, None], other=0.0)
        
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        is_valid = mask_m[:, None] & mask_n[None, :]
        causal_mask = (off_m[:, None] >= off_n[None, :]) & is_valid
        s = tl.where(causal_mask, s, float("-inf"))
        
        p = tl.exp(s - l[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        ds = (dp - d_val[:, None]) * p * scale
        
        ds_bf16 = ds.to(tl.bfloat16)
        dq_acc += tl.dot(ds_bf16, k, out_dtype=tl.float32)
        
    off_b_h_dq = pid_b * stride_dqb + pid_h * stride_dqh
    dQ_ptr = dQ + off_b_h_dq + off_m[:, None] * stride_dqs + tl.arange(0, d)[None, :] * stride_dqd
    tl.store(dQ_ptr, dq_acc.to(dQ.dtype.element_ty), mask=mask_m[:, None])


@triton.autotune(
    configs=_configs,
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
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = off_n < S
    
    dk_acc = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    off_b_h_k = pid_b * stride_kb + pid_h * stride_kh
    off_b_h_v = pid_b * stride_vb + pid_h * stride_vh
    
    K_ptr = K + off_b_h_k + off_n[:, None] * stride_ks + tl.arange(0, d)[None, :] * stride_kd
    V_ptr = V + off_b_h_v + off_n[:, None] * stride_vs + tl.arange(0, d)[None, :] * stride_vd
    
    k = tl.load(K_ptr, mask=mask_n[:, None], other=0.0)
    v = tl.load(V_ptr, mask=mask_n[:, None], other=0.0)
    
    start_m = pid_n * BLOCK_N
    start_m = (start_m // BLOCK_M) * BLOCK_M
    
    off_b_h_q = pid_b * stride_qb + pid_h * stride_qh
    off_b_h_do = pid_b * stride_dob + pid_h * stride_doh
    off_b_h_o = pid_b * stride_ob + pid_h * stride_oh
    off_b_h_l = pid_b * stride_lb + pid_h * stride_lh
    
    for start_m_idx in range(start_m, S, BLOCK_M):
        off_m = start_m_idx + tl.arange(0, BLOCK_M)
        mask_m = off_m < S
        
        Q_ptr = Q + off_b_h_q + off_m[:, None] * stride_qs + tl.arange(0, d)[None, :] * stride_qd
        dO_ptr = dO + off_b_h_do + off_m[:, None] * stride_dos + tl.arange(0, d)[None, :] * stride_dod
        O_ptr = O + off_b_h_o + off_m[:, None] * stride_os + tl.arange(0, d)[None, :] * stride_od
        
        q = tl.load(Q_ptr, mask=mask_m[:, None], other=0.0)
        do = tl.load(dO_ptr, mask=mask_m[:, None], other=0.0)
        o = tl.load(O_ptr, mask=mask_m[:, None], other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        is_valid = mask_m[:, None] & mask_n[None, :]
        causal_mask = (off_m[:, None] >= off_n[None, :]) & is_valid
        s = tl.where(causal_mask, s, float("-inf"))
        
        L_ptr = L + off_b_h_l + off_m * stride_ls
        l = tl.load(L_ptr, mask=mask_m, other=0.0)
        
        p = tl.exp(s - l[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        ds = (dp - d_val[:, None]) * p * scale
        
        p_bf16 = p.to(tl.bfloat16)
        dv_acc += tl.dot(tl.trans(p_bf16), do, out_dtype=tl.float32)
        
        ds_bf16 = ds.to(tl.bfloat16)
        dk_acc += tl.dot(tl.trans(ds_bf16), q, out_dtype=tl.float32)
        
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