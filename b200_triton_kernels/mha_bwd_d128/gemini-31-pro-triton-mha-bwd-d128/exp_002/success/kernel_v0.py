import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    sm_scale: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    Q_ptr = Q + b * stride_qb + h * stride_qh
    K_ptr = K + b * stride_kb + h * stride_kh
    V_ptr = V + b * stride_vb + h * stride_vh
    O_ptr = O + b * stride_ob + h * stride_oh
    dO_ptr = dO + b * stride_dob + h * stride_doh
    L_ptr = L + b * stride_lb + h * stride_lh
    dQ_ptr = dQ + b * stride_dqb + h * stride_dqh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO_ptr + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L_ptr + offs_m * stride_ls
    
    mask_m = offs_m < S
    mask_m_2d = mask_m[:, None]
    
    Q_m = tl.load(q_ptrs, mask=mask_m_2d, other=0.0)
    O_m = tl.load(o_ptrs, mask=mask_m_2d, other=0.0)
    dO_m = tl.load(do_ptrs, mask=mask_m_2d, other=0.0)
    L_m = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    D_m = tl.sum(tl.cast(O_m, tl.float32) * tl.cast(dO_m, tl.float32), axis=1)
    
    dQ_acc = tl.zeros((BLOCK_M, d), tl.float32)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for start_n in range(0, num_n_blocks):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_ptrs = K_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        K_n = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        V_n = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        S_mn = tl.dot(Q_m, tl.trans(K_n), out_dtype=tl.float32) * sm_scale
        
        mask = mask_m[:, None] & mask_n[None, :]
        S_mn = tl.where(mask, S_mn, float("-inf"))
        
        P_mn = tl.exp(S_mn - L_m[:, None])
        
        dP_mn = tl.dot(dO_m, tl.trans(V_n), out_dtype=tl.float32)
        dS_mn = P_mn * (dP_mn - D_m[:, None])
        
        dS_scaled = tl.cast(dS_mn * sm_scale, tl.bfloat16)
        dQ_acc = tl.dot(dS_scaled, K_n, dQ_acc, out_dtype=tl.float32)

    dq_ptrs = dQ_ptr + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, tl.cast(dQ_acc, tl.bfloat16), mask=mask_m_2d)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    sm_scale: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    Q_ptr = Q + b * stride_qb + h * stride_qh
    K_ptr = K + b * stride_kb + h * stride_kh
    V_ptr = V + b * stride_vb + h * stride_vh
    O_ptr = O + b * stride_ob + h * stride_oh
    dO_ptr = dO + b * stride_dob + h * stride_doh
    L_ptr = L + b * stride_lb + h * stride_lh
    dK_ptr = dK + b * stride_dkb + h * stride_dkh
    dV_ptr = dV + b * stride_dvb + h * stride_dvh

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    mask_n = offs_n < S
    mask_n_2d = mask_n[:, None]

    k_ptrs = K_ptr + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V_ptr + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    K_n = tl.load(k_ptrs, mask=mask_n_2d, other=0.0)
    V_n = tl.load(v_ptrs, mask=mask_n_2d, other=0.0)

    dK_acc = tl.zeros((BLOCK_N, d), tl.float32)
    dV_acc = tl.zeros((BLOCK_N, d), tl.float32)

    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for start_m in range(0, num_m_blocks):
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        mask_m_2d = mask_m[:, None]
        
        q_ptrs = Q_ptr + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        o_ptrs = O_ptr + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        do_ptrs = dO_ptr + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        l_ptrs = L_ptr + offs_m * stride_ls

        Q_m = tl.load(q_ptrs, mask=mask_m_2d, other=0.0)
        O_m = tl.load(o_ptrs, mask=mask_m_2d, other=0.0)
        dO_m = tl.load(do_ptrs, mask=mask_m_2d, other=0.0)
        L_m = tl.load(l_ptrs, mask=mask_m, other=0.0)

        D_m = tl.sum(tl.cast(O_m, tl.float32) * tl.cast(dO_m, tl.float32), axis=1)

        S_mn = tl.dot(Q_m, tl.trans(K_n), out_dtype=tl.float32) * sm_scale
        
        mask = mask_m[:, None] & mask_n[None, :]
        S_mn = tl.where(mask, S_mn, float("-inf"))
        
        P_mn = tl.exp(S_mn - L_m[:, None])
        
        P_b16 = tl.cast(P_mn, tl.bfloat16)
        dV_acc = tl.dot(tl.trans(P_b16), dO_m, dV_acc, out_dtype=tl.float32)
        
        dP_mn = tl.dot(dO_m, tl.trans(V_n), out_dtype=tl.float32)
        dS_mn = P_mn * (dP_mn - D_m[:, None])
        
        dS_scaled = tl.cast(dS_mn * sm_scale, tl.bfloat16)
        dK_acc = tl.dot(tl.trans(dS_scaled), Q_m, dK_acc, out_dtype=tl.float32)

    dk_ptrs = dK_ptr + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV_ptr + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, tl.cast(dK_acc, tl.bfloat16), mask=mask_n_2d)
    tl.store(dv_ptrs, tl.cast(dV_acc, tl.bfloat16), mask=mask_n_2d)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes gradients dQ, dK, dV given multi-head attention inputs and grad output.
    All inputs and outputs are bfloat16, except L which is float32.
    """
    torch.cuda.set_device(Q.device)
    
    # Handle possible extra dummy dimensions on L based on reference code.
    if L.dim() == 4:
        L = L.squeeze(-1)
        
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)
    
    grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H, 1)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, d
    )
    
    grid_dk_dv = lambda META: (triton.cdiv(S, META["BLOCK_N"]), B * H, 1)
    bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, O, dO, L, dK, dV,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, d
    )