import math
import torch
import triton
import triton.language as tl

def get_autotune_config_dq():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ]

@triton.autotune(
    configs=get_autotune_config_dq(),
    key=['S']
)
@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    off_b_h_q = pid_b * stride_qb + pid_h * stride_qh
    off_b_h_do = pid_b * stride_dob + pid_h * stride_doh
    off_b_h_o = pid_b * stride_ob + pid_h * stride_oh
    off_b_h_l = pid_b * stride_lb + pid_h * stride_lh
    off_b_h_dq = pid_b * stride_dqb + pid_h * stride_dqh

    q_ptrs = Q_ptr + off_b_h_q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO_ptr + off_b_h_do + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O_ptr + off_b_h_o + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    dq_ptrs = dQ_ptr + off_b_h_dq + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    l_ptrs = L_ptr + off_b_h_l + offs_m * stride_ls

    mask_m = offs_m < S
    mask_m_2d = offs_m[:, None] < S

    q = tl.load(q_ptrs, mask=mask_m_2d, other=0.0)
    do = tl.load(do_ptrs, mask=mask_m_2d, other=0.0)
    o = tl.load(o_ptrs, mask=mask_m_2d, other=0.0)
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)

    Di = tl.sum((do.to(tl.float32) * o.to(tl.float32)), axis=1)

    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    off_b_h_k = pid_b * stride_kb + pid_h * stride_kh
    off_b_h_v = pid_b * stride_vb + pid_h * stride_vh

    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n_block in range(num_n_blocks):
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n_1d = offs_n < S
        mask_n_2d = offs_n[:, None] < S

        k_ptrs = K_ptr + off_b_h_k + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + off_b_h_v + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=mask_n_2d, other=0.0)
        v = tl.load(v_ptrs, mask=mask_n_2d, other=0.0)

        qk = tl.dot(q, tl.trans(k))
        qk = qk * scale

        mask_2d = mask_m_2d & mask_n_1d[None, :]
        qk = tl.where(mask_2d, qk, float('-inf'))

        p = tl.exp(qk - l_i[:, None])

        dp = tl.dot(do, tl.trans(v))

        ds = p * (dp - Di[:, None]) * scale

        dq += tl.dot(ds.to(q.dtype), k)

    tl.store(dq_ptrs, dq.to(dQ_ptr.dtype.element_ty), mask=mask_m_2d)


def get_autotune_config_dkdv():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ]

@triton.autotune(
    configs=get_autotune_config_dkdv(),
    key=['S']
)
@triton.jit
def bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_n_1d = offs_n < S
    mask_n_2d = offs_n[:, None] < S

    off_b_h_k = pid_b * stride_kb + pid_h * stride_kh
    off_b_h_v = pid_b * stride_vb + pid_h * stride_vh
    off_b_h_dk = pid_b * stride_dkb + pid_h * stride_dkh
    off_b_h_dv = pid_b * stride_dvb + pid_h * stride_dvh

    k_ptrs = K_ptr + off_b_h_k + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V_ptr + off_b_h_v + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    k = tl.load(k_ptrs, mask=mask_n_2d, other=0.0)
    v = tl.load(v_ptrs, mask=mask_n_2d, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    off_b_h_q = pid_b * stride_qb + pid_h * stride_qh
    off_b_h_do = pid_b * stride_dob + pid_h * stride_doh
    off_b_h_o = pid_b * stride_ob + pid_h * stride_oh
    off_b_h_l = pid_b * stride_lb + pid_h * stride_lh

    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for m_block in range(num_m_blocks):
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m_1d = offs_m < S
        mask_m_2d = offs_m[:, None] < S

        q_ptrs = Q_ptr + off_b_h_q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        do_ptrs = dO_ptr + off_b_h_do + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        o_ptrs = O_ptr + off_b_h_o + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        l_ptrs = L_ptr + off_b_h_l + offs_m * stride_ls

        q = tl.load(q_ptrs, mask=mask_m_2d, other=0.0)
        do = tl.load(do_ptrs, mask=mask_m_2d, other=0.0)
        o = tl.load(o_ptrs, mask=mask_m_2d, other=0.0)
        l_i = tl.load(l_ptrs, mask=mask_m_1d, other=0.0)

        Di = tl.sum((do.to(tl.float32) * o.to(tl.float32)), axis=1)

        qk = tl.dot(q, tl.trans(k))
        qk = qk * scale

        mask_2d = mask_m_2d & mask_n_1d[None, :]
        qk = tl.where(mask_2d, qk, float('-inf'))

        p = tl.exp(qk - l_i[:, None])

        dp = tl.dot(do, tl.trans(v))

        ds = p * (dp - Di[:, None]) * scale

        dv += tl.dot(tl.trans(p.to(q.dtype)), do)
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    dk_ptrs = dK_ptr + off_b_h_dk + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV_ptr + off_b_h_dv + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd

    tl.store(dk_ptrs, dk.to(dK_ptr.dtype.element_ty), mask=mask_n_2d)
    tl.store(dv_ptrs, dv.to(dV_ptr.dtype.element_ty), mask=mask_n_2d)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    lse = L.squeeze(-1) if L.dim() == 4 else L

    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), H, B)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, lse, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, scale,
        BLOCK_D=128
    )

    grid_dkdv = lambda META: (triton.cdiv(S, META['BLOCK_N']), H, B)
    bwd_dkdv_kernel[grid_dkdv](
        Q, K, V, O, dO, lse, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, scale,
        BLOCK_D=128
    )