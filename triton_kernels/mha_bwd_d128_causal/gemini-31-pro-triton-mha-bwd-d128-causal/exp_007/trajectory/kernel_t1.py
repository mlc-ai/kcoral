import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_dQ_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    i_start = pid_m * BLOCK_M
    if i_start >= S:
        return
        
    offs_m = i_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    mask_m = offs_m < S
    
    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + b * stride_lb + h * stride_lh + offs_m * stride_ls
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    k_ptrs = K + b * stride_kb + h * stride_kh + offs_d[None, :] * stride_kd
    v_ptrs = V + b * stride_vb + h * stride_vh + offs_d[None, :] * stride_vd
    
    j_boundary = (i_start // BLOCK_N) * BLOCK_N
    
    # 1. Unmasked blocks (no causal masking required, j_start + BLOCK_N <= i_start is guaranteed)
    for j_start in range(0, j_boundary, BLOCK_N):
        offs_n = j_start + tl.arange(0, BLOCK_N)
        k_curr = tl.load(k_ptrs + offs_n[:, None] * stride_ks)
        v_curr = tl.load(v_ptrs + offs_n[:, None] * stride_vs)
        
        s_ij = tl.dot(q, tl.trans(k_curr)) * scale
        p_ij = tl.exp(s_ij - lse[:, None])
        
        dp_ij = tl.dot(do, tl.trans(v_curr))
        ds_ij = p_ij * (dp_ij - delta[:, None])
        
        dq = tl.dot((ds_ij * scale).to(tl.bfloat16), k_curr, dq)
        
    # 2. Causal masked blocks (overlap boundary)
    for j_start in range(j_boundary, min(i_start + BLOCK_M, S), BLOCK_N):
        offs_n = j_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        k_curr = tl.load(k_ptrs + offs_n[:, None] * stride_ks, mask=mask_n[:, None], other=0.0)
        v_curr = tl.load(v_ptrs + offs_n[:, None] * stride_vs, mask=mask_n[:, None], other=0.0)
        
        s_ij = tl.dot(q, tl.trans(k_curr)) * scale
        
        causal_mask = offs_n[None, :] <= offs_m[:, None]
        valid_mask = causal_mask & mask_m[:, None] & mask_n[None, :]
        
        s_ij = tl.where(valid_mask, s_ij, float("-inf"))
        p_ij = tl.exp(s_ij - lse[:, None])
        
        dp_ij = tl.dot(do, tl.trans(v_curr))
        ds_ij = p_ij * (dp_ij - delta[:, None])
        
        dq = tl.dot((ds_ij * scale).to(tl.bfloat16), k_curr, dq)

    dq_ptrs = dQ + b * stride_dqb + h * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(tl.bfloat16), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_dKdV_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    j_start = pid_n * BLOCK_N
    if j_start >= S:
        return
        
    offs_n = j_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    mask_n = offs_n < S
    
    k_ptrs = K + b * stride_kb + h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + b * stride_vb + h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_d[None, :] * stride_qd
    o_ptrs = O + b * stride_ob + h * stride_oh + offs_d[None, :] * stride_od
    do_ptrs = dO + b * stride_dob + h * stride_doh + offs_d[None, :] * stride_dod
    l_ptrs = L + b * stride_lb + h * stride_lh
    
    i_boundary = ((j_start + BLOCK_N - 1) // BLOCK_M + 1) * BLOCK_M
    i_start_min = (j_start // BLOCK_M) * BLOCK_M
    
    # 1. Causal masked blocks (overlap boundary)
    for i_start in range(i_start_min, min(i_boundary, S), BLOCK_M):
        offs_m = i_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_curr = tl.load(q_ptrs + offs_m[:, None] * stride_qs, mask=mask_m[:, None], other=0.0)
        o_curr = tl.load(o_ptrs + offs_m[:, None] * stride_os, mask=mask_m[:, None], other=0.0)
        do_curr = tl.load(do_ptrs + offs_m[:, None] * stride_dos, mask=mask_m[:, None], other=0.0)
        lse_curr = tl.load(l_ptrs + offs_m * stride_ls, mask=mask_m, other=0.0)
        
        delta_curr = tl.sum(o_curr.to(tl.float32) * do_curr.to(tl.float32), axis=1)
        
        # Computing S^T prevents dynamic transposed-layout formation in the inner operations.
        s_ji = tl.dot(k, tl.trans(q_curr)) * scale
        
        causal_mask = offs_n[:, None] <= offs_m[None, :]
        valid_mask = causal_mask & mask_n[:, None] & mask_m[None, :]
        
        s_ji = tl.where(valid_mask, s_ji, float("-inf"))
        p_ji = tl.exp(s_ji - lse_curr[None, :])
        
        dv = tl.dot(p_ji.to(tl.bfloat16), do_curr, dv)
        
        dp_ji = tl.dot(v, tl.trans(do_curr))
        ds_ji = p_ji * (dp_ji - delta_curr[None, :])
        
        dk = tl.dot((ds_ji * scale).to(tl.bfloat16), q_curr, dk)
        
    # 2. Unmasked blocks (fully valid pairs)
    for i_start in range(i_boundary, S, BLOCK_M):
        offs_m = i_start + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        q_curr = tl.load(q_ptrs + offs_m[:, None] * stride_qs, mask=mask_m[:, None], other=0.0)
        o_curr = tl.load(o_ptrs + offs_m[:, None] * stride_os, mask=mask_m[:, None], other=0.0)
        do_curr = tl.load(do_ptrs + offs_m[:, None] * stride_dos, mask=mask_m[:, None], other=0.0)
        lse_curr = tl.load(l_ptrs + offs_m * stride_ls, mask=mask_m, other=0.0)
        
        delta_curr = tl.sum(o_curr.to(tl.float32) * do_curr.to(tl.float32), axis=1)
        
        s_ji = tl.dot(k, tl.trans(q_curr)) * scale
        p_ji = tl.exp(s_ji - lse_curr[None, :])
        
        dv = tl.dot(p_ji.to(tl.bfloat16), do_curr, dv)
        
        dp_ji = tl.dot(v, tl.trans(do_curr))
        ds_ji = p_ji * (dp_ji - delta_curr[None, :])
        
        dk = tl.dot((ds_ji * scale).to(tl.bfloat16), q_curr, dk)

    dk_ptrs = dK + b * stride_dkb + h * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + b * stride_dvb + h * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Compute causal SDPA backward and write exactly to the provided dQ, dK, dV outputs.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)

    if L.dim() == 4:
        L = L.squeeze(-1)

    grid_dq = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B * H
    )
    bwd_dQ_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, scale,
        d=d
    )

    grid_dkdv = lambda META: (
        triton.cdiv(S, META["BLOCK_N"]),
        B * H
    )
    bwd_dKdV_kernel[grid_dkdv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, scale,
        d=d
    )