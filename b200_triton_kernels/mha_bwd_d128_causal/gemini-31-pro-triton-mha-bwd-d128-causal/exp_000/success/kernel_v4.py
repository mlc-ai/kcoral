import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
    ],
    key=["S"],
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
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    q_bh_offset = pid_b * stride_qb + pid_h * stride_qh
    k_bh_offset = pid_b * stride_kb + pid_h * stride_kh
    v_bh_offset = pid_b * stride_vb + pid_h * stride_vh
    o_bh_offset = pid_b * stride_ob + pid_h * stride_oh
    do_bh_offset = pid_b * stride_dob + pid_h * stride_doh
    l_bh_offset = pid_b * stride_lb + pid_h * stride_lh
    dq_bh_offset = pid_b * stride_dqb + pid_h * stride_dqh
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d)
    mask_m = offs_m < S
    valid_m = mask_m[:, None]
    
    q_ptrs = Q + q_bh_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + o_bh_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + do_bh_offset + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + l_bh_offset + offs_m * stride_ls
    
    q_i = tl.load(q_ptrs, mask=valid_m, other=0.0)
    o_i = tl.load(o_ptrs, mask=valid_m, other=0.0)
    do_i = tl.load(do_ptrs, mask=valid_m, other=0.0)
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute row sums in FP32
    D_i = tl.sum(tl.cast(do_i, tl.float32) * tl.cast(o_i, tl.float32), axis=1)
    
    dq_acc = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    num_blocks_n = tl.cdiv(S, BLOCK_N)
    max_n_idx = (pid_m * BLOCK_M + BLOCK_M - 1) // BLOCK_N
    max_n_idx = tl.minimum(max_n_idx, num_blocks_n - 1)
    
    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = K + k_bh_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + v_bh_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    for n in range(0, max_n_idx + 1):
        mask_n = offs_n < S
        valid_n = mask_n[None, :]
        
        k_j = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_j = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        s_ij = tl.dot(q_i, k_j.T) * scale
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = causal_mask & valid_m & valid_n
        
        p_ij = tl.exp(s_ij - l_i[:, None])
        p_ij = tl.where(valid_mask, p_ij, 0.0)
        
        dp_ij = tl.dot(do_i, v_j.T)
        
        ds_ij = p_ij * (dp_ij - D_i[:, None]) * scale
        
        # Explicit casts to target BF16 Tensor Cores optimally without large active FP32 tiles
        ds_ij_bf16 = tl.cast(ds_ij, Q.dtype.element_ty)
        dq_acc = tl.dot(ds_ij_bf16, k_j, acc=dq_acc)
        
        offs_n += BLOCK_N
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    dq_ptrs = dQ + dq_bh_offset + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(Q.dtype.element_ty), mask=valid_m)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["S"],
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
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    q_bh_offset = pid_b * stride_qb + pid_h * stride_qh
    k_bh_offset = pid_b * stride_kb + pid_h * stride_kh
    v_bh_offset = pid_b * stride_vb + pid_h * stride_vh
    o_bh_offset = pid_b * stride_ob + pid_h * stride_oh
    do_bh_offset = pid_b * stride_dob + pid_h * stride_doh
    l_bh_offset = pid_b * stride_lb + pid_h * stride_lh
    dk_bh_offset = pid_b * stride_dkb + pid_h * stride_dkh
    dv_bh_offset = pid_b * stride_dvb + pid_h * stride_dvh
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    mask_n = offs_n < S
    valid_n = mask_n[:, None]
    
    k_ptrs = K + k_bh_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + v_bh_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k_j = tl.load(k_ptrs, mask=valid_n, other=0.0)
    v_j = tl.load(v_ptrs, mask=valid_n, other=0.0)
    
    dk_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    num_blocks_m = tl.cdiv(S, BLOCK_M)
    start_m = (pid_n * BLOCK_N) // BLOCK_M
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_ptrs = Q + q_bh_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + o_bh_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + do_bh_offset + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + l_bh_offset + offs_m * stride_ls
    
    for m in range(start_m, num_blocks_m):
        mask_m = offs_m < S
        valid_m = mask_m[None, :]
        
        q_i = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o_i = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        D_i = tl.sum(tl.cast(do_i, tl.float32) * tl.cast(o_i, tl.float32), axis=1)
        
        # Express as K_j @ Q_i^T for optimal native Hopper transposed inner product scaling
        s_ji = tl.dot(k_j, q_i.T) * scale
        
        causal_mask = offs_n[:, None] <= offs_m[None, :]
        valid_mask = causal_mask & valid_n & valid_m
        
        p_ji = tl.exp(s_ji - l_i[None, :])
        p_ji = tl.where(valid_mask, p_ji, 0.0)
        
        dp_ji = tl.dot(v_j, do_i.T)
        
        ds_ji = p_ji * (dp_ji - D_i[None, :]) * scale
        
        p_ji_bf16 = tl.cast(p_ji, Q.dtype.element_ty)
        ds_ji_bf16 = tl.cast(ds_ji, Q.dtype.element_ty)
        
        dv_acc = tl.dot(p_ji_bf16, do_i, acc=dv_acc)
        dk_acc = tl.dot(ds_ji_bf16, q_i, acc=dk_acc)
        
        offs_m += BLOCK_M
        q_ptrs += BLOCK_M * stride_qs
        o_ptrs += BLOCK_M * stride_os
        do_ptrs += BLOCK_M * stride_dos
        l_ptrs += BLOCK_M * stride_ls
        
    dk_ptrs = dK + dk_bh_offset + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + dv_bh_offset + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk_acc.to(Q.dtype.element_ty), mask=valid_n)
    tl.store(dv_ptrs, dv_acc.to(Q.dtype.element_ty), mask=valid_n)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal multi-head attention backward avoiding high register pressure 
    and atomic collisions via destination-passing. Native logic maximizes utilization
    of Hopper WGMMA block structure.
    """
    with torch.cuda.device(Q.device):
        B, H, S, D = Q.shape
        scale = 1.0 / math.sqrt(D)
        
        L_3d = L.squeeze(-1) if L.dim() == 4 else L
        
        grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
        bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, L_3d, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L_3d.stride(0), L_3d.stride(1), L_3d.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            B, H, S, scale,
            d=128
        )
        
        grid_dkv = lambda META: (triton.cdiv(S, META["BLOCK_N"]), B * H)
        bwd_dk_dv_kernel[grid_dkv](
            Q, K, V, O, dO, L_3d, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L_3d.stride(0), L_3d.stride(1), L_3d.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            B, H, S, scale,
            d=128
        )
        
        return dQ, dK, dV