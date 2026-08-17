import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # SMEM math: Q, dO, O (Resident: 96KB) + K, V (2 stages: 64KB * 2 = 128KB) = 224KB <= 227KB limit
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        # SMEM math: Resident 96KB + K, V (4 stages: 32KB * 4 = 128KB) = 224KB <= 227KB limit
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        # SMEM math: Resident 48KB + K, V (2 stages: 64KB * 2 = 128KB) = 176KB <= 227KB limit
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
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
    S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // 48
    h = pid_bh % 48

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    
    Q_base = Q + b * stride_qb + h * stride_qh
    K_base = K + b * stride_kb + h * stride_kh
    V_base = V + b * stride_vb + h * stride_vh
    O_base = O + b * stride_ob + h * stride_oh
    dO_base = dO + b * stride_dob + h * stride_doh
    L_base = L + b * stride_lb + h * stride_lh
    
    mask_m = offs_m < S
    
    q_ptrs = Q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    l_ptrs = L_base + offs_m * stride_ls
    
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    l_val = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Precompute per-program Delta scalar row values locally
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    
    k_ptrs = K_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    num_n = tl.cdiv(S, BLOCK_N)
    for n0 in range(num_n):
        mask_n = (n0 * BLOCK_N + offs_n) < S
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, float("-inf"))
        
        p = tl.math.exp(scores - l_val[:, None])
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * scale
        
        dq += tl.dot(ds.to(tl.bfloat16), k)
        
        # Advance pointers safely outside of implicit software pipelines
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    dQ_base = dQ + b * stride_dqb + h * stride_dqh
    dq_ptrs = dQ_base + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=mask_m[:, None])


@triton.autotune(
    configs=[
        # SMEM math: K, V (Resident: 64KB) + Q, dO, O (3 stages: 48KB * 3 = 144KB) = 208KB <= 227KB limit
        triton.Config({"BLOCK_N": 128, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_M": 64}, num_warps=8, num_stages=3),
        # SMEM math: Resident 32KB + Q, dO, O (2 stages: 96KB * 2 = 192KB) = 224KB <= 227KB limit 
        triton.Config({"BLOCK_N": 64, "BLOCK_M": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_M": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_M": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_dkdv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, scale,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // 48
    h = pid_bh % 48

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    Q_base = Q + b * stride_qb + h * stride_qh
    K_base = K + b * stride_kb + h * stride_kh
    V_base = V + b * stride_vb + h * stride_vh
    O_base = O + b * stride_ob + h * stride_oh
    dO_base = dO + b * stride_dob + h * stride_doh
    L_base = L + b * stride_lb + h * stride_lh
    
    mask_n = offs_n < S
    
    k_ptrs = K_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    
    q_ptrs = Q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    l_ptrs = L_base + offs_m * stride_ls
    
    num_m = tl.cdiv(S, BLOCK_M)
    for m0 in range(num_m):
        mask_m = (m0 * BLOCK_M + offs_m) < S
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        l_val = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        scores_t = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        scores_t = tl.where(mask_n[:, None] & mask_m[None, :], scores_t, float("-inf"))
        
        p_t = tl.math.exp(scores_t - l_val[None, :])
        
        dv += tl.dot(p_t.to(tl.bfloat16), do)
        
        dp_t = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_t = p_t * (dp_t - delta[None, :]) * scale
        
        dk += tl.dot(ds_t.to(tl.bfloat16), q)
        
        q_ptrs += BLOCK_M * stride_qs
        do_ptrs += BLOCK_M * stride_dos
        o_ptrs += BLOCK_M * stride_os
        l_ptrs += BLOCK_M * stride_ls
        
    dK_base = dK + b * stride_dkb + h * stride_dkh
    dV_base = dV + b * stride_dvb + h * stride_dvh
    
    dk_ptrs = dK_base + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV_base + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Standard Triton attention backwards pass avoiding global atomics entirely.
    Workload domains are mapped efficiently to L2 Cache with exclusive Split-Grid regions. 
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    # Dispatch exclusively owned `dQ` calculations mapping locally across KV sets.
    grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, scale, HEAD_DIM=d
    )
    
    # Dispatch exclusively owned `dK` / `dV` calculations mapping locally across Q,dO,O sets.
    grid_dkdv = lambda META: (triton.cdiv(S, META["BLOCK_N"]), B * H)
    bwd_dkdv_kernel[grid_dkdv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, scale, HEAD_DIM=d
    )