import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_R": 128, "BLOCK_C": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_R": 128, "BLOCK_C": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_R": 64, "BLOCK_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_R": 64, "BLOCK_C": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"]
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
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, d: tl.constexpr
):
    # i_id changes fastest to keep programs of the same head scheduled concurrently.
    # This maximizes L2 cache sharing for K_j and V_j during the inner loop.
    i_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    H = 48
    b_id = bh_id // H
    h_id = bh_id % H
    
    offs_d = tl.arange(0, d)
    offs_r = i_id * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = offs_r < S
    
    q_ptr = Q + b_id * stride_qb + h_id * stride_qh + offs_r[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptr = dO + b_id * stride_dob + h_id * stride_doh + offs_r[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptr = O + b_id * stride_ob + h_id * stride_oh + offs_r[:, None] * stride_os + offs_d[None, :] * stride_od
    l_ptr = L + b_id * stride_lb + h_id * stride_lh + offs_r * stride_ls
    
    # Outer loop variables (loaded once per block)
    q_i = tl.load(q_ptr, mask=mask_r[:, None], other=0.0)
    do_i = tl.load(do_ptr, mask=mask_r[:, None], other=0.0)
    o_i = tl.load(o_ptr, mask=mask_r[:, None], other=0.0)
    l_i = tl.load(l_ptr, mask=mask_r, other=0.0)
    
    # Precompute D_i for this query block
    d_i = tl.sum(o_i.to(tl.float32) * do_i.to(tl.float32), axis=1)
    
    dq_i = tl.zeros([BLOCK_R, d], dtype=tl.float32)
    num_steps = tl.cdiv(S, BLOCK_C)
    
    for j in range(0, num_steps):
        offs_c = j * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = offs_c < S
        
        k_ptr = K + b_id * stride_kb + h_id * stride_kh + offs_c[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptr = V + b_id * stride_vb + h_id * stride_vh + offs_c[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k_j = tl.load(k_ptr, mask=mask_c[:, None], other=0.0)
        v_j = tl.load(v_ptr, mask=mask_c[:, None], other=0.0)
        
        # All GEMMs are configured to leverage optimal RS-GEMMs and SS-GEMMs natively
        s_ij = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * scale
        
        p_ij = tl.exp(s_ij - l_i[:, None])
        p_ij = tl.where((mask_r[:, None]) & (mask_c[None, :]), p_ij, 0.0)
        
        dp_ij = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)
        
        ds_ij = p_ij * (dp_ij - d_i[:, None]) * scale
        ds_ij_bf16 = ds_ij.to(q_i.dtype)
        
        dq_i = tl.dot(ds_ij_bf16, k_j, acc=dq_i, out_dtype=tl.float32)
        
    dq_out_ptr = dQ + b_id * stride_dqb + h_id * stride_dqh + offs_r[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_out_ptr, dq_i.to(q_i.dtype), mask=mask_r[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 128, "BLOCK_R": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_C": 64, "BLOCK_R": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 64, "BLOCK_R": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"]
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
    S, scale,
    BLOCK_C: tl.constexpr, BLOCK_R: tl.constexpr, d: tl.constexpr
):
    # j_id changes fastest to allow L2 sharing of Q_i, dO_i, O_i during the inner loop
    j_id = tl.program_id(0)
    bh_id = tl.program_id(1)
    
    H = 48
    b_id = bh_id // H
    h_id = bh_id % H
    
    offs_d = tl.arange(0, d)
    offs_c = j_id * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < S
    
    k_ptr = K + b_id * stride_kb + h_id * stride_kh + offs_c[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptr = V + b_id * stride_vb + h_id * stride_vh + offs_c[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k_j = tl.load(k_ptr, mask=mask_c[:, None], other=0.0)
    v_j = tl.load(v_ptr, mask=mask_c[:, None], other=0.0)
    
    dk_j = tl.zeros([BLOCK_C, d], dtype=tl.float32)
    dv_j = tl.zeros([BLOCK_C, d], dtype=tl.float32)
    
    num_steps = tl.cdiv(S, BLOCK_R)
    
    for i in range(0, num_steps):
        offs_r = i * BLOCK_R + tl.arange(0, BLOCK_R)
        mask_r = offs_r < S
        
        q_ptr = Q + b_id * stride_qb + h_id * stride_qh + offs_r[:, None] * stride_qs + offs_d[None, :] * stride_qd
        do_ptr = dO + b_id * stride_dob + h_id * stride_doh + offs_r[:, None] * stride_dos + offs_d[None, :] * stride_dod
        o_ptr = O + b_id * stride_ob + h_id * stride_oh + offs_r[:, None] * stride_os + offs_d[None, :] * stride_od
        l_ptr = L + b_id * stride_lb + h_id * stride_lh + offs_r * stride_ls
        
        q_i = tl.load(q_ptr, mask=mask_r[:, None], other=0.0)
        do_i = tl.load(do_ptr, mask=mask_r[:, None], other=0.0)
        o_i = tl.load(o_ptr, mask=mask_r[:, None], other=0.0)
        l_i = tl.load(l_ptr, mask=mask_r, other=0.0)
        
        d_i = tl.sum(o_i.to(tl.float32) * do_i.to(tl.float32), axis=1)
        
        # Dual formulation to keep intermediate states (S, P, dP, dS) in transposed form (BLOCK_C, BLOCK_R)
        # This guarantees register operands are never transposed in the GEMMs, unlocking RS-GEMM speeds on Hopper.
        
        s_trans = tl.dot(k_j, tl.trans(q_i), out_dtype=tl.float32) * scale
        
        p_trans = tl.exp(s_trans - l_i[None, :])
        p_trans = tl.where((mask_c[:, None]) & (mask_r[None, :]), p_trans, 0.0)
        
        dp_trans = tl.dot(v_j, tl.trans(do_i), out_dtype=tl.float32)
        
        ds_trans = p_trans * (dp_trans - d_i[None, :]) * scale
        
        p_trans_bf16 = p_trans.to(q_i.dtype)
        ds_trans_bf16 = ds_trans.to(q_i.dtype)
        
        # First operand in registers (non-transposed), second in SMEM -> RS-GEMM
        dv_j = tl.dot(p_trans_bf16, do_i, acc=dv_j, out_dtype=tl.float32)
        dk_j = tl.dot(ds_trans_bf16, q_i, acc=dk_j, out_dtype=tl.float32)
        
    dk_out_ptr = dK + b_id * stride_dkb + h_id * stride_dkh + offs_c[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_out_ptr = dV + b_id * stride_dvb + h_id * stride_dvh + offs_c[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_out_ptr, dk_j.to(k_j.dtype), mask=mask_c[:, None])
    tl.store(dv_out_ptr, dv_j.to(v_j.dtype), mask=mask_c[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes standard SDPA backward for MHA, minimizing HBM trips and bypassing
    hardware constraints via transposed register layouts and grouped grid execution.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    stride_lb = L.stride(0)
    stride_lh = L.stride(1)
    stride_ls = L.stride(2)
    
    # 1. Dispatch dQ kernel
    grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_R"]), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        stride_lb, stride_lh, stride_ls,
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, scale,
        d=128
    )
    
    # 2. Dispatch dK, dV kernel
    grid_dkdv = lambda META: (triton.cdiv(S, META["BLOCK_C"]), B * H)
    bwd_dk_dv_kernel[grid_dkdv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        stride_lb, stride_lh, stride_ls,
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, scale,
        d=128
    )