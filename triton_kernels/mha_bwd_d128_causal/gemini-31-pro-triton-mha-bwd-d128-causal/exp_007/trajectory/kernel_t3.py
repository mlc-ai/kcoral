import torch
import triton
import triton.language as tl


@triton.jit
def zero_tensors(
    ptr1, ptr2, total_elements, d: tl.constexpr,
    stride_b1, stride_h1, stride_s1, stride_d1,
    stride_b2, stride_h2, stride_s2, stride_d2,
    H, S, BLOCK: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total_elements
    
    b = idx // (H * S)
    h = (idx // S) % H
    s = idx % S
    
    offs_d = tl.arange(0, d)
    
    p1 = ptr1 + b[:, None] * stride_b1 + h[:, None] * stride_h1 + s[:, None] * stride_s1 + offs_d[None, :] * stride_d1
    p2 = ptr2 + b[:, None] * stride_b2 + h[:, None] * stride_h2 + s[:, None] * stride_s2 + offs_d[None, :] * stride_d2
    
    zeros = tl.zeros([BLOCK, d], dtype=tl.bfloat16)
    tl.store(p1, zeros, mask=mask[:, None])
    tl.store(p2, zeros, mask=mask[:, None])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_kernel(
    Q, K, V, O, dO, L, dQ, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
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
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, d)
    mask_m = offs_m < S
    
    # Adjust base pointers for the current Batch and Head
    Q += b * stride_qb + h * stride_qh
    O += b * stride_ob + h * stride_oh
    dO += b * stride_dob + h * stride_doh
    L += b * stride_lb + h * stride_lh
    dQ += b * stride_dqb + h * stride_dqh
    
    K += b * stride_kb + h * stride_kh
    V += b * stride_vb + h * stride_vh
    dK += b * stride_dkb + h * stride_dkh
    dV += b * stride_dvb + h * stride_dvh
    
    q_ptrs = Q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    o_ptrs = O + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    do_ptrs = dO + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
    l_ptrs = L + offs_m * stride_ls
    
    # Load inputs corresponding to Query block completely once
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    lse = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    k_ptrs = K + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    dk_ptrs = dK + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    j_boundary = (i_start // BLOCK_N) * BLOCK_N
    
    # 1. Unmasked blocks iteration (fully valid pairs naturally governed by causality)
    for j_start in range(0, j_boundary, BLOCK_N):
        k_curr = tl.load(k_ptrs)
        v_curr = tl.load(v_ptrs)
        
        acc_s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s_ij = tl.dot(q, tl.trans(k_curr), acc_s) * scale
        
        # We must mask padded queries M explicitly to prevent generating positive infinity dot products
        s_ij = tl.where(mask_m[:, None], s_ij, float("-inf"))
        p_ij = tl.exp(s_ij - lse[:, None])
        
        dv_acc = tl.zeros([BLOCK_N, d], dtype=tl.float32)
        dv_curr = tl.dot(tl.trans(p_ij.to(tl.bfloat16)), do, dv_acc)
        tl.atomic_add(dv_ptrs, dv_curr.to(tl.bfloat16), sem="relaxed")
        
        dp_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp_ij = tl.dot(do, tl.trans(v_curr), dp_acc)
        ds_ij = p_ij * (dp_ij - delta[:, None])
        ds_ij_scaled = (ds_ij * scale).to(tl.bfloat16)
        
        dq = tl.dot(ds_ij_scaled, k_curr, dq)
        
        dk_acc = tl.zeros([BLOCK_N, d], dtype=tl.float32)
        dk_curr = tl.dot(tl.trans(ds_ij_scaled), q, dk_acc)
        tl.atomic_add(dk_ptrs, dk_curr.to(tl.bfloat16), sem="relaxed")
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        dk_ptrs += BLOCK_N * stride_dks
        dv_ptrs += BLOCK_N * stride_dvs

    # 2. Masked boundary block
    for j_start in range(j_boundary, min(i_start + BLOCK_M, S), BLOCK_N):
        offs_n_curr = j_start + offs_n
        mask_n = offs_n_curr < S
        
        k_curr = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_curr = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        acc_s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s_ij = tl.dot(q, tl.trans(k_curr), acc_s) * scale
        
        causal_mask = offs_n_curr[None, :] <= offs_m[:, None]
        valid_mask = causal_mask & mask_m[:, None] & mask_n[None, :]
        
        s_ij = tl.where(valid_mask, s_ij, float("-inf"))
        p_ij = tl.exp(s_ij - lse[:, None])
        p_ij = tl.where(valid_mask, p_ij, 0.0)
        
        dv_acc = tl.zeros([BLOCK_N, d], dtype=tl.float32)
        dv_curr = tl.dot(tl.trans(p_ij.to(tl.bfloat16)), do, dv_acc)
        tl.atomic_add(dv_ptrs, dv_curr.to(tl.bfloat16), mask=mask_n[:, None], sem="relaxed")
        
        dp_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp_ij = tl.dot(do, tl.trans(v_curr), dp_acc)
        ds_ij = p_ij * (dp_ij - delta[:, None])
        ds_ij_scaled = (ds_ij * scale).to(tl.bfloat16)
        
        dq = tl.dot(ds_ij_scaled, k_curr, dq)
        
        dk_acc = tl.zeros([BLOCK_N, d], dtype=tl.float32)
        dk_curr = tl.dot(tl.trans(ds_ij_scaled), q, dk_acc)
        tl.atomic_add(dk_ptrs, dk_curr.to(tl.bfloat16), mask=mask_n[:, None], sem="relaxed")
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        dk_ptrs += BLOCK_N * stride_dks
        dv_ptrs += BLOCK_N * stride_dvs

    dq_ptrs = dQ + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(tl.bfloat16), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Compute causal SDPA backward and write exactly to the provided dQ, dK, dV outputs.
    Optimizes overlapping Math + Loads into 1 atomic-pass minimizing recalculations drastically. 
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)

    if L.dim() == 4:
        L = L.squeeze(-1)

    # Initialize Atomic Destination buffers across their valid domains natively and stride-independently
    total_bhs = B * H * S
    grid_zero = (triton.cdiv(total_bhs, 256),)
    zero_tensors[grid_zero](
        dK, dV, total_bhs, d,
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        H, S, BLOCK=256
    )

    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B * H
    )
    bwd_kernel[grid](
        Q, K, V, O, dO, L, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, scale,
        d=d
    )