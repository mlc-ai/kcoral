import torch
import triton
import triton.language as tl

@triton.jit
def bwd_q_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, scale, d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    m_start = pid_m * BLOCK_SIZE
    m_offs = m_start + tl.arange(0, BLOCK_SIZE)
    d_offs = tl.arange(0, d)
    
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    do_base = dO + pid_b * stride_dob + pid_h * stride_doh
    l_base = L + pid_b * stride_lb + pid_h * stride_lh
    dq_base = dQ + pid_b * stride_dqb + pid_h * stride_dqh
    
    mask_m = m_offs < S
    
    q_ptrs = q_base + m_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    o_ptrs = o_base + m_offs[:, None] * stride_os + d_offs[None, :] * stride_od
    do_ptrs = do_base + m_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod
    l_ptrs = l_base + m_offs * stride_ls
    
    q_i = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o_i = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    delta_i = tl.sum(o_i.to(tl.float32) * do_i.to(tl.float32), axis=1)
    
    acc_dq = tl.zeros((BLOCK_SIZE, d), dtype=tl.float32)
    
    for j in range(0, pid_m + 1):
        n_start = j * BLOCK_SIZE
        n_offs = n_start + tl.arange(0, BLOCK_SIZE)
        mask_n = n_offs < S
        
        k_ptrs = k_base + n_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
        v_ptrs = v_base + n_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd
        
        k_j = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_j = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        dp = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)
        scores = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * scale
        
        valid_mask = (m_offs[:, None] >= n_offs[None, :]) & mask_m[:, None] & mask_n[None, :]
        scores = tl.where(valid_mask, scores, -float("inf"))
        
        # log2(e) = 1.4426950408889634
        p = tl.math.exp2((scores - l_i[:, None]) * 1.4426950408889634)
        p = tl.where(valid_mask, p, 0.0)
        
        ds = p * (dp - delta_i[:, None]) * scale
        ds = tl.where(valid_mask, ds, 0.0)
        
        acc_dq = tl.dot(ds.to(tl.bfloat16), k_j, acc=acc_dq, out_dtype=tl.float32)
        
    dq_ptrs = dq_base + m_offs[:, None] * stride_dqs + d_offs[None, :] * stride_dqd
    tl.store(dq_ptrs, acc_dq.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def bwd_kv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, num_m_blocks, scale, d: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    n_start = pid_n * BLOCK_SIZE
    n_offs = n_start + tl.arange(0, BLOCK_SIZE)
    d_offs = tl.arange(0, d)
    
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    do_base = dO + pid_b * stride_dob + pid_h * stride_doh
    l_base = L + pid_b * stride_lb + pid_h * stride_lh
    dk_base = dK + pid_b * stride_dkb + pid_h * stride_dkh
    dv_base = dV + pid_b * stride_dvb + pid_h * stride_dvh
    
    mask_n = n_offs < S
    
    k_ptrs = k_base + n_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
    v_ptrs = v_base + n_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd
    
    k_j = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v_j = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    acc_dk = tl.zeros((BLOCK_SIZE, d), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE, d), dtype=tl.float32)
    
    for i in range(pid_n, num_m_blocks):
        m_start = i * BLOCK_SIZE
        m_offs = m_start + tl.arange(0, BLOCK_SIZE)
        mask_m = m_offs < S
        
        q_ptrs = q_base + m_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
        o_ptrs = o_base + m_offs[:, None] * stride_os + d_offs[None, :] * stride_od
        do_ptrs = do_base + m_offs[:, None] * stride_dos + d_offs[None, :] * stride_dod
        l_ptrs = l_base + m_offs * stride_ls
        
        q_i = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o_i = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do_i = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta_i = tl.sum(o_i.to(tl.float32) * do_i.to(tl.float32), axis=1)
        
        dp = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)
        scores = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * scale
        
        valid_mask = (m_offs[:, None] >= n_offs[None, :]) & mask_m[:, None] & mask_n[None, :]
        scores = tl.where(valid_mask, scores, -float("inf"))
        
        p = tl.math.exp2((scores - l_i[:, None]) * 1.4426950408889634)
        p = tl.where(valid_mask, p, 0.0)
        
        ds = p * (dp - delta_i[:, None]) * scale
        ds = tl.where(valid_mask, ds, 0.0)
        
        acc_dk = tl.dot(tl.trans(ds.to(tl.bfloat16)), q_i, acc=acc_dk, out_dtype=tl.float32)
        acc_dv = tl.dot(tl.trans(p.to(tl.bfloat16)), do_i, acc=acc_dv, out_dtype=tl.float32)
        
    dk_ptrs = dk_base + n_offs[:, None] * stride_dks + d_offs[None, :] * stride_dkd
    dv_ptrs = dv_base + n_offs[:, None] * stride_dvs + d_offs[None, :] * stride_dvd
    
    tl.store(dk_ptrs, acc_dk.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, acc_dv.to(tl.bfloat16), mask=mask_n[:, None])

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    # Safely extract strides, handling cases where L might be passed as a 3D or 4D tensor
    L_strides = L.stride()
    stride_lb, stride_lh, stride_ls = L_strides[0], L_strides[1], L_strides[2]
    
    B, H, S, d = Q.shape
    
    BLOCK_SIZE = 64
    num_m_blocks = triton.cdiv(S, BLOCK_SIZE)
    scale = 1.0 / (d ** 0.5)
    
    grid = (num_m_blocks, H, B)
    
    bwd_q_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        stride_lb, stride_lh, stride_ls,
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, scale, d,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2
    )
    
    bwd_kv_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        stride_lb, stride_lh, stride_ls,
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, num_m_blocks, scale, d,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2
    )