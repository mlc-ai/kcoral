import torch
import triton
import triton.language as tl

@triton.jit
def _bwd_dq(
    Q, K, V, O, DO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, H, scale,
    BLOCK_S_Q: tl.constexpr,
    BLOCK_S_KV: tl.constexpr,
    d: tl.constexpr
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    offs_q = pid_q * BLOCK_S_Q
    
    # Q, dO, O descriptors for the fixed Q block
    desc_Q = tl.make_tensor_descriptor(
        Q + pid_b * stride_qb + pid_h * stride_qh,
        shape=[S, d],
        strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_S_Q, d],
        padding_option="zero"
    )
    desc_DO = tl.make_tensor_descriptor(
        DO + pid_b * stride_dob + pid_h * stride_doh,
        shape=[S, d],
        strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_S_Q, d],
        padding_option="zero"
    )
    desc_O = tl.make_tensor_descriptor(
        O + pid_b * stride_ob + pid_h * stride_oh,
        shape=[S, d],
        strides=[stride_os, stride_od],
        block_shape=[BLOCK_S_Q, d],
        padding_option="zero"
    )
    
    q = tl.load(desc_Q, [offs_q, 0])
    do = tl.load(desc_DO, [offs_q, 0])
    o = tl.load(desc_O, [offs_q, 0])
    
    # Compute on-chip Delta (row-wise dot product of O and dO)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq_acc = tl.zeros([BLOCK_S_Q, d], dtype=tl.float32)
    
    offs_q_arr = offs_q + tl.arange(0, BLOCK_S_Q)
    mask_q = offs_q_arr < S
    l = tl.load(L + pid_b * stride_lb + pid_h * stride_lh + offs_q_arr * stride_ls, mask=mask_q, other=0.0)
    
    # K, V descriptors to loop over
    desc_K = tl.make_tensor_descriptor(
        K + pid_b * stride_kb + pid_h * stride_kh,
        shape=[S, d],
        strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_S_KV, d],
        padding_option="zero"
    )
    desc_V = tl.make_tensor_descriptor(
        V + pid_b * stride_vb + pid_h * stride_vh,
        shape=[S, d],
        strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_S_KV, d],
        padding_option="zero"
    )
    
    num_kv_blocks = tl.cdiv(S, BLOCK_S_KV)
    for kv_idx in range(num_kv_blocks):
        offs_k = kv_idx * BLOCK_S_KV
        
        k = tl.load(desc_K, [offs_k, 0])
        v = tl.load(desc_V, [offs_k, 0])
        
        # S_ij = Q @ K.T * scale
        qk = tl.zeros([BLOCK_S_Q, BLOCK_S_KV], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        qk = qk * scale
        
        # P = exp(S_ij - L)
        p = tl.exp(qk - l[:, None])
        
        # Apply causal mask (none) and sequence boundary mask
        offs_k_arr = offs_k + tl.arange(0, BLOCK_S_KV)
        mask_k = offs_k_arr < S
        p = tl.where(mask_q[:, None] & mask_k[None, :], p, 0.0)
        
        # dP = dO @ V.T
        dp = tl.zeros([BLOCK_S_Q, BLOCK_S_KV], dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), dp)
        
        # dS = P * (dP - Delta) * scale
        ds = p * (dp - delta[:, None]) * scale
        
        # dQ_i += dS @ K_j
        dq_acc = tl.dot(ds.to(k.dtype), k, dq_acc)
        
    desc_dQ = tl.make_tensor_descriptor(
        dQ + pid_b * stride_dqb + pid_h * stride_dqh,
        shape=[S, d],
        strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_S_Q, d]
    )
    tl.store(desc_dQ, [offs_q, 0], dq_acc.to(dQ.dtype.element_ty))


@triton.jit
def _bwd_dkdv(
    Q, K, V, O, DO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, H, scale,
    BLOCK_S_Q: tl.constexpr,
    BLOCK_S_KV: tl.constexpr,
    d: tl.constexpr
):
    pid_kv = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    offs_k = pid_kv * BLOCK_S_KV
    
    # K, V descriptors for the fixed KV block
    desc_K = tl.make_tensor_descriptor(
        K + pid_b * stride_kb + pid_h * stride_kh,
        shape=[S, d],
        strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_S_KV, d],
        padding_option="zero"
    )
    desc_V = tl.make_tensor_descriptor(
        V + pid_b * stride_vb + pid_h * stride_vh,
        shape=[S, d],
        strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_S_KV, d],
        padding_option="zero"
    )
    
    k = tl.load(desc_K, [offs_k, 0])
    v = tl.load(desc_V, [offs_k, 0])
    
    dk_acc = tl.zeros([BLOCK_S_KV, d], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_S_KV, d], dtype=tl.float32)
    
    # Q, dO, O descriptors to loop over
    desc_Q = tl.make_tensor_descriptor(
        Q + pid_b * stride_qb + pid_h * stride_qh,
        shape=[S, d],
        strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_S_Q, d],
        padding_option="zero"
    )
    desc_DO = tl.make_tensor_descriptor(
        DO + pid_b * stride_dob + pid_h * stride_doh,
        shape=[S, d],
        strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_S_Q, d],
        padding_option="zero"
    )
    desc_O = tl.make_tensor_descriptor(
        O + pid_b * stride_ob + pid_h * stride_oh,
        shape=[S, d],
        strides=[stride_os, stride_od],
        block_shape=[BLOCK_S_Q, d],
        padding_option="zero"
    )
    
    offs_k_arr = offs_k + tl.arange(0, BLOCK_S_KV)
    mask_k = offs_k_arr < S
    
    num_q_blocks = tl.cdiv(S, BLOCK_S_Q)
    for q_idx in range(num_q_blocks):
        offs_q = q_idx * BLOCK_S_Q
        
        q = tl.load(desc_Q, [offs_q, 0])
        do = tl.load(desc_DO, [offs_q, 0])
        o = tl.load(desc_O, [offs_q, 0])
        
        # Compute on-chip Delta (row-wise dot product of O and dO)
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        offs_q_arr = offs_q + tl.arange(0, BLOCK_S_Q)
        mask_q = offs_q_arr < S
        l = tl.load(L + pid_b * stride_lb + pid_h * stride_lh + offs_q_arr * stride_ls, mask=mask_q, other=0.0)
        
        # S_ij = Q @ K.T * scale
        qk = tl.zeros([BLOCK_S_Q, BLOCK_S_KV], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        qk = qk * scale
        
        # P = exp(S_ij - L)
        p = tl.exp(qk - l[:, None])
        p = tl.where(mask_q[:, None] & mask_k[None, :], p, 0.0)
        
        # dP = dO @ V.T
        dp = tl.zeros([BLOCK_S_Q, BLOCK_S_KV], dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), dp)
        
        # dS = P * (dP - Delta) * scale
        ds = p * (dp - delta[:, None]) * scale
        
        ds_dtype = ds.to(k.dtype)
        p_dtype = p.to(k.dtype)
        
        # dV_j += P^T @ dO_i
        dv_acc = tl.dot(tl.trans(p_dtype), do, dv_acc)
        # dK_j += dS^T @ Q_i
        dk_acc = tl.dot(tl.trans(ds_dtype), q, dk_acc)
        
    desc_dK = tl.make_tensor_descriptor(
        dK + pid_b * stride_dkb + pid_h * stride_dkh,
        shape=[S, d],
        strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_S_KV, d]
    )
    desc_dV = tl.make_tensor_descriptor(
        dV + pid_b * stride_dvb + pid_h * stride_dvh,
        shape=[S, d],
        strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_S_KV, d]
    )
    tl.store(desc_dK, [offs_k, 0], dk_acc.to(dK.dtype.element_ty))
    tl.store(desc_dV, [offs_k, 0], dv_acc.to(dV.dtype.element_ty))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the multi-head attention backward pass using TMA device-created descriptors.
    """
    with torch.cuda.device(Q.device):
        # Configure allocator for device-side tensor-descriptors
        def alloc_fn(size: int, alignment: int, stream):
            return torch.empty(size, device="cuda", dtype=torch.int8)
        triton.set_allocator(alloc_fn)
        
        B, H, S, d_val = Q.shape
        scale = 1.0 / (d_val ** 0.5)
        
        # Extract strides
        stride_qb, stride_qh, stride_qs, stride_qd = Q.stride()
        stride_kb, stride_kh, stride_ks, stride_kd = K.stride()
        stride_vb, stride_vh, stride_vs, stride_vd = V.stride()
        stride_ob, stride_oh, stride_os, stride_od = O.stride()
        stride_dob, stride_doh, stride_dos, stride_dod = dO.stride()
        stride_lb, stride_lh, stride_ls = L.stride()
        stride_dqb, stride_dqh, stride_dqs, stride_dqd = dQ.stride()
        stride_dkb, stride_dkh, stride_dks, stride_dkd = dK.stride()
        stride_dvb, stride_dvh, stride_dvs, stride_dvd = dV.stride()
        
        # Selected tile shapes
        BLOCK_S_Q = 64
        BLOCK_S_KV = 64
        
        # Launch dQ kernel
        grid_dq = (triton.cdiv(S, BLOCK_S_Q), B * H)
        _bwd_dq[grid_dq](
            Q, K, V, O, dO, L, dQ,
            stride_qb, stride_qh, stride_qs, stride_qd,
            stride_kb, stride_kh, stride_ks, stride_kd,
            stride_vb, stride_vh, stride_vs, stride_vd,
            stride_ob, stride_oh, stride_os, stride_od,
            stride_dob, stride_doh, stride_dos, stride_dod,
            stride_lb, stride_lh, stride_ls,
            stride_dqb, stride_dqh, stride_dqs, stride_dqd,
            S, H, scale,
            BLOCK_S_Q=BLOCK_S_Q, BLOCK_S_KV=BLOCK_S_KV, d=d_val,
            num_warps=4, num_stages=3
        )
        
        # Launch dK and dV kernel
        grid_dkdv = (triton.cdiv(S, BLOCK_S_KV), B * H)
        _bwd_dkdv[grid_dkdv](
            Q, K, V, O, dO, L, dK, dV,
            stride_qb, stride_qh, stride_qs, stride_qd,
            stride_kb, stride_kh, stride_ks, stride_kd,
            stride_vb, stride_vh, stride_vs, stride_vd,
            stride_ob, stride_oh, stride_os, stride_od,
            stride_dob, stride_doh, stride_dos, stride_dod,
            stride_lb, stride_lh, stride_ls,
            stride_dkb, stride_dkh, stride_dks, stride_dkd,
            stride_dvb, stride_dvh, stride_dvs, stride_dvd,
            S, H, scale,
            BLOCK_S_Q=BLOCK_S_Q, BLOCK_S_KV=BLOCK_S_KV, d=d_val,
            num_warps=4, num_stages=3
        )
        
        return dQ, dK, dV