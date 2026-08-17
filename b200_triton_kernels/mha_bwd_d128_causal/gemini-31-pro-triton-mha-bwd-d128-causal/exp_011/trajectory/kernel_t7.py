import math
import torch
import triton
import triton.language as tl


# Required Triton infrastructure enabling safe Device-Created Tensor Descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, dQ, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    off_b = pid_b.to(tl.int64)
    off_h = pid_h.to(tl.int64)
    
    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh
    do_offset = off_b * stride_dob + off_h * stride_doh
    dq_offset = off_b * stride_dqb + off_h * stride_dqh
    
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset, shape=[S, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset, shape=[S, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset, shape=[S, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_offset, shape=[S, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + do_offset, shape=[S, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    dq_desc = tl.make_tensor_descriptor(
        dQ + dq_offset, shape=[S, D_HEAD], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, D_HEAD]
    )
    
    start_m = pid_m * BLOCK_M
    q = q_desc.load([start_m, 0])
    do = do_desc.load([start_m, 0])
    out = o_desc.load([start_m, 0])
    
    l_offset = off_b * stride_lb + off_h * stride_lh
    off_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = off_m < S
    off_m_safe = tl.where(mask_m, off_m, 0).to(tl.int64)
    l_ptrs = L + l_offset + off_m_safe * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Compute row-wise D_i efficiently
    d = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
    dq = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)
    
    limit_n_full = start_m
    max_n = min(start_m + BLOCK_M, S)
    
    # --- PHASE 1: Pure pipeline. No masks mapping dynamically handles limits naturally ---
    for start_n in range(0, limit_n_full, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        qk_l = qk - l[:, None]
        p = tl.exp(qk_l)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d[:, None]) * sm_scale
        dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)

    # --- PHASE 2: Causal triangle. Requires mask bindings to prevent looking into the future ---
    for start_n in range(limit_n_full, max_n, BLOCK_N):
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        qk_l = qk - l[:, None]
        
        off_n = start_n + tl.arange(0, BLOCK_N)
        valid_mask = (off_m[:, None] >= off_n[None, :]) & mask_m[:, None] & (off_n[None, :] < S)
        qk_l = tl.where(valid_mask, qk_l, float("-inf"))
        
        p = tl.exp(qk_l)
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - d[:, None]) * sm_scale
        dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
        
    dq_desc.store([start_m, 0], dq.to(dQ.dtype.element_ty))


@triton.jit
def bwd_dkdv_kernel(
    Q, K, V, O, dO, dK, dV, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    off_b = pid_b.to(tl.int64)
    off_h = pid_h.to(tl.int64)
    
    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh
    do_offset = off_b * stride_dob + off_h * stride_doh
    dk_offset = off_b * stride_dkb + off_h * stride_dkh
    dv_offset = off_b * stride_dvb + off_h * stride_dvh
    
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset, shape=[S, D_HEAD], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset, shape=[S, D_HEAD], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset, shape=[S, D_HEAD], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D_HEAD], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_offset, shape=[S, D_HEAD], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + do_offset, shape=[S, D_HEAD], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, D_HEAD], padding_option="zero"
    )
    dk_desc = tl.make_tensor_descriptor(
        dK + dk_offset, shape=[S, D_HEAD], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, D_HEAD]
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + dv_offset, shape=[S, D_HEAD], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, D_HEAD]
    )
    
    start_n = pid_n * BLOCK_N
    k = k_desc.load([start_n, 0])
    v = v_desc.load([start_n, 0])
    
    dk = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D_HEAD], dtype=tl.float32)
    
    off_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = off_n < S
    
    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    limit_causal = min(start_n + BLOCK_N, S)
    l_offset = off_b * stride_lb + off_h * stride_lh
    
    # --- PHASE 1: Transposed Causal evaluation loop --- 
    for start_m in range(start_m_initial, limit_causal, BLOCK_M):
        q = q_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        out = o_desc.load([start_m, 0])
        
        off_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = off_m < S
        off_m_safe = tl.where(mask_m, off_m, 0).to(tl.int64)
        l = tl.load(L + l_offset + off_m_safe * stride_ls, mask=mask_m, other=0.0)
        d = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
        
        qk_t = tl.dot(k, tl.trans(q), out_dtype=tl.float32) * sm_scale
        qk_l_t = qk_t - l[None, :]
        
        valid_mask = (off_m[None, :] >= off_n[:, None]) & mask_m[None, :] & mask_n[:, None]
        qk_l_t = tl.where(valid_mask, qk_l_t, float("-inf"))
        
        p_t = tl.exp(qk_l_t)
        dv += tl.dot(p_t.to(k.dtype), do, out_dtype=tl.float32)
        
        dp_t = tl.dot(v, tl.trans(do), out_dtype=tl.float32)
        ds_t = p_t * (dp_t - d[None, :]) * sm_scale
        dk += tl.dot(ds_t.to(k.dtype), q, out_dtype=tl.float32)

    # --- PHASE 2: Transposed pure pipeline ---
    for start_m in range(limit_causal, S, BLOCK_M):
        q = q_desc.load([start_m, 0])
        do = do_desc.load([start_m, 0])
        out = o_desc.load([start_m, 0])
        
        off_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = off_m < S
        off_m_safe = tl.where(mask_m, off_m, 0).to(tl.int64)
        l = tl.load(L + l_offset + off_m_safe * stride_ls, mask=mask_m, other=0.0)
        d = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
        
        qk_t = tl.dot(k, tl.trans(q), out_dtype=tl.float32) * sm_scale
        qk_l_t = qk_t - l[None, :]
        
        valid_mask = mask_m[None, :] & mask_n[:, None]
        qk_l_t = tl.where(valid_mask, qk_l_t, float("-inf"))
        
        p_t = tl.exp(qk_l_t)
        dv += tl.dot(p_t.to(k.dtype), do, out_dtype=tl.float32)
        
        dp_t = tl.dot(v, tl.trans(do), out_dtype=tl.float32)
        ds_t = p_t * (dp_t - d[None, :]) * sm_scale
        dk += tl.dot(ds_t.to(k.dtype), q, out_dtype=tl.float32)
        
    dk_desc.store([start_n, 0], dk.to(dK.dtype.element_ty))
    dv_desc.store([start_n, 0], dv.to(dV.dtype.element_ty))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes standard destination-passing causal multi-head attention backward.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / math.sqrt(d)

    # Ideal allocation mapped directly for Hopper TMA limits (4 stages = 144 KB SMEM cleanly)
    BLOCK_M_DQ = 128
    BLOCK_N_DQ = 64
    grid_dq = (triton.cdiv(S, BLOCK_M_DQ), B, H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, dQ, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, sm_scale,
        BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ, D_HEAD=d,
        num_warps=8, num_stages=4
    )

    BLOCK_M_DK = 64
    BLOCK_N_DK = 128
    grid_dkdv = (triton.cdiv(S, BLOCK_N_DK), B, H)
    bwd_dkdv_kernel[grid_dkdv](
        Q, K, V, O, dO, dK, dV, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, sm_scale,
        BLOCK_M=BLOCK_M_DK, BLOCK_N=BLOCK_N_DK, D_HEAD=d,
        num_warps=8, num_stages=4
    )