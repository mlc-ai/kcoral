import torch
import triton
import triton.language as tl

# Configure Triton's descriptor allocator to enable zero-overhead TMA setup directly on the device 
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

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
    S, scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_base = Q_ptr + b * stride_qb + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    do_base = dO_ptr + b * stride_dob + h * stride_doh
    dq_base = dQ_ptr + b * stride_dqb + h * stride_dqh

    q_desc = tl.make_tensor_descriptor(q_base, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(k_base, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_base, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_base, shape=[S, d], strides=[stride_os, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_base, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    dq_desc = tl.make_tensor_descriptor(dq_base, shape=[S, d], strides=[stride_dqs, 1], block_shape=[BLOCK_M, d], padding_option="zero")

    q = q_desc.load([pid_m * BLOCK_M, 0])
    o = o_desc.load([pid_m * BLOCK_M, 0]).to(tl.float32)
    do = do_desc.load([pid_m * BLOCK_M, 0])
    
    l_base = L_ptr + b * stride_lb + h * stride_lh
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    L = tl.load(l_base + offs_m * stride_ls, mask=mask_m, other=0.0)
    
    D = tl.sum(o * do.to(tl.float32), axis=1)

    acc_dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    max_j = tl.minimum(pid_m * BLOCK_M + BLOCK_M, S)
    num_blocks = (max_j + BLOCK_N - 1) // BLOCK_N

    for n_idx in tl.range(0, num_blocks, num_stages=2):
        k = k_desc.load([n_idx * BLOCK_N, 0])
        v = v_desc.load([n_idx * BLOCK_N, 0])
        
        s_ij = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        current_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        causal_mask = current_n[None, :] <= offs_m[:, None]
        mask_n = current_n < S
        valid_mask = causal_mask & mask_n[None, :] & mask_m[:, None]
        
        p_ij = tl.exp(s_ij - L[:, None])
        p_ij = tl.where(valid_mask, p_ij, 0.0)
        
        dp_ij = tl.dot(do, v.T, out_dtype=tl.float32)
        
        ds_ij = p_ij * (dp_ij - D[:, None])
        ds_ij_scaled = ds_ij * scale
        
        acc_dq = tl.dot(ds_ij_scaled.to(tl.bfloat16), k, acc_dq)
        
    dq_desc.store([pid_m * BLOCK_M, 0], acc_dq.to(tl.bfloat16))


@triton.jit
def bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    
    q_base = Q_ptr + b * stride_qb + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    o_base = O_ptr + b * stride_ob + h * stride_oh
    do_base = dO_ptr + b * stride_dob + h * stride_doh
    dk_base = dK_ptr + b * stride_dkb + h * stride_dkh
    dv_base = dV_ptr + b * stride_dvb + h * stride_dvh

    q_desc = tl.make_tensor_descriptor(q_base, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    k_desc = tl.make_tensor_descriptor(k_base, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_base, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_base, shape=[S, d], strides=[stride_os, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_base, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    dk_desc = tl.make_tensor_descriptor(dk_base, shape=[S, d], strides=[stride_dks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    dv_desc = tl.make_tensor_descriptor(dv_base, shape=[S, d], strides=[stride_dvs, 1], block_shape=[BLOCK_N, d], padding_option="zero")

    k = k_desc.load([pid_n * BLOCK_N, 0])
    v = v_desc.load([pid_n * BLOCK_N, 0])
    
    acc_dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    start_m_idx = (pid_n * BLOCK_N) // BLOCK_M
    num_m_blocks = (S + BLOCK_M - 1) // BLOCK_M
    
    l_base = L_ptr + b * stride_lb + h * stride_lh
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S

    for m_idx in tl.range(start_m_idx, num_m_blocks, num_stages=2):
        q = q_desc.load([m_idx * BLOCK_M, 0])
        o = o_desc.load([m_idx * BLOCK_M, 0]).to(tl.float32)
        do = do_desc.load([m_idx * BLOCK_M, 0])
        
        current_m = m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = current_m < S
        L = tl.load(l_base + current_m * stride_ls, mask=mask_m, other=0.0)
        
        D = tl.sum(o * do.to(tl.float32), axis=1)
        
        # Native RHS-layout transposition bypasses LHS MMA layout anomalies 
        s_ji = tl.dot(k, q.T, out_dtype=tl.float32) * scale
        
        causal_mask = offs_n[:, None] <= current_m[None, :]
        valid_mask = causal_mask & mask_n[:, None] & mask_m[None, :]
        
        p_ji = tl.exp(s_ji - L[None, :])
        p_ji = tl.where(valid_mask, p_ji, 0.0)
        
        acc_dv = tl.dot(p_ji.to(tl.bfloat16), do, acc_dv)
        
        dp_ji = tl.dot(v, do.T, out_dtype=tl.float32)
        
        ds_ji = p_ji * (dp_ji - D[None, :])
        ds_ji_scaled = ds_ji * scale
        
        acc_dk = tl.dot(ds_ji_scaled.to(tl.bfloat16), q, acc_dk)
        
    dk_desc.store([pid_n * BLOCK_N, 0], acc_dk.to(tl.bfloat16))
    dv_desc.store([pid_n * BLOCK_N, 0], acc_dv.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Destinaton-passing Triton causal attention backward point.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)

    if L.dim() == 4:
        L = L.squeeze(-1)

    # dQ Kernel leverages 128x128 footprint
    grid_dq = (triton.cdiv(S, 128), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, scale, H,
        BLOCK_M=128, BLOCK_N=128, d=128,
        num_warps=8, num_stages=2
    )

    # dK/dV Kernel pivots shapes safely restricting shared memory caching limits to <228 KiB
    grid_dk_dv = (triton.cdiv(S, 128), B * H)
    bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, scale, H,
        BLOCK_M=64, BLOCK_N=128, d=128,
        num_warps=4, num_stages=2
    )