import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


@triton.jit
def bwd_q_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs,
    S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    dq_ptr = dQ + pid_b * stride_dqb + pid_h * stride_dqh
    
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, d], strides=[stride_os, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    dq_desc = tl.make_tensor_descriptor(dq_ptr, shape=[S, d], strides=[stride_dqs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    
    m_coord = pid_m * BLOCK_M
    q_i = q_desc.load([m_coord, 0])
    o_i = o_desc.load([m_coord, 0])
    do_i = do_desc.load([m_coord, 0])
    
    l_ptr = L + pid_b * stride_lb + pid_h * stride_lh
    m_offs = m_coord + tl.arange(0, BLOCK_M)
    l_i = tl.load(l_ptr + m_offs * stride_ls, mask=m_offs < S, other=0.0)
    
    delta_i = tl.sum(o_i.to(tl.float32) * do_i.to(tl.float32), axis=1)
    
    acc_dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    n_blocks = (pid_m * BLOCK_M + BLOCK_M - 1) // BLOCK_N + 1
    
    for n in tl.range(0, n_blocks, num_stages=3):
        n_coord = n * BLOCK_N
        k_j = k_desc.load([n_coord, 0])
        v_j = v_desc.load([n_coord, 0])
        
        scores = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * scale
        dp = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)
        
        n_offs = n_coord + tl.arange(0, BLOCK_N)
        valid_mask = (m_offs[:, None] >= n_offs[None, :]) & (m_offs[:, None] < S) & (n_offs[None, :] < S)
        scores = tl.where(valid_mask, scores, -float("inf"))
        
        p = tl.exp(scores - l_i[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        ds = p * (dp - delta_i[:, None]) * scale
        ds = tl.where(valid_mask, ds, 0.0)
        
        acc_dq = tl.dot(ds.to(tl.bfloat16), k_j, acc=acc_dq, out_dtype=tl.float32)
        
    dq_desc.store([m_coord, 0], acc_dq.to(tl.bfloat16))


@triton.jit
def bwd_kv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks,
    stride_dvb, stride_dvh, stride_dvs,
    S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    dk_ptr = dK + pid_b * stride_dkb + pid_h * stride_dkh
    dv_ptr = dV + pid_b * stride_dvb + pid_h * stride_dvh
    
    q_desc = tl.make_tensor_descriptor(q_ptr, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(o_ptr, shape=[S, d], strides=[stride_os, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(do_ptr, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    
    k_desc = tl.make_tensor_descriptor(k_ptr, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(v_ptr, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    dk_desc = tl.make_tensor_descriptor(dk_ptr, shape=[S, d], strides=[stride_dks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    dv_desc = tl.make_tensor_descriptor(dv_ptr, shape=[S, d], strides=[stride_dvs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    
    n_coord = pid_n * BLOCK_N
    k_j = k_desc.load([n_coord, 0])
    v_j = v_desc.load([n_coord, 0])
    
    n_offs = n_coord + tl.arange(0, BLOCK_N)
    l_ptr = L + pid_b * stride_lb + pid_h * stride_lh
    
    acc_dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    m_start_idx = n_coord // BLOCK_M
    num_m_blocks = (S + BLOCK_M - 1) // BLOCK_M
    
    for m in tl.range(m_start_idx, num_m_blocks, num_stages=3):
        m_coord = m * BLOCK_M
        q_i = q_desc.load([m_coord, 0])
        o_i = o_desc.load([m_coord, 0])
        do_i = do_desc.load([m_coord, 0])
        
        m_offs = m_coord + tl.arange(0, BLOCK_M)
        l_i = tl.load(l_ptr + m_offs * stride_ls, mask=m_offs < S, other=0.0)
        
        delta_i = tl.sum(o_i.to(tl.float32) * do_i.to(tl.float32), axis=1)
        
        scores = tl.dot(q_i, tl.trans(k_j), out_dtype=tl.float32) * scale
        dp = tl.dot(do_i, tl.trans(v_j), out_dtype=tl.float32)
        
        valid_mask = (m_offs[:, None] >= n_offs[None, :]) & (m_offs[:, None] < S) & (n_offs[None, :] < S)
        scores = tl.where(valid_mask, scores, -float("inf"))
        
        p = tl.exp(scores - l_i[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        ds = p * (dp - delta_i[:, None]) * scale
        ds = tl.where(valid_mask, ds, 0.0)
        
        acc_dk = tl.dot(tl.trans(ds.to(tl.bfloat16)), q_i, acc=acc_dk, out_dtype=tl.float32)
        acc_dv = tl.dot(tl.trans(p.to(tl.bfloat16)), do_i, acc=acc_dv, out_dtype=tl.float32)
        
    dk_desc.store([n_coord, 0], acc_dk.to(tl.bfloat16))
    dv_desc.store([n_coord, 0], acc_dv.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    triton.set_allocator(alloc_fn)
    
    L_strides = L.stride()
    stride_lb, stride_lh, stride_ls = L_strides[0], L_strides[1], L_strides[2]
    
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    grid_q = (triton.cdiv(S, 128), H, B)
    bwd_q_kernel[grid_q](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        stride_lb, stride_lh, stride_ls,
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        S, scale,
        BLOCK_M=128, BLOCK_N=64, d=128,
        num_warps=8, num_stages=3
    )
    
    grid_kv = (triton.cdiv(S, 128), H, B)
    bwd_kv_kernel[grid_kv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        stride_lb, stride_lh, stride_ls,
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        S, scale,
        BLOCK_M=64, BLOCK_N=128, d=128,
        num_warps=8, num_stages=3
    )