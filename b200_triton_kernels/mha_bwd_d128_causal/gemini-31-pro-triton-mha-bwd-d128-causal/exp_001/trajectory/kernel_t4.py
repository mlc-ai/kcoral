import torch
import triton
import triton.language as tl

@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, dO, L, dQ,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_o_b, stride_o_h, stride_o_s,
    stride_do_b, stride_do_h, stride_do_s,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dq_b, stride_dq_h, stride_dq_s,
    B, H, S, d_scale,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    start_q = pid_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    offset_q = pid_b * stride_q_b + pid_h * stride_q_h
    offset_k = pid_b * stride_k_b + pid_h * stride_k_h
    offset_v = pid_b * stride_v_b + pid_h * stride_v_h
    offset_o = pid_b * stride_o_b + pid_h * stride_o_h
    offset_do = pid_b * stride_do_b + pid_h * stride_do_h
    offset_l = pid_b * stride_l_b + pid_h * stride_l_h
    offset_dq = pid_b * stride_dq_b + pid_h * stride_dq_h

    q_desc = tl.make_tensor_descriptor(Q + offset_q, shape=[S, BLOCK_D], strides=[stride_q_s, 1], block_shape=[BLOCK_Q, BLOCK_D], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(O + offset_o, shape=[S, BLOCK_D], strides=[stride_o_s, 1], block_shape=[BLOCK_Q, BLOCK_D], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(dO + offset_do, shape=[S, BLOCK_D], strides=[stride_do_s, 1], block_shape=[BLOCK_Q, BLOCK_D], padding_option="zero")
    
    q = q_desc.load([start_q, 0])
    o = o_desc.load([start_q, 0])
    do = do_desc.load([start_q, 0])
    
    l = tl.load(L + offset_l + offs_q * stride_l_s, mask=offs_q < S, other=0.0)
    
    # Precompute d_val outside the K loop
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq_acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
    
    end_k = tl.minimum((start_q + BLOCK_Q - 1) // BLOCK_K + 1, tl.cdiv(S, BLOCK_K))
    
    k_desc = tl.make_tensor_descriptor(K + offset_k, shape=[S, BLOCK_D], strides=[stride_k_s, 1], block_shape=[BLOCK_K, BLOCK_D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(V + offset_v, shape=[S, BLOCK_D], strides=[stride_v_s, 1], block_shape=[BLOCK_K, BLOCK_D], padding_option="zero")
    
    for i in tl.range(0, end_k, num_stages=3):
        start_k = i * BLOCK_K
        k = k_desc.load([start_k, 0])
        v = v_desc.load([start_k, 0])
        
        # Native fast path: q is row-major, k.T is column-major 
        s_qk = tl.dot(q, k.T) * d_scale
        
        need_causal_mask = start_k + BLOCK_K > start_q
        need_s_mask = (start_q + BLOCK_Q > S) or (start_k + BLOCK_K > S)
        
        if need_causal_mask or need_s_mask:
            offs_k_curr = start_k + tl.arange(0, BLOCK_K)
            mask = (offs_q[:, None] >= offs_k_curr[None, :]) & (offs_q[:, None] < S) & (offs_k_curr[None, :] < S)
            s_qk = tl.where(mask, s_qk, float('-inf'))
            p = tl.exp(s_qk - l[:, None])
            p = tl.where(mask, p, 0.0)
        else:
            p = tl.exp(s_qk - l[:, None])
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - d_val[:, None]) * d_scale
        
        ds_bf16 = ds.to(q.dtype)
        dq_acc = tl.dot(ds_bf16, k, acc=dq_acc)
        
    dq_desc = tl.make_tensor_descriptor(dQ + offset_dq, shape=[S, BLOCK_D], strides=[stride_dq_s, 1], block_shape=[BLOCK_Q, BLOCK_D])
    dq_desc.store([start_q, 0], dq_acc.to(dQ.dtype.element_ty))


@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    stride_q_b, stride_q_h, stride_q_s,
    stride_k_b, stride_k_h, stride_k_s,
    stride_v_b, stride_v_h, stride_v_s,
    stride_o_b, stride_o_h, stride_o_s,
    stride_do_b, stride_do_h, stride_do_s,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dk_b, stride_dk_h, stride_dk_s,
    stride_dv_b, stride_dv_h, stride_dv_s,
    B, H, S, d_scale,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_k = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    start_k = pid_k * BLOCK_K
    offs_k = start_k + tl.arange(0, BLOCK_K)
    
    offset_k = pid_b * stride_k_b + pid_h * stride_k_h
    offset_v = pid_b * stride_v_b + pid_h * stride_v_h
    offset_q = pid_b * stride_q_b + pid_h * stride_q_h
    offset_o = pid_b * stride_o_b + pid_h * stride_o_h
    offset_do = pid_b * stride_do_b + pid_h * stride_do_h
    offset_l = pid_b * stride_l_b + pid_h * stride_l_h
    offset_dk = pid_b * stride_dk_b + pid_h * stride_dk_h
    offset_dv = pid_b * stride_dv_b + pid_h * stride_dv_h

    k_desc = tl.make_tensor_descriptor(K + offset_k, shape=[S, BLOCK_D], strides=[stride_k_s, 1], block_shape=[BLOCK_K, BLOCK_D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(V + offset_v, shape=[S, BLOCK_D], strides=[stride_v_s, 1], block_shape=[BLOCK_K, BLOCK_D], padding_option="zero")
    
    k = k_desc.load([start_k, 0])
    v = v_desc.load([start_k, 0])
    
    dk_acc = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_K, BLOCK_D], dtype=tl.float32)
    
    start_q_first = (start_k // BLOCK_Q) * BLOCK_Q
    num_q_steps = tl.cdiv(S - start_q_first, BLOCK_Q)
    
    q_desc = tl.make_tensor_descriptor(Q + offset_q, shape=[S, BLOCK_D], strides=[stride_q_s, 1], block_shape=[BLOCK_Q, BLOCK_D], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(O + offset_o, shape=[S, BLOCK_D], strides=[stride_o_s, 1], block_shape=[BLOCK_Q, BLOCK_D], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(dO + offset_do, shape=[S, BLOCK_D], strides=[stride_do_s, 1], block_shape=[BLOCK_Q, BLOCK_D], padding_option="zero")
    
    for i in tl.range(0, num_q_steps, num_stages=3):
        start_q = start_q_first + i * BLOCK_Q
        q = q_desc.load([start_q, 0])
        o = o_desc.load([start_q, 0])
        do = do_desc.load([start_q, 0])
        
        offs_q_curr = start_q + tl.arange(0, BLOCK_Q)
        l = tl.load(L + offset_l + offs_q_curr * stride_l_s, mask=offs_q_curr < S, other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        # Native WGMMA optimization: Compute directly s^T = k @ q.T to circumvent p.T register transposes
        s_kq = tl.dot(k, q.T) * d_scale
        
        need_causal_mask = start_q < start_k + BLOCK_K
        need_s_mask = (start_q + BLOCK_Q > S) or (start_k + BLOCK_K > S)
        
        if need_causal_mask or need_s_mask:
            mask = (offs_k[:, None] <= offs_q_curr[None, :]) & (offs_k[:, None] < S) & (offs_q_curr[None, :] < S)
            s_kq = tl.where(mask, s_kq, float('-inf'))
            p_kq = tl.exp(s_kq - l[None, :])
            p_kq = tl.where(mask, p_kq, 0.0)
        else:
            p_kq = tl.exp(s_kq - l[None, :])
            
        p_kq_bf16 = p_kq.to(do.dtype)
        dv_acc = tl.dot(p_kq_bf16, do, acc=dv_acc)
        
        # Compute dp^T iteratively: dp_kq = v @ do.T
        dp_kq = tl.dot(v, do.T)
        ds_kq = p_kq * (dp_kq - d_val[None, :]) * d_scale
        
        ds_kq_bf16 = ds_kq.to(q.dtype)
        dk_acc = tl.dot(ds_kq_bf16, q, acc=dk_acc)
        
    dk_desc = tl.make_tensor_descriptor(dK + offset_dk, shape=[S, BLOCK_D], strides=[stride_dk_s, 1], block_shape=[BLOCK_K, BLOCK_D])
    dv_desc = tl.make_tensor_descriptor(dV + offset_dv, shape=[S, BLOCK_D], strides=[stride_dv_s, 1], block_shape=[BLOCK_K, BLOCK_D])
    
    dk_desc.store([start_k, 0], dk_acc.to(dK.dtype.element_ty))
    dv_desc.store([start_k, 0], dv_acc.to(dV.dtype.element_ty))


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

_allocator_set = False

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes standard causal multi-head attention backward pass.
    Writes gradients in place into dQ, dK, dV mapping seamlessly via Hopper's TMA.
    """
    global _allocator_set
    if not _allocator_set:
        triton.set_allocator(alloc_fn)
        _allocator_set = True

    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        d_scale = 1.0 / (d ** 0.5)
        
        BLOCK_D = 128
        
        # Sizing balances Shared Memory budget (~228 KiB physical on H100) vs execution efficiency
        BLOCK_Q_DQ = 128
        BLOCK_K_DQ = 64
        
        grid_dq = (triton.cdiv(S, BLOCK_Q_DQ), B * H)
        bwd_kernel_dq[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            B, H, S, d_scale,
            BLOCK_Q_DQ, BLOCK_K_DQ, BLOCK_D,
            num_warps=4, num_stages=3
        )
        
        BLOCK_Q_DK = 64
        BLOCK_K_DK = 128
        
        grid_dk_dv = (triton.cdiv(S, BLOCK_K_DK), B * H)
        bwd_kernel_dk_dv[grid_dk_dv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            B, H, S, d_scale,
            BLOCK_Q_DK, BLOCK_K_DK, BLOCK_D,
            num_warps=4, num_stages=3
        )
        
        return dQ, dK, dV