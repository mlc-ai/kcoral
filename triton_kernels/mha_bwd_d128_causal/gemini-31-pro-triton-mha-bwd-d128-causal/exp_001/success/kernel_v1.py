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

    # Descriptors for standard row-major (S, BLOCK_D) blocks
    q_desc = tl.make_tensor_descriptor(Q + offset_q, shape=[S, BLOCK_D], strides=[stride_q_s, 1], block_shape=[BLOCK_Q, BLOCK_D], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(O + offset_o, shape=[S, BLOCK_D], strides=[stride_o_s, 1], block_shape=[BLOCK_Q, BLOCK_D], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(dO + offset_do, shape=[S, BLOCK_D], strides=[stride_do_s, 1], block_shape=[BLOCK_Q, BLOCK_D], padding_option="zero")
    
    q = q_desc.load([start_q, 0])
    o = o_desc.load([start_q, 0])
    do = do_desc.load([start_q, 0])
    
    # L is 1-dimensional for this (B, H) slice, load it with bounds check
    l = tl.load(L + offset_l + offs_q * stride_l_s, mask=offs_q < S, other=0.0)
    
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq_acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
    
    end_k = tl.minimum((start_q + BLOCK_Q - 1) // BLOCK_K + 1, tl.cdiv(S, BLOCK_K))
    
    k_desc = tl.make_tensor_descriptor(K + offset_k, shape=[S, BLOCK_D], strides=[stride_k_s, 1], block_shape=[BLOCK_K, BLOCK_D], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(V + offset_v, shape=[S, BLOCK_D], strides=[stride_v_s, 1], block_shape=[BLOCK_K, BLOCK_D], padding_option="zero")
    
    for i in range(end_k):
        start_k = i * BLOCK_K
        k = k_desc.load([start_k, 0])
        v = v_desc.load([start_k, 0])
        
        s_qk = tl.dot(q, k.T) * d_scale
        
        # Causal mask bounds check
        offs_k_curr = start_k + tl.arange(0, BLOCK_K)
        mask = (offs_q[:, None] >= offs_k_curr[None, :]) & (offs_q[:, None] < S) & (offs_k_curr[None, :] < S)
        s_qk = tl.where(mask, s_qk, float('-inf'))
        
        p = tl.exp(s_qk - l[:, None])
        p = tl.where(mask, p, 0.0)
        
        dp = tl.dot(do, v.T)
        
        # Compute gradient w.r.t attention scores
        ds = p * (dp - d_val[:, None]) * d_scale
        
        dq_acc = tl.dot(ds.to(q.dtype), k, acc=dq_acc)
        
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
    
    # Calculate earliest eligible Q block relying on the causal mask
    start_q_first = (start_k // BLOCK_Q) * BLOCK_Q
    num_q_steps = tl.cdiv(S - start_q_first, BLOCK_Q)
    
    q_desc = tl.make_tensor_descriptor(Q + offset_q, shape=[S, BLOCK_D], strides=[stride_q_s, 1], block_shape=[BLOCK_Q, BLOCK_D], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(O + offset_o, shape=[S, BLOCK_D], strides=[stride_o_s, 1], block_shape=[BLOCK_Q, BLOCK_D], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(dO + offset_do, shape=[S, BLOCK_D], strides=[stride_do_s, 1], block_shape=[BLOCK_Q, BLOCK_D], padding_option="zero")
    
    for i in range(num_q_steps):
        start_q = start_q_first + i * BLOCK_Q
        q = q_desc.load([start_q, 0])
        o = o_desc.load([start_q, 0])
        do = do_desc.load([start_q, 0])
        
        offs_q_curr = start_q + tl.arange(0, BLOCK_Q)
        l = tl.load(L + offset_l + offs_q_curr * stride_l_s, mask=offs_q_curr < S, other=0.0)
        
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        s_qk = tl.dot(q, k.T) * d_scale
        
        mask = (offs_q_curr[:, None] >= offs_k[None, :]) & (offs_q_curr[:, None] < S) & (offs_k[None, :] < S)
        s_qk = tl.where(mask, s_qk, float('-inf'))
        
        p = tl.exp(s_qk - l[:, None])
        p = tl.where(mask, p, 0.0)
        
        dv_acc = tl.dot(tl.trans(p).to(do.dtype), do, acc=dv_acc)
        
        dp = tl.dot(do, v.T)
        ds = p * (dp - d_val[:, None]) * d_scale
        dk_acc = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk_acc)
        
    dk_desc = tl.make_tensor_descriptor(dK + offset_dk, shape=[S, BLOCK_D], strides=[stride_dk_s, 1], block_shape=[BLOCK_K, BLOCK_D])
    dv_desc = tl.make_tensor_descriptor(dV + offset_dv, shape=[S, BLOCK_D], strides=[stride_dv_s, 1], block_shape=[BLOCK_K, BLOCK_D])
    
    dk_desc.store([start_k, 0], dk_acc.to(dK.dtype.element_ty))
    dv_desc.store([start_k, 0], dv_acc.to(dV.dtype.element_ty))


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

_allocator_set = False

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    global _allocator_set
    if not _allocator_set:
        triton.set_allocator(alloc_fn)
        _allocator_set = True

    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        d_scale = 1.0 / (d ** 0.5)
        
        BLOCK_D = 128
        
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
            num_warps=8, num_stages=3
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
            num_warps=8, num_stages=3
        )
        
        return dQ, dK, dV