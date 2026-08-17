import torch
import triton
import triton.language as tl

# Optimized autotune configurations mapping accurately to Blackwell TMEM + MMA characteristics
autotune_configs_dq = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
]

autotune_configs_dkdv = [
    triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=autotune_configs_dq, key=['S'])
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs,
    H, S, d: tl.constexpr,
    softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    dq_ptr = dQ + pid_b * stride_dqb + pid_h * stride_dqh
    l_ptr = L + pid_b * stride_lb + pid_h * stride_lh
    
    # Device-created descriptors lower directly to Blackwell TMA instructions
    desc_q = tl.make_tensor_descriptor(q_ptr, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_o = tl.make_tensor_descriptor(o_ptr, shape=[S, d], strides=[stride_os, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_do = tl.make_tensor_descriptor(do_ptr, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_dq = tl.make_tensor_descriptor(dq_ptr, shape=[S, d], strides=[stride_dqs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_k = tl.make_tensor_descriptor(k_ptr, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    desc_v = tl.make_tensor_descriptor(v_ptr, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    
    offs_m = pid_m * BLOCK_M
    q = desc_q.load([offs_m, 0])
    o = desc_o.load([offs_m, 0])
    do = desc_do.load([offs_m, 0])
    
    is_m_full = (offs_m + BLOCK_M) <= S
    offs_m_arr = offs_m + tl.arange(0, BLOCK_M)
    valid_m = offs_m_arr < S
    
    l = tl.load(l_ptr + offs_m_arr * stride_ls, mask=valid_m, other=0.0)
    
    # Combine scalar math multipliers uniformly 
    scale_log2 = softmax_scale * 1.4426950408889634
    l_log2 = l * 1.4426950408889634
    
    # Precompute Delta securely out of the loop
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    # Accelerated tight loop exploiting unmasked pipelining for full alignment scenarios
    num_kv_tiles_full = S // BLOCK_N
    for kv_idx in range(0, num_kv_tiles_full):
        offs_n = kv_idx * BLOCK_N
        k = desc_k.load([offs_n, 0])
        v = desc_v.load([offs_n, 0])
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * scale_log2
        if not is_m_full:
            scores = tl.where(valid_m[:, None], scores, -float('inf'))
            
        p = tl.math.exp2(scores - l_log2[:, None])
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * softmax_scale
        
        dq = tl.dot(ds.to(q.dtype), k, acc=dq, out_dtype=tl.float32)

    # Resolve tail limits defensively
    if S % BLOCK_N != 0:
        offs_n = num_kv_tiles_full * BLOCK_N
        k = desc_k.load([offs_n, 0])
        v = desc_v.load([offs_n, 0])
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32) * scale_log2
        valid_n = (offs_n + tl.arange(0, BLOCK_N)) < S
        valid = valid_n[None, :]
        if not is_m_full:
            valid = valid & valid_m[:, None]
            
        scores = tl.where(valid, scores, -float('inf'))
        p = tl.math.exp2(scores - l_log2[:, None])
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * softmax_scale
        
        dq = tl.dot(ds.to(q.dtype), k, acc=dq, out_dtype=tl.float32)
        
    desc_dq.store([offs_m, 0], dq.to(q.dtype))


@triton.autotune(configs=autotune_configs_dkdv, key=['S'])
@triton.jit
def bwd_dkdv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks,
    stride_dvb, stride_dvh, stride_dvs,
    H, S, d: tl.constexpr,
    softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    dk_ptr = dK + pid_b * stride_dkb + pid_h * stride_dkh
    dv_ptr = dV + pid_b * stride_dvb + pid_h * stride_dvh
    l_ptr = L + pid_b * stride_lb + pid_h * stride_lh
    
    desc_k = tl.make_tensor_descriptor(k_ptr, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    desc_v = tl.make_tensor_descriptor(v_ptr, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    desc_dk = tl.make_tensor_descriptor(dk_ptr, shape=[S, d], strides=[stride_dks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    desc_dv = tl.make_tensor_descriptor(dv_ptr, shape=[S, d], strides=[stride_dvs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    
    desc_q = tl.make_tensor_descriptor(q_ptr, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_o = tl.make_tensor_descriptor(o_ptr, shape=[S, d], strides=[stride_os, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_do = tl.make_tensor_descriptor(do_ptr, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    
    offs_n = pid_n * BLOCK_N
    k = desc_k.load([offs_n, 0])
    v = desc_v.load([offs_n, 0])
    
    is_n_full = (offs_n + BLOCK_N) <= S
    offs_n_arr = offs_n + tl.arange(0, BLOCK_N)
    valid_n = offs_n_arr < S
    
    dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    scale_log2 = softmax_scale * 1.4426950408889634
    
    # Tight pipelined full inner loop skipping register-heavy masks internally
    num_q_tiles_full = S // BLOCK_M
    for q_idx in range(0, num_q_tiles_full):
        offs_m = q_idx * BLOCK_M
        
        q = desc_q.load([offs_m, 0])
        o = desc_o.load([offs_m, 0])
        do = desc_do.load([offs_m, 0])
        
        offs_m_arr = offs_m + tl.arange(0, BLOCK_M)
        l = tl.load(l_ptr + offs_m_arr * stride_ls)
        l_log2 = l * 1.4426950408889634
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        scores_t = tl.dot(k, q.T, out_dtype=tl.float32) * scale_log2
        if not is_n_full:
            scores_t = tl.where(valid_n[:, None], scores_t, -float('inf'))
            
        p_t = tl.math.exp2(scores_t - l_log2[None, :])
        
        dv = tl.dot(p_t.to(q.dtype), do, acc=dv, out_dtype=tl.float32)
        
        dp_t = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_t = p_t * (dp_t - delta[None, :]) * softmax_scale
        
        dk = tl.dot(ds_t.to(q.dtype), q, acc=dk, out_dtype=tl.float32)

    # Safe tail processing 
    if S % BLOCK_M != 0:
        offs_m = num_q_tiles_full * BLOCK_M
        q = desc_q.load([offs_m, 0])
        o = desc_o.load([offs_m, 0])
        do = desc_do.load([offs_m, 0])
        
        offs_m_arr = offs_m + tl.arange(0, BLOCK_M)
        valid_m = offs_m_arr < S
        l = tl.load(l_ptr + offs_m_arr * stride_ls, mask=valid_m, other=0.0)
        l_log2 = l * 1.4426950408889634
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        scores_t = tl.dot(k, q.T, out_dtype=tl.float32) * scale_log2
        valid = valid_m[None, :]
        if not is_n_full:
            valid = valid & valid_n[:, None]
            
        scores_t = tl.where(valid, scores_t, -float('inf'))
        p_t = tl.math.exp2(scores_t - l_log2[None, :])
        
        dv = tl.dot(p_t.to(q.dtype), do, acc=dv, out_dtype=tl.float32)
        
        dp_t = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_t = p_t * (dp_t - delta[None, :]) * softmax_scale
        
        dk = tl.dot(ds_t.to(q.dtype), q, acc=dk, out_dtype=tl.float32)
        
    desc_dk.store([offs_n, 0], dk.to(k.dtype))
    desc_dv.store([offs_n, 0], dv.to(v.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Compute destination-passing non-causal multi-head attention backward cleanly.
    Separates output ownership across `dQ` and `dK/dV` regions implementing tight unmasked full-tile internal loops.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        softmax_scale = 1.0 / (d ** 0.5)

        def alloc_fn(size: int, alignment: int, stream):
            return torch.empty(size, device=Q.device, dtype=torch.int8)
        triton.set_allocator(alloc_fn)

        grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
        bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            H, S, d,
            softmax_scale
        )

        grid_dkdv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
        bwd_dkdv_kernel[grid_dkdv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            H, S, d,
            softmax_scale
        )