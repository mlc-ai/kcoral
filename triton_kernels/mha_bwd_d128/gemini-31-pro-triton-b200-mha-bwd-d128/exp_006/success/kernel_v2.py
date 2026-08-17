import torch
import triton
import triton.language as tl

# Optimized autotune configurations for SM100 TMA paths
autotune_configs_dq = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
]

autotune_configs_dkdv = [
    triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_warps=4, num_stages=3),
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
    # L2 Swizzling for improved cache reuse of K and V blocks across programs
    pid_m_ = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    num_m = tl.cdiv(S, BLOCK_M)
    GROUP_M = 8
    group_id = pid_m_ // GROUP_M
    group_size = tl.minimum(num_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_m_ % group_size)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    dq_ptr = dQ + pid_b * stride_dqb + pid_h * stride_dqh
    
    # Device-created descriptors taking advantage of SM100 TMA
    desc_q = tl.make_tensor_descriptor(q_ptr, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_o = tl.make_tensor_descriptor(o_ptr, shape=[S, d], strides=[stride_os, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_do = tl.make_tensor_descriptor(do_ptr, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_dq = tl.make_tensor_descriptor(dq_ptr, shape=[S, d], strides=[stride_dqs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    
    desc_k = tl.make_tensor_descriptor(k_ptr, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    desc_v = tl.make_tensor_descriptor(v_ptr, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    
    offs_m = pid_m * BLOCK_M
    is_m_full = (offs_m + BLOCK_M) <= S
    
    q = desc_q.load([offs_m, 0])
    o = desc_o.load([offs_m, 0])
    do = desc_do.load([offs_m, 0])
    
    offs_m_arr = offs_m + tl.arange(0, BLOCK_M)
    l_ptr = L + pid_b * stride_lb + pid_h * stride_lh
    l = tl.load(l_ptr + offs_m_arr * stride_ls, mask=offs_m_arr < S, other=0.0)
    
    # Precompute Delta outside the loop using preloaded O and dO
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros((BLOCK_M, d), dtype=tl.float32)
    
    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    for kv_idx in range(0, num_kv_tiles):
        offs_n = kv_idx * BLOCK_N
        is_n_full = (offs_n + BLOCK_N) <= S
        
        k = desc_k.load([offs_n, 0])
        v = desc_v.load([offs_n, 0])
        
        if is_m_full and is_n_full:
            # Fast-path safe from out-of-bounds masks and branches
            acc_scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            scores = tl.dot(q, k.T, acc_scores) * softmax_scale
            
            p = tl.math.exp2((scores - l[:, None]) * 1.4426950408889634)
            
            acc_dp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            dp = tl.dot(do, v.T, acc_dp)
            
            ds = p * (dp - delta[:, None]) * softmax_scale
            dq = tl.dot(ds.to(q.dtype), k, dq)
        else:
            # Fallback handling boundary padding
            acc_scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            scores = tl.dot(q, k.T, acc_scores) * softmax_scale
            
            offs_n_arr = offs_n + tl.arange(0, BLOCK_N)
            valid = (offs_m_arr[:, None] < S) & (offs_n_arr[None, :] < S)
            scores = tl.where(valid, scores, -float("inf"))
            
            p = tl.math.exp2((scores - l[:, None]) * 1.4426950408889634)
            p = tl.where(valid, p, 0.0)
            
            acc_dp = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            dp = tl.dot(do, v.T, acc_dp)
            
            ds = p * (dp - delta[:, None]) * softmax_scale
            ds = tl.where(valid, ds, 0.0)
            
            dq = tl.dot(ds.to(q.dtype), k, dq)
            
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
    # L2 Swizzling for Q, O, dO reuse across concurrent program evaluations
    pid_n_ = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    num_n = tl.cdiv(S, BLOCK_N)
    GROUP_N = 8
    group_id = pid_n_ // GROUP_N
    group_size = tl.minimum(num_n - group_id * GROUP_N, GROUP_N)
    pid_n = group_id * GROUP_N + (pid_n_ % group_size)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    k_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    v_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    o_ptr = O + pid_b * stride_ob + pid_h * stride_oh
    do_ptr = dO + pid_b * stride_dob + pid_h * stride_doh
    dk_ptr = dK + pid_b * stride_dkb + pid_h * stride_dkh
    dv_ptr = dV + pid_b * stride_dvb + pid_h * stride_dvh
    
    desc_k = tl.make_tensor_descriptor(k_ptr, shape=[S, d], strides=[stride_ks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    desc_v = tl.make_tensor_descriptor(v_ptr, shape=[S, d], strides=[stride_vs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    desc_dk = tl.make_tensor_descriptor(dk_ptr, shape=[S, d], strides=[stride_dks, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    desc_dv = tl.make_tensor_descriptor(dv_ptr, shape=[S, d], strides=[stride_dvs, 1], block_shape=[BLOCK_N, d], padding_option="zero")
    
    desc_q = tl.make_tensor_descriptor(q_ptr, shape=[S, d], strides=[stride_qs, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_o = tl.make_tensor_descriptor(o_ptr, shape=[S, d], strides=[stride_os, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    desc_do = tl.make_tensor_descriptor(do_ptr, shape=[S, d], strides=[stride_dos, 1], block_shape=[BLOCK_M, d], padding_option="zero")
    
    offs_n = pid_n * BLOCK_N
    is_n_full = (offs_n + BLOCK_N) <= S
    
    k = desc_k.load([offs_n, 0])
    v = desc_v.load([offs_n, 0])
    
    dk = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, d), dtype=tl.float32)
    
    offs_n_arr = offs_n + tl.arange(0, BLOCK_N)
    
    num_q_tiles = tl.cdiv(S, BLOCK_M)
    for q_idx in range(0, num_q_tiles):
        offs_m = q_idx * BLOCK_M
        is_m_full = (offs_m + BLOCK_M) <= S
        
        q = desc_q.load([offs_m, 0])
        o = desc_o.load([offs_m, 0])
        do = desc_do.load([offs_m, 0])
        
        offs_m_arr = offs_m + tl.arange(0, BLOCK_M)
        l_ptr_m = L + pid_b * stride_lb + pid_h * stride_lh
        l = tl.load(l_ptr_m + offs_m_arr * stride_ls, mask=offs_m_arr < S, other=0.0)
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        if is_m_full and is_n_full:
            acc_scores_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            scores_t = tl.dot(k, q.T, acc_scores_t) * softmax_scale
            
            p_t = tl.math.exp2((scores_t - l[None, :]) * 1.4426950408889634)
            
            dv = tl.dot(p_t.to(q.dtype), do, dv)
            
            acc_dp_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            dp_t = tl.dot(v, do.T, acc_dp_t)
            
            ds_t = p_t * (dp_t - delta[None, :]) * softmax_scale
            dk = tl.dot(ds_t.to(q.dtype), q, dk)
        else:
            acc_scores_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            scores_t = tl.dot(k, q.T, acc_scores_t) * softmax_scale
            
            valid = (offs_n_arr[:, None] < S) & (offs_m_arr[None, :] < S)
            scores_t = tl.where(valid, scores_t, -float("inf"))
            
            p_t = tl.math.exp2((scores_t - l[None, :]) * 1.4426950408889634)
            p_t = tl.where(valid, p_t, 0.0)
            
            dv = tl.dot(p_t.to(q.dtype), do, dv)
            
            acc_dp_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            dp_t = tl.dot(v, do.T, acc_dp_t)
            
            ds_t = p_t * (dp_t - delta[None, :]) * softmax_scale
            ds_t = tl.where(valid, ds_t, 0.0)
            
            dk = tl.dot(ds_t.to(q.dtype), q, dk)
            
    desc_dk.store([offs_n, 0], dk.to(k.dtype))
    desc_dv.store([offs_n, 0], dv.to(v.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Compute destination-passing non-causal multi-head attention backward without atomics.
    Uses exclusive ownership assigning dQ to one dispatch grid and dK/dV to another.
    """
    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        softmax_scale = 1.0 / (d ** 0.5)

        # Utilize native host allocator exclusively for Triton device-created TMAs 
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