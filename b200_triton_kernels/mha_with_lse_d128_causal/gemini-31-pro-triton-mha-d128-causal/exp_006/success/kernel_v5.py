import math
import torch
import triton
import triton.language as tl


def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 4}, num_stages=4, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def _causal_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale_log2, S, H,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(S, BLOCK_M)
    
    # L2 Cache Swizzling
    num_pid_in_group = GROUP_M
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    start_m = first_pid_m + (pid % group_size_m)
    batch_head = tl.program_id(1)
    
    offset_m = start_m * BLOCK_M
    if offset_m >= S:
        return
        
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    q_ptr = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_ptr = K + batch_idx * stride_kb + head_idx * stride_kh
    v_ptr = V + batch_idx * stride_vb + head_idx * stride_vh
    o_ptr = O + batch_idx * stride_ob + head_idx * stride_oh
    
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[S, BLOCK_DMODEL], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_DMODEL], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[S, BLOCK_DMODEL], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_DMODEL], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[S, BLOCK_DMODEL], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_DMODEL], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[S, BLOCK_DMODEL], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_DMODEL]
    )
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    q = q_desc.load([offset_m, 0])
    
    limit_n = offset_m // BLOCK_N
    max_k = tl.minimum(S, offset_m + BLOCK_M)
    num_k_blocks = tl.cdiv(max_k, BLOCK_N)
    
    for start_n in range(0, num_k_blocks):
        offset_n = start_n * BLOCK_N
        k = k_desc.load([offset_n, 0])
        
        qk = tl.dot(q, k.trans(1, 0), out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        if start_n >= limit_n:
            offs_n_curr = offset_n + offs_n
            causal_mask = offs_m[:, None] >= offs_n_curr[None, :]
            seq_mask = offs_n_curr[None, :] < S
            mask = causal_mask & seq_mask
            qk = tl.where(mask, qk, float("-inf"))
            
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        l_i_new = alpha * l_i + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        
        v = v_desc.load([offset_n, 0])
        p_cast = p.to(Q.dtype.element_ty)
        acc = tl.dot(p_cast, v, acc, out_dtype=tl.float32)
        
        m_i = m_i_new
        l_i = l_i_new

    acc = acc * (1.0 / l_i[:, None])
    o_desc.store([offset_m, 0], acc.to(Q.dtype.element_ty))
    
    lse = m_i * 0.6931471805599453 + tl.log(l_i)
    lse_ptrs = LSE + batch_idx * stride_lseb + head_idx * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    if Q.shape[2] == 0:
        return
        
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    sm_scale = 1.0 / math.sqrt(D)
    sm_scale_log2 = sm_scale * 1.4426950408889634

    def grid(META):
        num_pid_m = triton.cdiv(S, META['BLOCK_M'])
        return (num_pid_m, B * H)

    _causal_fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale_log2, S, H,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_DMODEL=128
    )