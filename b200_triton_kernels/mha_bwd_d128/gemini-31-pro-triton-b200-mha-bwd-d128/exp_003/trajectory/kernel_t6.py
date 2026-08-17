import math
import torch
import triton
import triton.language as tl

# dQ kernel config bounds:
# Maximize M to iterate over N optimally. Keeps pipeline small: K & V inside the loop.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_kernel_dq(
    Q, K, V, O, dO, L, dQ,
    sm_scale,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    q_offset = b * stride_qb + h * stride_qh
    k_offset = b * stride_kb + h * stride_kh
    v_offset = b * stride_vb + h * stride_vh
    o_offset = b * stride_ob + h * stride_oh
    do_offset = b * stride_dob + h * stride_doh
    l_offset = b * stride_lb + h * stride_lh
    dq_offset = b * stride_dqb + h * stride_dqh
    
    # 2D Blackwell descriptors offload memory bounds checking
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset, shape=[S, d], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_offset, shape=[S, d], strides=[stride_os, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + do_offset, shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset, shape=[S, d], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset, shape=[S, d], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dq_desc = tl.make_tensor_descriptor(
        dQ + dq_offset, shape=[S, d], strides=[stride_dqs, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    
    offs_m = pid_m * BLOCK_M
    q = q_desc.load([offs_m, 0])
    o = o_desc.load([offs_m, 0])
    do = do_desc.load([offs_m, 0])
    
    offs_m_arr = offs_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m_arr < S
    
    l_ptrs = L + l_offset + offs_m_arr * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    do_f32 = do.to(tl.float32)
    o_f32 = o.to(tl.float32)
    D = tl.sum(do_f32 * o_f32, axis=1)
    
    acc_dq = tl.zeros([BLOCK_M, d], dtype=tl.float32)
    
    for j0 in range(0, tl.cdiv(S, BLOCK_N)):
        offs_n = j0 * BLOCK_N
        
        k = k_desc.load([offs_n, 0])
        v = v_desc.load([offs_n, 0])
        
        s_attn = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        
        offs_n_arr = offs_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n_arr < S
        mask_mn = mask_m[:, None] & mask_n[None, :]
        
        s_attn = tl.where(mask_mn, s_attn, float("-inf"))
        p = tl.exp(s_attn - l[:, None])
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        ds = p * (dp - D[:, None]) * sm_scale
        
        acc_dq = tl.dot(ds.to(tl.bfloat16), k, acc=acc_dq)
        
    dq_desc.store([offs_m, 0], acc_dq.to(tl.bfloat16))


# dK/dV kernel config bounds:
# Maximize N to iterate over M optimally. Keeps pipeline small: Q, O, dO inside the loop.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def bwd_kernel_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    sm_scale,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_dob, stride_doh, stride_dos,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks,
    stride_dvb, stride_dvh, stride_dvs,
    B, H, S, d: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b = pid_bh // H
    h = pid_bh % H
    
    q_offset = b * stride_qb + h * stride_qh
    k_offset = b * stride_kb + h * stride_kh
    v_offset = b * stride_vb + h * stride_vh
    o_offset = b * stride_ob + h * stride_oh
    do_offset = b * stride_dob + h * stride_doh
    l_offset = b * stride_lb + h * stride_lh
    dk_offset = b * stride_dkb + h * stride_dkh
    dv_offset = b * stride_dvb + h * stride_dvh
    
    k_desc = tl.make_tensor_descriptor(
        K + k_offset, shape=[S, d], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset, shape=[S, d], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dk_desc = tl.make_tensor_descriptor(
        dK + dk_offset, shape=[S, d], strides=[stride_dks, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + dv_offset, shape=[S, d], strides=[stride_dvs, 1],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset, shape=[S, d], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_offset, shape=[S, d], strides=[stride_os, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + do_offset, shape=[S, d], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    
    offs_n = pid_n * BLOCK_N
    k = k_desc.load([offs_n, 0])
    v = v_desc.load([offs_n, 0])
    
    offs_n_arr = offs_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n_arr < S
    
    acc_dk = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    acc_dv = tl.zeros([BLOCK_N, d], dtype=tl.float32)
    
    for i0 in range(0, tl.cdiv(S, BLOCK_M)):
        offs_m = i0 * BLOCK_M
        
        q = q_desc.load([offs_m, 0])
        o = o_desc.load([offs_m, 0])
        do = do_desc.load([offs_m, 0])
        
        offs_m_arr = offs_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m_arr < S
        
        l_ptrs = L + l_offset + offs_m_arr * stride_ls
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        do_f32 = do.to(tl.float32)
        o_f32 = o.to(tl.float32)
        D = tl.sum(do_f32 * o_f32, axis=1)
        
        # Mathematical Trick: generate transposed arrays directly to avoid explicit `.trans()` inside the loop!
        # Computing S^T = K @ Q^T results in a shape of [BLOCK_N, BLOCK_M]
        s_attn_T = tl.dot(k, tl.trans(q), out_dtype=tl.float32) * sm_scale
        
        mask_nm = mask_n[:, None] & mask_m[None, :]
        s_attn_T = tl.where(mask_nm, s_attn_T, float("-inf"))
        
        p_T = tl.exp(s_attn_T - l[None, :]) # Shape: [BLOCK_N, BLOCK_M]
        
        # Computing dP^T = V @ dO^T results in a shape of [BLOCK_N, BLOCK_M]
        dp_T = tl.dot(v, tl.trans(do), out_dtype=tl.float32)
        
        # Vector broadcasting handles alignment naturally across the reduced axis
        ds_T = p_T * (dp_T - D[None, :]) * sm_scale 
        
        # Use generated matrices symmetrically without incurring shared-memory transposes
        acc_dk = tl.dot(ds_T.to(tl.bfloat16), q, acc=acc_dk)
        acc_dv = tl.dot(p_T.to(tl.bfloat16), do, acc=acc_dv)
        
    dk_desc.store([offs_n, 0], acc_dk.to(tl.bfloat16))
    dv_desc.store([offs_n, 0], acc_dv.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    # Enforce safe SMEM fallback allocator to construct dynamic device TensorDescriptors 
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)

    with torch.cuda.device(Q.device):
        B, H, S, d = Q.shape
        sm_scale = 1.0 / math.sqrt(d)
        
        if L.dim() == 4:
            L = L.squeeze(-1)
            
        grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H, 1)
        bwd_kernel_dq[grid_dq](
            Q, K, V, O, dO, L, dQ,
            sm_scale,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            B, H, S, d=d
        )
        
        grid_dk_dv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H, 1)
        bwd_kernel_dk_dv[grid_dk_dv](
            Q, K, V, O, dO, L, dK, dV,
            sm_scale,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            B, H, S, d=d
        )