import torch
import triton
import triton.language as tl


# Required Triton host setup for device-created tensor descriptors 
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=5, num_warps=4),
    ],
    key=['seq_len']
)
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
    num_heads, seq_len, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    pid_m = tl.program_id(0)
    bh_idx = tl.program_id(1)

    batch = bh_idx // num_heads
    head = bh_idx % num_heads

    m_offset = pid_m * BLOCK_M
    offs_m = m_offset + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len

    Q_ptr = Q + batch * stride_qb + head * stride_qh
    O_ptr = O + batch * stride_ob + head * stride_oh
    dO_ptr = dO + batch * stride_dob + head * stride_doh
    dQ_ptr = dQ + batch * stride_dqb + head * stride_dqh

    # Blackwell natively loads standard 2D continuous spaces using TMA
    q_desc = tl.make_tensor_descriptor(
        Q_ptr, shape=[seq_len, D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O_ptr, shape=[seq_len, D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO_ptr, shape=[seq_len, D], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    dq_desc = tl.make_tensor_descriptor(
        dQ_ptr, shape=[seq_len, D], strides=[stride_dqs, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )

    q = q_desc.load([m_offset, 0])
    o = o_desc.load([m_offset, 0])
    do = do_desc.load([m_offset, 0])

    l_ptrs = L + batch * stride_lb + head * stride_lh + offs_m * stride_ls
    l = tl.load(l_ptrs, mask=mask_m, other=0.0)

    # Compute rowwise delta = sum(dO * O, axis=1) locally in FP32
    do_fp32 = tl.cast(do, tl.float32)
    o_fp32 = tl.cast(o, tl.float32)
    delta = tl.sum(do_fp32 * o_fp32, axis=1)

    dq = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    K_ptr = K + batch * stride_kb + head * stride_kh
    V_ptr = V + batch * stride_vb + head * stride_vh
    k_desc = tl.make_tensor_descriptor(
        K_ptr, shape=[seq_len, D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr, shape=[seq_len, D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )

    num_kv_blocks = tl.cdiv(seq_len, BLOCK_N)
    for n_idx in range(num_kv_blocks):
        n_offset = n_idx * BLOCK_N
        offs_n = n_offset + tl.arange(0, BLOCK_N)

        k = k_desc.load([n_offset, 0])
        v = v_desc.load([n_offset, 0])

        s = tl.dot(q, tl.trans(k))
        s = s * sm_scale

        valid = mask_m[:, None] & (offs_n[None, :] < seq_len)
        s = tl.where(valid, s, float('-inf'))

        # Use FP32 Native HW scaling explicitly over exp to dodge base scaling precision loss
        p = tl.math.exp2((s - l[:, None]) * 1.4426950408889634)
        p = tl.where(valid, p, 0.0)

        dp = tl.dot(do, tl.trans(v))

        ds = p * (dp - delta[:, None]) * sm_scale
        ds = tl.where(valid, ds, 0.0)
        ds_bf16 = tl.cast(ds, tl.bfloat16)

        dq = tl.dot(ds_bf16, k, dq)

    dq_desc.store([m_offset, 0], tl.cast(dq, tl.bfloat16))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=5, num_warps=4),
    ],
    key=['seq_len']
)
@triton.jit
def bwd_dkv_kernel(
    Q, K, V, O, dO, dK, dV, L,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    num_heads, seq_len, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    pid_n = tl.program_id(0)
    bh_idx = tl.program_id(1)

    batch = bh_idx // num_heads
    head = bh_idx % num_heads

    n_offset = pid_n * BLOCK_N
    offs_n = n_offset + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seq_len

    K_ptr = K + batch * stride_kb + head * stride_kh
    V_ptr = V + batch * stride_vb + head * stride_vh
    dK_ptr = dK + batch * stride_dkb + head * stride_dkh
    dV_ptr = dV + batch * stride_dvb + head * stride_dvh

    k_desc = tl.make_tensor_descriptor(
        K_ptr, shape=[seq_len, D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr, shape=[seq_len, D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    dk_desc = tl.make_tensor_descriptor(
        dK_ptr, shape=[seq_len, D], strides=[stride_dks, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    dv_desc = tl.make_tensor_descriptor(
        dV_ptr, shape=[seq_len, D], strides=[stride_dvs, 1],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )

    k = k_desc.load([n_offset, 0])
    v = v_desc.load([n_offset, 0])

    dk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    Q_ptr = Q + batch * stride_qb + head * stride_qh
    O_ptr = O + batch * stride_ob + head * stride_oh
    dO_ptr = dO + batch * stride_dob + head * stride_doh

    q_desc = tl.make_tensor_descriptor(
        Q_ptr, shape=[seq_len, D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O_ptr, shape=[seq_len, D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO_ptr, shape=[seq_len, D], strides=[stride_dos, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )

    num_q_blocks = tl.cdiv(seq_len, BLOCK_M)
    for m_idx in range(num_q_blocks):
        m_offset = m_idx * BLOCK_M
        offs_m = m_offset + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len

        q = q_desc.load([m_offset, 0])
        do = do_desc.load([m_offset, 0])
        o = o_desc.load([m_offset, 0])

        l_ptrs = L + batch * stride_lb + head * stride_lh + offs_m * stride_ls
        l = tl.load(l_ptrs, mask=mask_m, other=0.0)

        # Delta recomputed seamlessly over memory streamed queries 
        do_fp32 = tl.cast(do, tl.float32)
        o_fp32 = tl.cast(o, tl.float32)
        delta = tl.sum(do_fp32 * o_fp32, axis=1)

        # Math transpose avoiding memory spills: S^T = K @ Q^T
        s_t = tl.dot(k, tl.trans(q))
        s_t = s_t * sm_scale

        valid = mask_n[:, None] & mask_m[None, :]
        s_t = tl.where(valid, s_t, float('-inf'))

        p_t = tl.math.exp2((s_t - l[None, :]) * 1.4426950408889634)
        p_t = tl.where(valid, p_t, 0.0)

        # dP^T = V @ dO^T
        dp_t = tl.dot(v, tl.trans(do))
        
        ds_t = p_t * (dp_t - delta[None, :]) * sm_scale
        ds_t = tl.where(valid, ds_t, 0.0)
        
        ds_t_bf16 = tl.cast(ds_t, tl.bfloat16)
        p_t_bf16 = tl.cast(p_t, tl.bfloat16)

        # Maps securely back to hardware mapped FP32 TMEM matrices 
        dk = tl.dot(ds_t_bf16, q, dk)
        dv = tl.dot(p_t_bf16, do, dv)

    dk_desc.store([n_offset, 0], tl.cast(dk, tl.bfloat16))
    dv_desc.store([n_offset, 0], tl.cast(dv, tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Standard standard-Triton non-causal multi-head attention backward kernel optimized for TMA/TMEM on Blackwell SM100.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sm_scale = 1.0 / (d ** 0.5)
    
    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, dQ, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        num_heads=H, seq_len=S, sm_scale=sm_scale, D=d
    )

    grid_dkv = lambda META: (triton.cdiv(S, META['BLOCK_N']), B * H)
    bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, dK, dV, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        num_heads=H, seq_len=S, sm_scale=sm_scale, D=d
    )