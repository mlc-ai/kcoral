import math
import torch
import triton
import triton.language as tl

# Set up allocator strictly for device-created tensor descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_preprocess_config():
    return [
        triton.Config({"BLOCK_M": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 128}, num_warps=4),
    ]

@triton.autotune(configs=get_preprocess_config(), key=['S'])
@triton.jit
def bwd_preprocess_kernel(
    O_ptr, dO_ptr, dQ_ptr,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    offset_m = pid_m * BLOCK_M
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    off_b_h_o = pid_b * stride_ob + pid_h * stride_oh
    desc_o = tl.make_tensor_descriptor(
        O_ptr + off_b_h_o, shape=[S, BLOCK_D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    off_b_h_do = pid_b * stride_dob + pid_h * stride_doh
    desc_do = tl.make_tensor_descriptor(
        dO_ptr + off_b_h_do, shape=[S, BLOCK_D], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    o = desc_o.load([offset_m, 0])
    do = desc_do.load([offset_m, 0])
    
    # Precompute rowsum(dO * O)
    Di = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    # Stash Di (float32) into the beginning of each row of dQ (bfloat16) to avoid reloading O later
    off_b_h_dq = pid_b * stride_dqb + pid_h * stride_dqh
    row_ptrs = dQ_ptr + off_b_h_dq + offs_m * stride_dqs
    f_ptrs = tl.cast(row_ptrs, tl.pointer_type(tl.float32))
    tl.store(f_ptrs, Di, mask=mask_m)


def get_autotune_config_dkdv():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ]

@triton.autotune(configs=get_autotune_config_dkdv(), key=['S'])
@triton.jit
def bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offset_n = pid_n * BLOCK_N
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n = offset_n + offs_n_base
    mask_n = offs_n < S
    
    off_b_h_k = pid_b * stride_kb + pid_h * stride_kh
    desc_k = tl.make_tensor_descriptor(
        K_ptr + off_b_h_k, shape=[S, BLOCK_D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    off_b_h_v = pid_b * stride_vb + pid_h * stride_vh
    desc_v = tl.make_tensor_descriptor(
        V_ptr + off_b_h_v, shape=[S, BLOCK_D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    
    k = desc_k.load([offset_n, 0])
    v = desc_v.load([offset_n, 0])

    dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

    off_b_h_q = pid_b * stride_qb + pid_h * stride_qh
    desc_q = tl.make_tensor_descriptor(
        Q_ptr + off_b_h_q, shape=[S, BLOCK_D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    off_b_h_do = pid_b * stride_dob + pid_h * stride_doh
    desc_do = tl.make_tensor_descriptor(
        dO_ptr + off_b_h_do, shape=[S, BLOCK_D], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )

    off_b_h_l = pid_b * stride_lb + pid_h * stride_lh
    l_base = L_ptr + off_b_h_l
    
    off_b_h_dq = pid_b * stride_dqb + pid_h * stride_dqh
    dq_base = dQ_ptr + off_b_h_dq

    offs_m_base = tl.arange(0, BLOCK_M)

    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for m_block in range(num_m_blocks):
        offset_m = m_block * BLOCK_M
        offs_m = offset_m + offs_m_base
        mask_m = offs_m < S
        
        q = desc_q.load([offset_m, 0])
        do = desc_do.load([offset_m, 0])
        
        l_i = tl.load(l_base + offs_m * stride_ls, mask=mask_m, other=0.0)
        
        # Read the efficiently pre-computed Di instead of reloading O tensor entirely
        row_ptrs = dq_base + offs_m * stride_dqs
        f_ptrs = tl.cast(row_ptrs, tl.pointer_type(tl.float32))
        Di = tl.load(f_ptrs, mask=mask_m, other=0.0)
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk = qk * scale

        mask_2d = mask_m[:, None] & mask_n[None, :]
        qk = tl.where(mask_2d, qk, float('-inf'))

        p = tl.exp(qk - l_i[:, None])

        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - Di[:, None]) * scale

        dv = tl.dot(tl.trans(p.to(q.dtype)), do, acc=dv)
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q, acc=dk)

    off_b_h_dk = pid_b * stride_dkb + pid_h * stride_dkh
    desc_dk = tl.make_tensor_descriptor(
        dK_ptr + off_b_h_dk, shape=[S, BLOCK_D], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, BLOCK_D]
    )
    off_b_h_dv = pid_b * stride_dvb + pid_h * stride_dvh
    desc_dv = tl.make_tensor_descriptor(
        dV_ptr + off_b_h_dv, shape=[S, BLOCK_D], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, BLOCK_D]
    )

    desc_dk.store([offset_n, 0], dk.to(dK_ptr.dtype.element_ty))
    desc_dv.store([offset_n, 0], dv.to(dV_ptr.dtype.element_ty))


def get_autotune_config_dq():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ]

# Use `restore_value` to prevent the autotuner from overwriting the safely-stashed Di state inside `dQ_ptr` during its multi-trial execution!
@triton.autotune(configs=get_autotune_config_dq(), key=['S'], restore_value=['dQ_ptr'])
@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offset_m = pid_m * BLOCK_M
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = offset_m + offs_m_base
    mask_m = offs_m < S
    
    off_b_h_q = pid_b * stride_qb + pid_h * stride_qh
    desc_q = tl.make_tensor_descriptor(
        Q_ptr + off_b_h_q, shape=[S, BLOCK_D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    off_b_h_do = pid_b * stride_dob + pid_h * stride_doh
    desc_do = tl.make_tensor_descriptor(
        dO_ptr + off_b_h_do, shape=[S, BLOCK_D], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )

    q = desc_q.load([offset_m, 0])
    do = desc_do.load([offset_m, 0])
    
    off_b_h_l = pid_b * stride_lb + pid_h * stride_lh
    l_i = tl.load(L_ptr + off_b_h_l + offs_m * stride_ls, mask=mask_m, other=0.0)
    
    off_b_h_dq = pid_b * stride_dqb + pid_h * stride_dqh
    row_ptrs = dQ_ptr + off_b_h_dq + offs_m * stride_dqs
    f_ptrs = tl.cast(row_ptrs, tl.pointer_type(tl.float32))
    Di = tl.load(f_ptrs, mask=mask_m, other=0.0)
    
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    off_b_h_k = pid_b * stride_kb + pid_h * stride_kh
    desc_k = tl.make_tensor_descriptor(
        K_ptr + off_b_h_k, shape=[S, BLOCK_D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    off_b_h_v = pid_b * stride_vb + pid_h * stride_vh
    desc_v = tl.make_tensor_descriptor(
        V_ptr + off_b_h_v, shape=[S, BLOCK_D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    
    offs_n_base = tl.arange(0, BLOCK_N)
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n_block in range(num_n_blocks):
        offset_n = n_block * BLOCK_N
        offs_n = offset_n + offs_n_base
        mask_n = offs_n < S
        
        k = desc_k.load([offset_n, 0])
        v = desc_v.load([offset_n, 0])
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk = qk * scale

        mask_2d = mask_m[:, None] & mask_n[None, :]
        qk = tl.where(mask_2d, qk, float('-inf'))
        
        p = tl.exp(qk - l_i[:, None])
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        ds = p * (dp - Di[:, None]) * scale
        
        dq = tl.dot(ds.to(q.dtype), k, acc=dq)

    # Finally properly overwrites stashed `Di` state leaving proper `dQ` gradients entirely
    desc_dq = tl.make_tensor_descriptor(
        dQ_ptr + off_b_h_dq, shape=[S, BLOCK_D], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, BLOCK_D]
    )
    desc_dq.store([offset_m, 0], dq.to(dQ_ptr.dtype.element_ty))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Triton TMA-based Multi-Head Attention backward kernel.
    Utilizes a preprocessing kernel to stash Di = rowsum(dO * O) in the first 4 bytes of dQ.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    lse = L.squeeze(-1) if L.dim() == 4 else L

    grid_pre = lambda META: (triton.cdiv(S, META['BLOCK_M']), H, B)
    bwd_preprocess_kernel[grid_pre](
        O, dO, dQ,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, BLOCK_D=128
    )

    grid_dkdv = lambda META: (triton.cdiv(S, META['BLOCK_N']), H, B)
    bwd_dkdv_kernel[grid_dkdv](
        Q, K, V, dO, lse, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, scale,
        BLOCK_D=128
    )

    grid_dq = lambda META: (triton.cdiv(S, META['BLOCK_M']), H, B)
    bwd_dq_kernel[grid_dq](
        Q, K, V, dO, lse, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, scale,
        BLOCK_D=128
    )

    return dQ, dK, dV