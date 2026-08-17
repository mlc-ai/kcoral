import torch
import triton
import triton.language as tl

# Infrastructure allocation requirement mandated by standard Triton for dynamically building device-side TMA descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # SMEM calculations verified <= 227KB limit
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    S, scale, LOG2E,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // 48
    h = pid_bh % 48

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    Q_base = Q + b * stride_qb + h * stride_qh
    dO_base = dO + b * stride_dob + h * stride_doh
    O_base = O + b * stride_ob + h * stride_oh
    
    q_desc = tl.make_tensor_descriptor(
        Q_base, shape=[S, HEAD_DIM], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO_base, shape=[S, HEAD_DIM], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O_base, shape=[S, HEAD_DIM], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
    )
    
    q = q_desc.load([pid_m * BLOCK_M, 0])
    do = do_desc.load([pid_m * BLOCK_M, 0])
    o = o_desc.load([pid_m * BLOCK_M, 0])
    
    safe_offs_m = tl.where(offs_m < S, offs_m, 0)
    l_ptrs = L + b * stride_lb + h * stride_lh + safe_offs_m * stride_ls
    l_val = tl.load(l_ptrs, mask=(offs_m < S), other=0.0)
    
    # Scale Q and dO globally once, completely eradicating inner-loop tile multiplications 
    q_scaled = (q.to(tl.float32) * (scale * LOG2E)).to(q.dtype)
    do_scaled = (do.to(tl.float32) * scale).to(do.dtype)
    l_val_scaled = l_val * LOG2E
    
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    delta_scaled = delta * scale
    
    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    
    K_base = K + b * stride_kb + h * stride_kh
    V_base = V + b * stride_vb + h * stride_vh
    
    k_desc = tl.make_tensor_descriptor(
        K_base, shape=[S, HEAD_DIM], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_base, shape=[S, HEAD_DIM], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
    )
    
    num_n = tl.cdiv(S, BLOCK_N)
    for n0 in tl.range(0, num_n):
        k = k_desc.load([n0 * BLOCK_N, 0])
        v = v_desc.load([n0 * BLOCK_N, 0])
        
        offs_n = n0 * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # Scaling baked-in optimally via inputs
        scores = tl.dot(q_scaled, k.T, out_dtype=tl.float32)
        
        mask_m_val = offs_m < S
        mask_n = offs_n < S
        scores = tl.where(mask_m_val[:, None] & mask_n[None, :], scores, float("-inf"))
        
        p = tl.math.exp2(scores - l_val_scaled[:, None])
        
        dp_scaled = tl.dot(do_scaled, v.T, out_dtype=tl.float32)
        ds = p * (dp_scaled - delta_scaled[:, None])
        
        dq = tl.dot(ds.to(q.dtype), k, acc=dq)
        
    dQ_base = dQ + b * stride_dqb + h * stride_dqh
    dq_desc = tl.make_tensor_descriptor(
        dQ_base, shape=[S, HEAD_DIM], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
    )
    dq_desc.store([pid_m * BLOCK_M, 0], dq.to(dQ.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_M": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_M": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_M": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def bwd_dkdv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    S, scale, LOG2E,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // 48
    h = pid_bh % 48

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    K_base = K + b * stride_kb + h * stride_kh
    V_base = V + b * stride_vb + h * stride_vh
    
    k_desc = tl.make_tensor_descriptor(
        K_base, shape=[S, HEAD_DIM], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_base, shape=[S, HEAD_DIM], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
    )
    
    k = k_desc.load([pid_n * BLOCK_N, 0])
    v = v_desc.load([pid_n * BLOCK_N, 0])
    
    # Scale K and V globally once  
    k_scaled = (k.to(tl.float32) * (scale * LOG2E)).to(k.dtype)
    v_scaled = (v.to(tl.float32) * scale).to(v.dtype)
    
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    
    Q_base = Q + b * stride_qb + h * stride_qh
    dO_base = dO + b * stride_dob + h * stride_doh
    O_base = O + b * stride_ob + h * stride_oh
    
    q_desc = tl.make_tensor_descriptor(
        Q_base, shape=[S, HEAD_DIM], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO_base, shape=[S, HEAD_DIM], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O_base, shape=[S, HEAD_DIM], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, HEAD_DIM], padding_option="zero"
    )
    
    num_m = tl.cdiv(S, BLOCK_M)
    for m0 in tl.range(0, num_m):
        q = q_desc.load([m0 * BLOCK_M, 0])
        do = do_desc.load([m0 * BLOCK_M, 0])
        o = o_desc.load([m0 * BLOCK_M, 0])
        
        offs_m = m0 * BLOCK_M + tl.arange(0, BLOCK_M)
        safe_offs_m = tl.where(offs_m < S, offs_m, 0)
        l_ptrs = L + b * stride_lb + h * stride_lh + safe_offs_m * stride_ls
        l_val = tl.load(l_ptrs, mask=(offs_m < S), other=0.0)
        
        l_val_scaled = l_val * LOG2E
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        delta_scaled = delta * scale
        
        scores_t = tl.dot(k_scaled, q.T, out_dtype=tl.float32)
        
        mask_m_val = offs_m < S
        scores_t = tl.where(mask_n[:, None] & mask_m_val[None, :], scores_t, float("-inf"))
        
        p_t = tl.math.exp2(scores_t - l_val_scaled[None, :])
        
        dv = tl.dot(p_t.to(q.dtype), do, acc=dv)
        
        dp_t_scaled = tl.dot(v_scaled, do.T, out_dtype=tl.float32)
        ds_t = p_t * (dp_t_scaled - delta_scaled[None, :])
        
        dk = tl.dot(ds_t.to(q.dtype), q, acc=dk)
        
    dK_base = dK + b * stride_dkb + h * stride_dkh
    dV_base = dV + b * stride_dvb + h * stride_dvh
    
    dk_desc = tl.make_tensor_descriptor(
        dK_base, shape=[S, HEAD_DIM], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
    )
    dv_desc = tl.make_tensor_descriptor(
        dV_base, shape=[S, HEAD_DIM], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, HEAD_DIM], padding_option="zero"
    )
    
    dk_desc.store([pid_n * BLOCK_N, 0], dk.to(dK.dtype.element_ty))
    dv_desc.store([pid_n * BLOCK_N, 0], dv.to(dV.dtype.element_ty))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Standard Triton attention backward avoiding global gradient atomics via Split Grid Launches.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    LOG2E = 1.4426950408889634
    
    # Exclusively assigned Query region
    grid_dq = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, scale, LOG2E, HEAD_DIM=d
    )
    
    # Exclusively assigned KV Region
    grid_dkdv = lambda META: (triton.cdiv(S, META["BLOCK_N"]), B * H)
    bwd_dkdv_kernel[grid_dkdv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, scale, LOG2E, HEAD_DIM=d
    )