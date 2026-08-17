import math
import torch
import triton
import triton.language as tl

# Configure Triton's device-side descriptor allocator for Blackwell TMA
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["seq_len"]
)
@triton.jit
def _bwd_dq_kernel(
    q, k, v, o, do, lse, dq,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    seq_len, head_dim, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    start_m = pid_m * BLOCK_M
    
    q_base = q + pid_b * stride_qb + pid_h * stride_qh
    k_base = k + pid_b * stride_kb + pid_h * stride_kh
    v_base = v + pid_b * stride_vb + pid_h * stride_vh
    o_base = o + pid_b * stride_ob + pid_h * stride_oh
    do_base = do + pid_b * stride_dob + pid_h * stride_doh
    dq_base = dq + pid_b * stride_dqb + pid_h * stride_dqh
    
    # TMA Descriptors for highly optimal pipelined HBM reads/writes
    desc_q = tl.make_tensor_descriptor(
        q_base, shape=[seq_len, head_dim], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    desc_o = tl.make_tensor_descriptor(
        o_base, shape=[seq_len, head_dim], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    desc_do = tl.make_tensor_descriptor(
        do_base, shape=[seq_len, head_dim], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    desc_k = tl.make_tensor_descriptor(
        k_base, shape=[seq_len, head_dim], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    desc_v = tl.make_tensor_descriptor(
        v_base, shape=[seq_len, head_dim], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    desc_dq = tl.make_tensor_descriptor(
        dq_base, shape=[seq_len, head_dim], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    
    q_tile = tl.load(desc_q, [start_m, 0])
    o_tile = tl.load(desc_o, [start_m, 0])
    do_tile = tl.load(desc_do, [start_m, 0])
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    
    # Standard pointer load for LSE as trailing stride is non-contiguous
    lse_ptrs = lse + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
    lse_tile = tl.load(lse_ptrs, mask=mask_m, other=0.0)
    
    delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
    dq_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    
    max_n = tl.minimum(start_m + BLOCK_M, seq_len)
    num_steps = tl.cdiv(max_n, BLOCK_N)
    
    for step in range(num_steps):
        start_n = step * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = causal_mask & mask_m[:, None] & mask_n[None, :]
        
        k_tile = tl.load(desc_k, [start_n, 0])
        v_tile = tl.load(desc_v, [start_n, 0])
        
        scores_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        scores = tl.dot(q_tile, k_tile.T, scores_acc) * sm_scale
        scores = tl.where(valid_mask, scores, float('-inf'))
        
        p = tl.math.exp(scores - lse_tile[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        dp_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(do_tile, v_tile.T, dp_acc)
        
        ds = p * (dp - delta[:, None]) * sm_scale
        
        ds_cast = ds.to(q_tile.dtype)
        dq_acc = tl.dot(ds_cast, k_tile, dq_acc)
        
    tl.store(desc_dq, [start_m, 0], dq_acc.to(q_tile.dtype))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["seq_len"]
)
@triton.jit
def _bwd_dkv_kernel(
    q, k, v, o, do, lse, dk, dv,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    seq_len, head_dim, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    start_n = pid_n * BLOCK_N
    
    q_base = q + pid_b * stride_qb + pid_h * stride_qh
    k_base = k + pid_b * stride_kb + pid_h * stride_kh
    v_base = v + pid_b * stride_vb + pid_h * stride_vh
    o_base = o + pid_b * stride_ob + pid_h * stride_oh
    do_base = do + pid_b * stride_dob + pid_h * stride_doh
    dk_base = dk + pid_b * stride_dkb + pid_h * stride_dkh
    dv_base = dv + pid_b * stride_dvb + pid_h * stride_dvh
    
    desc_q = tl.make_tensor_descriptor(
        q_base, shape=[seq_len, head_dim], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    desc_o = tl.make_tensor_descriptor(
        o_base, shape=[seq_len, head_dim], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    desc_do = tl.make_tensor_descriptor(
        do_base, shape=[seq_len, head_dim], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    desc_k = tl.make_tensor_descriptor(
        k_base, shape=[seq_len, head_dim], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    desc_v = tl.make_tensor_descriptor(
        v_base, shape=[seq_len, head_dim], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    desc_dk = tl.make_tensor_descriptor(
        dk_base, shape=[seq_len, head_dim], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    desc_dv = tl.make_tensor_descriptor(
        dv_base, shape=[seq_len, head_dim], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    
    k_tile = tl.load(desc_k, [start_n, 0])
    v_tile = tl.load(desc_v, [start_n, 0])
    
    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seq_len
    
    start_m_initial = (start_n // BLOCK_M) * BLOCK_M
    num_steps = tl.cdiv(seq_len - start_m_initial, BLOCK_M)
    
    for step in range(num_steps):
        start_m = start_m_initial + step * BLOCK_M
        
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = causal_mask & mask_m[:, None] & mask_n[None, :]
        
        q_tile = tl.load(desc_q, [start_m, 0])
        o_tile = tl.load(desc_o, [start_m, 0])
        do_tile = tl.load(desc_do, [start_m, 0])
        
        lse_ptrs = lse + pid_b * stride_lb + pid_h * stride_lh + offs_m * stride_ls
        lse_tile = tl.load(lse_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
        
        scores_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        scores = tl.dot(q_tile, k_tile.T, scores_acc) * sm_scale
        scores = tl.where(valid_mask, scores, float('-inf'))
        
        p = tl.math.exp(scores - lse_tile[:, None])
        p = tl.where(valid_mask, p, 0.0)
        
        dp_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(do_tile, v_tile.T, dp_acc)
        
        ds = p * (dp - delta[:, None]) * sm_scale
        
        ds_cast = ds.to(k_tile.dtype)
        p_cast = p.to(v_tile.dtype)
        
        dk_acc = tl.dot(ds_cast.T, q_tile, dk_acc)
        dv_acc = tl.dot(p_cast.T, do_tile, dv_acc)
        
    tl.store(desc_dk, [start_n, 0], dk_acc.to(k_tile.dtype))
    tl.store(desc_dv, [start_n, 0], dv_acc.to(v_tile.dtype))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes causal attention backward pass safely using split-ownership over query tiles.
    Fully uncoupled from PyTorch allocation, utilizes optimized SM100 TMA execution paths.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d_dim = Q.shape
    stride_lb = L.stride(0)
    stride_lh = L.stride(1)
    stride_ls = L.stride(2)
        
    sm_scale = 1.0 / math.sqrt(d_dim)
    
    def grid_dq(META):
        return (triton.cdiv(S, META['BLOCK_M']), H, B)
        
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        stride_lb, stride_lh, stride_ls,
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        S, d_dim, sm_scale,
        BLOCK_D=128
    )

    def grid_dkv(META):
        return (triton.cdiv(S, META['BLOCK_N']), H, B)

    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        stride_lb, stride_lh, stride_ls,
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        S, d_dim, sm_scale,
        BLOCK_D=128
    )

    return dQ, dK, dV