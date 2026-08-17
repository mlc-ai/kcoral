import math
import torch
import triton
import triton.language as tl

# Standard Triton descriptor allocation hook for device-created TMA bounds
def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
    ],
    key=['S']
)
@triton.jit
def _bwd_dq_kernel(
    Q, K, V, O, dO, LSE, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lseb, stride_lseh, stride_lses,
    S, softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    q_tile = tl.program_id(2)
    
    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    
    # Base offsets for this batch and head
    off_q = pid_b * stride_qb + pid_h * stride_qh
    off_k = pid_b * stride_kb + pid_h * stride_kh
    off_v = pid_b * stride_vb + pid_h * stride_vh
    off_o = pid_b * stride_ob + pid_h * stride_oh
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_dq = pid_b * stride_dqb + pid_h * stride_dqh
    off_lse = pid_b * stride_lseb + pid_h * stride_lseh
    
    # Device descriptors intrinsically support bounds padding safely mapping to Blackwell TMA directives
    q_desc = tl.make_tensor_descriptor(
        Q + off_q, shape=[S, d], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + off_o, shape=[S, d], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + off_do, shape=[S, d], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    dq_desc = tl.make_tensor_descriptor(
        dQ + off_dq, shape=[S, d], strides=[stride_dqs, stride_dqd],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    
    # Resident Query-dimension tensors
    q = tl.load(q_desc, [q_tile * BLOCK_M, 0])
    o = tl.load(o_desc, [q_tile * BLOCK_M, 0])
    do = tl.load(do_desc, [q_tile * BLOCK_M, 0])
    
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    lse = tl.load(LSE + off_lse + offs_m * stride_lses, mask=mask_m, other=0.0)
    
    dq = tl.zeros((BLOCK_M, d), tl.float32)
    
    k_desc = tl.make_tensor_descriptor(
        K + off_k, shape=[S, d], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + off_v, shape=[S, d], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    
    for kv_tile in range(num_kv_tiles):
        k = tl.load(k_desc, [kv_tile * BLOCK_N, 0])
        v = tl.load(v_desc, [kv_tile * BLOCK_N, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * softmax_scale
        p = tl.math.exp(qk - lse[:, None])
        
        offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_mn = mask_m[:, None] & (offs_n[None, :] < S)
        
        p = tl.where(mask_mn, p, 0.0)
        
        dp = tl.dot(do, v.T, out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * softmax_scale
        
        dq += tl.dot(ds.to(q.dtype), k, out_dtype=tl.float32)
        
    tl.store(dq_desc, [q_tile * BLOCK_M, 0], dq.to(dQ.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}, num_warps=4, num_stages=3),
    ],
    key=['S']
)
@triton.jit
def _bwd_dkdv_kernel(
    Q, K, V, O, dO, LSE, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lseb, stride_lseh, stride_lses,
    S, softmax_scale,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, d: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_tile = tl.program_id(2)
    
    num_q_tiles = tl.cdiv(S, BLOCK_M)
    
    # Base offsets for this batch and head
    off_q = pid_b * stride_qb + pid_h * stride_qh
    off_k = pid_b * stride_kb + pid_h * stride_kh
    off_v = pid_b * stride_vb + pid_h * stride_vh
    off_o = pid_b * stride_ob + pid_h * stride_oh
    off_do = pid_b * stride_dob + pid_h * stride_doh
    off_dk = pid_b * stride_dkb + pid_h * stride_dkh
    off_dv = pid_b * stride_dvb + pid_h * stride_dvh
    off_lse = pid_b * stride_lseb + pid_h * stride_lseh
    
    k_desc = tl.make_tensor_descriptor(
        K + off_k, shape=[S, d], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + off_v, shape=[S, d], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dk_desc = tl.make_tensor_descriptor(
        dK + off_dk, shape=[S, d], strides=[stride_dks, stride_dkd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + off_dv, shape=[S, d], strides=[stride_dvs, stride_dvd],
        block_shape=[BLOCK_N, d], padding_option="zero"
    )
    
    # Resident KV-dimension tensors
    k = tl.load(k_desc, [kv_tile * BLOCK_N, 0])
    v = tl.load(v_desc, [kv_tile * BLOCK_N, 0])
    
    dk = tl.zeros((BLOCK_N, d), tl.float32)
    dv = tl.zeros((BLOCK_N, d), tl.float32)
    
    offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    q_desc = tl.make_tensor_descriptor(
        Q + off_q, shape=[S, d], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + off_o, shape=[S, d], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    do_desc = tl.make_tensor_descriptor(
        dO + off_do, shape=[S, d], strides=[stride_dos, stride_dod],
        block_shape=[BLOCK_M, d], padding_option="zero"
    )
    
    for q_tile in range(num_q_tiles):
        q = tl.load(q_desc, [q_tile * BLOCK_M, 0])
        o = tl.load(o_desc, [q_tile * BLOCK_M, 0])
        do = tl.load(do_desc, [q_tile * BLOCK_M, 0])
        
        offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        lse = tl.load(LSE + off_lse + offs_m * stride_lses, mask=mask_m, other=0.0)
        
        qk_t = tl.dot(k, q.T, out_dtype=tl.float32) * softmax_scale
        p_t = tl.math.exp(qk_t - lse[None, :])
        
        mask_nm = mask_n[:, None] & mask_m[None, :]
        p_t = tl.where(mask_nm, p_t, 0.0)
        
        dv += tl.dot(p_t.to(q.dtype), do, out_dtype=tl.float32)
        
        dp_t = tl.dot(v, do.T, out_dtype=tl.float32)
        ds_t = p_t * (dp_t - delta[None, :]) * softmax_scale
        
        dk += tl.dot(ds_t.to(q.dtype), q, out_dtype=tl.float32)
        
    tl.store(dk_desc, [kv_tile * BLOCK_N, 0], dk.to(dK.dtype.element_ty))
    tl.store(dv_desc, [kv_tile * BLOCK_N, 0], dv.to(dV.dtype.element_ty))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes Standard-Triton B200 TMA Attention Backwards returning exclusively to target instances natively avoiding gradient overlaps.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    softmax_scale = 1.0 / math.sqrt(d)
    
    # Kernel 1: Query dimension Ownership mapped
    def grid_dq(META):
        return (B, H, triton.cdiv(S, META['BLOCK_M']))
        
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, softmax_scale, d=d
    )
    
    # Kernel 2: Key-Value dimension Ownership mapped
    def grid_dkdv(META):
        return (B, H, triton.cdiv(S, META['BLOCK_N']))
        
    _bwd_dkdv_kernel[grid_dkdv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        S, softmax_scale, d=d
    )