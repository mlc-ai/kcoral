import math
import torch
import triton
import triton.language as tl

# Configure standard infrastructure allocator for device-created TMA descriptors 
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.jit
def mha_bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    H, S, scale, LOG2_E,
    d: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    q_tile = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    
    batch_id = batch_head_id // H
    head_id = batch_head_id % H
    
    # Calculate exact base pointer limits resolving batch and head layouts
    Q += batch_id * stride_qb + head_id * stride_qh
    K += batch_id * stride_kb + head_id * stride_kh
    V += batch_id * stride_vb + head_id * stride_vh
    O += batch_id * stride_ob + head_id * stride_oh
    dO += batch_id * stride_dob + head_id * stride_doh
    L += batch_id * stride_lb + head_id * stride_lh
    dQ += batch_id * stride_dqb + head_id * stride_dqh
    
    # TMA logically loads safely bounding any padding explicitly mapping natively to 2D
    q_desc = tl.make_tensor_descriptor(Q, shape=[S, d], strides=[stride_qs, stride_qd], block_shape=[BLOCK_M, d], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(O, shape=[S, d], strides=[stride_os, stride_od], block_shape=[BLOCK_M, d], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(dO, shape=[S, d], strides=[stride_dos, stride_dod], block_shape=[BLOCK_M, d], padding_option="zero")
    
    k_desc = tl.make_tensor_descriptor(K, shape=[S, d], strides=[stride_ks, stride_kd], block_shape=[BLOCK_N, d], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(V, shape=[S, d], strides=[stride_vs, stride_vd], block_shape=[BLOCK_N, d], padding_option="zero")
    
    # Load exclusively resident iteration values cleanly via TMA
    q = q_desc.load([q_tile * BLOCK_M, 0])
    o = o_desc.load([q_tile * BLOCK_M, 0])
    do = do_desc.load([q_tile * BLOCK_M, 0])
    
    offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    # Conventional standard pointer used due to incompatible striding alignment typically exhibited by L 
    lse = tl.load(L + offs_m * stride_ls, mask=mask_m, other=0.0)
    
    # Linearly evaluate scalar query scale map mapped directly
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
    dq = tl.zeros((BLOCK_M, d), tl.float32)
    
    # Prevent completely arbitrary looping over non-contributing structural tiles
    max_kv_tile = tl.minimum(tl.cdiv(S, BLOCK_N), (q_tile * BLOCK_M + BLOCK_M + BLOCK_N - 1) // BLOCK_N)
    
    # Leverage explicit looping for seamless overlap mapping
    for kv_tile in tl.range(0, max_kv_tile, num_stages=3):
        k = k_desc.load([kv_tile * BLOCK_N, 0])
        v = v_desc.load([kv_tile * BLOCK_N, 0])
        
        offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        # Orient matrix multiplications logically allowing ideal native conversions
        scores = tl.dot(q, k.T) * scale
        
        valid_score = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        scores = tl.where(valid_score, scores, float("-inf"))
        
        p = tl.math.exp2((scores - lse[:, None]) * LOG2_E)
        dp = tl.dot(do, v.T)
        
        # Establish zero-safe masking preventing accumulation of explicit NaNs structurally out of bounds 
        ds = p * (dp - delta[:, None]) * scale
        ds = tl.where(valid_score, ds, 0.0)
        
        dq += tl.dot(ds.to(tl.bfloat16), k)
        
    offs_d = tl.arange(0, d)
    dq_ptrs = dQ + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def mha_bwd_dkdv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    H, S, scale, LOG2_E,
    d: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    kv_tile = tl.program_id(0)
    batch_head_id = tl.program_id(1)
    
    batch_id = batch_head_id // H
    head_id = batch_head_id % H
    
    Q += batch_id * stride_qb + head_id * stride_qh
    K += batch_id * stride_kb + head_id * stride_kh
    V += batch_id * stride_vb + head_id * stride_vh
    O += batch_id * stride_ob + head_id * stride_oh
    dO += batch_id * stride_dob + head_id * stride_doh
    L += batch_id * stride_lb + head_id * stride_lh
    dK += batch_id * stride_dkb + head_id * stride_dkh
    dV += batch_id * stride_dvb + head_id * stride_dvh
    
    k_desc = tl.make_tensor_descriptor(K, shape=[S, d], strides=[stride_ks, stride_kd], block_shape=[BLOCK_N, d], padding_option="zero")
    v_desc = tl.make_tensor_descriptor(V, shape=[S, d], strides=[stride_vs, stride_vd], block_shape=[BLOCK_N, d], padding_option="zero")
    
    q_desc = tl.make_tensor_descriptor(Q, shape=[S, d], strides=[stride_qs, stride_qd], block_shape=[BLOCK_M, d], padding_option="zero")
    o_desc = tl.make_tensor_descriptor(O, shape=[S, d], strides=[stride_os, stride_od], block_shape=[BLOCK_M, d], padding_option="zero")
    do_desc = tl.make_tensor_descriptor(dO, shape=[S, d], strides=[stride_dos, stride_dod], block_shape=[BLOCK_M, d], padding_option="zero")
    
    k = k_desc.load([kv_tile * BLOCK_N, 0])
    v = v_desc.load([kv_tile * BLOCK_N, 0])
    
    dk = tl.zeros((BLOCK_N, d), tl.float32)
    dv = tl.zeros((BLOCK_N, d), tl.float32)
    
    offs_n = kv_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S
    
    # Safely transpose implicitly starting causality evaluations cleanly limits structural loop skips
    min_q_tile = (kv_tile * BLOCK_N) // BLOCK_M
    max_q_tile = tl.cdiv(S, BLOCK_M)
    
    for q_tile in tl.range(min_q_tile, max_q_tile, num_stages=3):
        q = q_desc.load([q_tile * BLOCK_M, 0])
        o = o_desc.load([q_tile * BLOCK_M, 0])
        do = do_desc.load([q_tile * BLOCK_M, 0])
        
        offs_m = q_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S
        
        lse = tl.load(L + offs_m * stride_ls, mask=mask_m, other=0.0)
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        scores_t = tl.dot(k, q.T) * scale
        
        valid_score_t = (offs_m[None, :] >= offs_n[:, None]) & mask_m[None, :] & mask_n[:, None]
        scores_t = tl.where(valid_score_t, scores_t, float("-inf"))
        
        p_t = tl.math.exp2((scores_t - lse[None, :]) * LOG2_E)
        
        dv += tl.dot(p_t.to(tl.bfloat16), do)
        
        dp_t = tl.dot(v, do.T)
        ds_t = p_t * (dp_t - delta[None, :]) * scale
        ds_t = tl.where(valid_score_t, ds_t, 0.0)
        
        dk += tl.dot(ds_t.to(tl.bfloat16), q)
        
    offs_d = tl.arange(0, d)
    dk_ptrs = dK + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(tl.bfloat16), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    # Asymmetrically bounded to stay safely well beneath the 227 KiB limit (yields ~192-208 KiB footprints conservatively)
    BLOCK_M_DQ = 128
    BLOCK_N_DQ = 64
    
    BLOCK_N_DKDV = 128
    BLOCK_M_DKDV = 64
    
    num_q_tiles = triton.cdiv(S, BLOCK_M_DQ)
    num_kv_tiles = triton.cdiv(S, BLOCK_N_DKDV)
    
    scale = 1.0 / math.sqrt(d)
    LOG2_E = 1.4426950408889634
    
    if num_q_tiles > 0:
        grid_dq = (num_q_tiles, B * H)
        mha_bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, L, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            H, S, scale, LOG2_E,
            d=d, BLOCK_M=BLOCK_M_DQ, BLOCK_N=BLOCK_N_DQ,
            num_warps=8, num_stages=3
        )
        
        grid_dkdv = (num_kv_tiles, B * H)
        mha_bwd_dkdv_kernel[grid_dkdv](
            Q, K, V, O, dO, L, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            H, S, scale, LOG2_E,
            d=d, BLOCK_N=BLOCK_N_DKDV, BLOCK_M=BLOCK_M_DKDV,
            num_warps=8, num_stages=3
        )
        
    return dQ, dK, dV