import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    ],
    key=["S"],
)
@triton.jit
def bwd_kernel(
    Q, K, V, O, dO, L, dQ, dK, dV,
    H, S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    batch_idx = pid_bh // H
    head_idx = pid_bh % H
    
    num_kv_tiles = tl.cdiv(S, BLOCK_N)
    num_q_tiles = tl.cdiv(S, BLOCK_M)
    
    is_kv_owner = pid_s < num_kv_tiles
    
    if is_kv_owner:
        kv_idx = pid_s
        kv_start = kv_idx * BLOCK_N
        offs_n = kv_start + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        mask_n = offs_n < S
        
        k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        
        q_min = kv_start // BLOCK_M
        q_max = num_q_tiles
        
        for q_idx in range(q_min, q_max):
            q_start = q_idx * BLOCK_M
            offs_m = q_start + tl.arange(0, BLOCK_M)
            mask_m = offs_m < S
            
            q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
            do_ptrs = dO + batch_idx * stride_dob + head_idx * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
            o_ptrs = O + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
            lse_ptrs = L + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_ls
            
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
            o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
            lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
            
            delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
            
            scores_t = tl.dot(k, q.T) * scale
            
            causal_mask_t = offs_n[:, None] <= offs_m[None, :]
            valid_mask_t = causal_mask_t & mask_n[:, None] & mask_m[None, :]
            
            scores_t = tl.where(valid_mask_t, scores_t, float('-inf'))
            p_t = tl.exp(scores_t - lse[None, :])
            
            dv += tl.dot(p_t.to(q.dtype), do)
            
            dp_t = tl.dot(v, do.T)
            ds_t = p_t * (dp_t - delta[None, :]) * scale
            ds_t = tl.where(valid_mask_t, ds_t, 0.0)
            
            dk += tl.dot(ds_t.to(q.dtype), q)
            
        dk_ptrs = dK + batch_idx * stride_dkb + head_idx * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dv_ptrs = dV + batch_idx * stride_dvb + head_idx * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        
        tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
        tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_n[:, None])
        
    else:
        q_idx = pid_s - num_kv_tiles
        q_start = q_idx * BLOCK_M
        offs_m = q_start + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < S
        
        q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        do_ptrs = dO + batch_idx * stride_dob + head_idx * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        o_ptrs = O + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        lse_ptrs = L + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_ls
        
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        
        dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        
        kv_max = tl.cdiv(tl.minimum(S, q_start + BLOCK_M), BLOCK_N)
        
        for kv_idx in range(0, kv_max):
            kv_start = kv_idx * BLOCK_N
            offs_n = kv_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S
            
            k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
            
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            
            scores = tl.dot(q, k.T) * scale
            
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            valid_mask = causal_mask & mask_m[:, None] & mask_n[None, :]
            
            scores = tl.where(valid_mask, scores, float('-inf'))
            p = tl.exp(scores - lse[:, None])
            
            dp = tl.dot(do, v.T)
            
            ds = p * (dp - delta[:, None]) * scale
            ds = tl.where(valid_mask, ds, 0.0)
            
            dq += tl.dot(ds.to(q.dtype), k)
            
        dq_ptrs = dQ + batch_idx * stride_dqb + head_idx * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
        tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the causal multi-head attention backward pass.
    Writes the gradients to the preallocated dQ, dK, and dV tensors.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    grid = lambda META: (
        B * H,
        triton.cdiv(S, META['BLOCK_N']) + triton.cdiv(S, META['BLOCK_M'])
    )
    
    bwd_kernel[grid](
        Q, K, V, O, dO, L, dQ, dK, dV,
        H, S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        scale,
        BLOCK_D=d,
    )