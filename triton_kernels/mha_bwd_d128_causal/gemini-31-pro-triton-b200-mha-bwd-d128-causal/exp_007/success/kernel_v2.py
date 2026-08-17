import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M_DQ": 128, "BLOCK_N_DQ": 64, "BLOCK_M_DKDV": 64, "BLOCK_N_DKDV": 128, "PIPELINE_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M_DQ": 128, "BLOCK_N_DQ": 64, "BLOCK_M_DKDV": 64, "BLOCK_N_DKDV": 128, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M_DQ": 128, "BLOCK_N_DQ": 128, "BLOCK_M_DKDV": 128, "BLOCK_N_DKDV": 128, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M_DQ": 64, "BLOCK_N_DQ": 64, "BLOCK_M_DKDV": 64, "BLOCK_N_DKDV": 64, "PIPELINE_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M_DQ": 64, "BLOCK_N_DQ": 64, "BLOCK_M_DKDV": 64, "BLOCK_N_DKDV": 64, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M_DQ": 128, "BLOCK_N_DQ": 64, "BLOCK_M_DKDV": 64, "BLOCK_N_DKDV": 128, "PIPELINE_STAGES": 2}, num_warps=4, num_stages=2),
    ],
    key=["S"],
)
@triton.jit
def bwd_kernel(
    Q, K, V, O, dO, L, dQ, dK, dV,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_lb, stride_lh, stride_ls,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    scale, S, H, DIVISIBLE_S: tl.constexpr,
    BLOCK_M_DQ: tl.constexpr, BLOCK_N_DQ: tl.constexpr, 
    BLOCK_M_DKDV: tl.constexpr, BLOCK_N_DKDV: tl.constexpr, 
    BLOCK_D: tl.constexpr, PIPELINE_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch_idx = pid_bh // H
    head_idx = pid_bh % H
    
    num_kv_tiles = tl.cdiv(S, BLOCK_N_DKDV)
    num_q_tiles = tl.cdiv(S, BLOCK_M_DQ)
    
    if pid < num_kv_tiles:
        # -------------------------------------------------------------
        # dK / dV owner region
        # -------------------------------------------------------------
        kv_idx = pid
        kv_start = kv_idx * BLOCK_N_DKDV
        offs_n = kv_start + tl.arange(0, BLOCK_N_DKDV)
        offs_d = tl.arange(0, BLOCK_D)
        
        k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        if DIVISIBLE_S:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            mask_n = offs_n < S
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        dk = tl.zeros((BLOCK_N_DKDV, BLOCK_D), tl.float32)
        dv = tl.zeros((BLOCK_N_DKDV, BLOCK_D), tl.float32)
        
        q_start_idx = kv_start // BLOCK_M_DKDV
        q_dense_start_idx = tl.minimum(tl.cdiv(S, BLOCK_M_DKDV), (kv_start + BLOCK_N_DKDV + BLOCK_M_DKDV - 1) // BLOCK_M_DKDV)
        num_q = tl.cdiv(S, BLOCK_M_DKDV)
        
        q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + tl.arange(0, BLOCK_M_DKDV)[:, None] * stride_qs + offs_d[None, :] * stride_qd
        do_ptrs = dO + batch_idx * stride_dob + head_idx * stride_doh + tl.arange(0, BLOCK_M_DKDV)[:, None] * stride_dos + offs_d[None, :] * stride_dod
        o_ptrs = O + batch_idx * stride_ob + head_idx * stride_oh + tl.arange(0, BLOCK_M_DKDV)[:, None] * stride_os + offs_d[None, :] * stride_od
        lse_ptrs = L + batch_idx * stride_lb + head_idx * stride_lh + tl.arange(0, BLOCK_M_DKDV) * stride_ls
        
        q_ptrs += q_start_idx * BLOCK_M_DKDV * stride_qs
        do_ptrs += q_start_idx * BLOCK_M_DKDV * stride_dos
        o_ptrs += q_start_idx * BLOCK_M_DKDV * stride_os
        lse_ptrs += q_start_idx * BLOCK_M_DKDV * stride_ls
        
        # Masked Q loop (causal boundary intersection)
        for q_idx in tl.range(q_start_idx, q_dense_start_idx, num_stages=PIPELINE_STAGES):
            q_start = q_idx * BLOCK_M_DKDV
            offs_m = q_start + tl.arange(0, BLOCK_M_DKDV)
            
            if DIVISIBLE_S:
                q = tl.load(q_ptrs)
                do = tl.load(do_ptrs)
                o = tl.load(o_ptrs)
                lse = tl.load(lse_ptrs)
                
                delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
                scores_t = tl.dot(k, q.T) * scale
                
                valid_mask_t = offs_n[:, None] <= offs_m[None, :]
                scores_t = tl.where(valid_mask_t, scores_t, float('-inf'))
                p_t = tl.exp(scores_t - lse[None, :])
                p_t = tl.where(valid_mask_t, p_t, 0.0)
                
                dv += tl.dot(p_t.to(q.dtype), do)
                dp_t = tl.dot(v, do.T)
                ds_t = p_t * (dp_t - delta[None, :]) * scale
                ds_t = tl.where(valid_mask_t, ds_t, 0.0)
                
                dk += tl.dot(ds_t.to(q.dtype), q)
            else:
                mask_m = offs_m < S
                q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
                do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
                o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
                lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
                
                delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
                scores_t = tl.dot(k, q.T) * scale
                
                valid_mask_t = (offs_n[:, None] <= offs_m[None, :]) & mask_n[:, None] & mask_m[None, :]
                scores_t = tl.where(valid_mask_t, scores_t, float('-inf'))
                p_t = tl.exp(scores_t - lse[None, :])
                p_t = tl.where(valid_mask_t, p_t, 0.0)
                
                dv += tl.dot(p_t.to(q.dtype), do)
                dp_t = tl.dot(v, do.T)
                ds_t = p_t * (dp_t - delta[None, :]) * scale
                ds_t = tl.where(valid_mask_t, ds_t, 0.0)
                
                dk += tl.dot(ds_t.to(q.dtype), q)
                
            q_ptrs += BLOCK_M_DKDV * stride_qs
            do_ptrs += BLOCK_M_DKDV * stride_dos
            o_ptrs += BLOCK_M_DKDV * stride_os
            lse_ptrs += BLOCK_M_DKDV * stride_ls

        # Dense Q loop (fully causal safe)
        for q_idx in tl.range(q_dense_start_idx, num_q, num_stages=PIPELINE_STAGES):
            if DIVISIBLE_S:
                q = tl.load(q_ptrs)
                do = tl.load(do_ptrs)
                o = tl.load(o_ptrs)
                lse = tl.load(lse_ptrs)
                
                delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
                scores_t = tl.dot(k, q.T) * scale
                
                p_t = tl.exp(scores_t - lse[None, :])
                
                dv += tl.dot(p_t.to(q.dtype), do)
                dp_t = tl.dot(v, do.T)
                ds_t = p_t * (dp_t - delta[None, :]) * scale
                
                dk += tl.dot(ds_t.to(q.dtype), q)
            else:
                q_start = q_idx * BLOCK_M_DKDV
                offs_m = q_start + tl.arange(0, BLOCK_M_DKDV)
                mask_m = offs_m < S
                
                q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
                do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
                o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
                lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
                
                delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
                scores_t = tl.dot(k, q.T) * scale
                
                valid_mask_t = mask_n[:, None] & mask_m[None, :]
                scores_t = tl.where(valid_mask_t, scores_t, float('-inf'))
                p_t = tl.exp(scores_t - lse[None, :])
                p_t = tl.where(valid_mask_t, p_t, 0.0)
                
                dv += tl.dot(p_t.to(q.dtype), do)
                dp_t = tl.dot(v, do.T)
                ds_t = p_t * (dp_t - delta[None, :]) * scale
                ds_t = tl.where(valid_mask_t, ds_t, 0.0)
                
                dk += tl.dot(ds_t.to(q.dtype), q)
                
            q_ptrs += BLOCK_M_DKDV * stride_qs
            do_ptrs += BLOCK_M_DKDV * stride_dos
            o_ptrs += BLOCK_M_DKDV * stride_os
            lse_ptrs += BLOCK_M_DKDV * stride_ls

        dk_ptrs = dK + batch_idx * stride_dkb + head_idx * stride_dkh + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
        dv_ptrs = dV + batch_idx * stride_dvb + head_idx * stride_dvh + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
        
        if DIVISIBLE_S:
            tl.store(dk_ptrs, dk.to(k.dtype))
            tl.store(dv_ptrs, dv.to(v.dtype))
        else:
            tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
            tl.store(dv_ptrs, dv.to(v.dtype), mask=mask_n[:, None])

    elif pid < num_kv_tiles + num_q_tiles:
        # -------------------------------------------------------------
        # dQ owner region
        # -------------------------------------------------------------
        q_idx = pid - num_kv_tiles
        q_start = q_idx * BLOCK_M_DQ
        offs_m = q_start + tl.arange(0, BLOCK_M_DQ)
        offs_d = tl.arange(0, BLOCK_D)
        
        q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        do_ptrs = dO + batch_idx * stride_dob + head_idx * stride_doh + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod
        o_ptrs = O + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        lse_ptrs = L + batch_idx * stride_lb + head_idx * stride_lh + offs_m * stride_ls
        
        if DIVISIBLE_S:
            q = tl.load(q_ptrs)
            do = tl.load(do_ptrs)
            o = tl.load(o_ptrs)
            lse = tl.load(lse_ptrs)
        else:
            mask_m = offs_m < S
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
            o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
            lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1)
        dq = tl.zeros((BLOCK_M_DQ, BLOCK_D), tl.float32)
        
        k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + tl.arange(0, BLOCK_N_DQ)[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + tl.arange(0, BLOCK_N_DQ)[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        kv_dense_iters = q_start // BLOCK_N_DQ
        kv_max_iters = tl.cdiv(tl.minimum(S, q_start + BLOCK_M_DQ), BLOCK_N_DQ)
        
        # Dense KV loop (fully causal safe)
        for kv_idx in tl.range(0, kv_dense_iters, num_stages=PIPELINE_STAGES):
            if DIVISIBLE_S:
                k = tl.load(k_ptrs)
                v = tl.load(v_ptrs)
                
                scores = tl.dot(q, k.T) * scale
                p = tl.exp(scores - lse[:, None])
                
                dp = tl.dot(do, v.T)
                ds = p * (dp - delta[:, None]) * scale
                
                dq += tl.dot(ds.to(q.dtype), k)
            else:
                offs_n = kv_idx * BLOCK_N_DQ + tl.arange(0, BLOCK_N_DQ)
                mask_n = offs_n < S
                
                k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
                
                scores = tl.dot(q, k.T) * scale
                valid_mask = mask_m[:, None] & mask_n[None, :]
                scores = tl.where(valid_mask, scores, float('-inf'))
                
                p = tl.exp(scores - lse[:, None])
                p = tl.where(valid_mask, p, 0.0)
                
                dp = tl.dot(do, v.T)
                ds = p * (dp - delta[:, None]) * scale
                ds = tl.where(valid_mask, ds, 0.0)
                
                dq += tl.dot(ds.to(q.dtype), k)
                
            k_ptrs += BLOCK_N_DQ * stride_ks
            v_ptrs += BLOCK_N_DQ * stride_vs

        # Masked KV loop (causal boundary intersection)
        for kv_idx in tl.range(kv_dense_iters, kv_max_iters, num_stages=PIPELINE_STAGES):
            offs_n = kv_idx * BLOCK_N_DQ + tl.arange(0, BLOCK_N_DQ)
            
            if DIVISIBLE_S:
                k = tl.load(k_ptrs)
                v = tl.load(v_ptrs)
                
                scores = tl.dot(q, k.T) * scale
                valid_mask = offs_m[:, None] >= offs_n[None, :]
                scores = tl.where(valid_mask, scores, float('-inf'))
                
                p = tl.exp(scores - lse[:, None])
                p = tl.where(valid_mask, p, 0.0)
                
                dp = tl.dot(do, v.T)
                ds = p * (dp - delta[:, None]) * scale
                ds = tl.where(valid_mask, ds, 0.0)
                
                dq += tl.dot(ds.to(q.dtype), k)
            else:
                mask_n = offs_n < S
                k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
                
                scores = tl.dot(q, k.T) * scale
                valid_mask = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
                scores = tl.where(valid_mask, scores, float('-inf'))
                
                p = tl.exp(scores - lse[:, None])
                p = tl.where(valid_mask, p, 0.0)
                
                dp = tl.dot(do, v.T)
                ds = p * (dp - delta[:, None]) * scale
                ds = tl.where(valid_mask, ds, 0.0)
                
                dq += tl.dot(ds.to(q.dtype), k)
                
            k_ptrs += BLOCK_N_DQ * stride_ks
            v_ptrs += BLOCK_N_DQ * stride_vs
            
        dq_ptrs = dQ + batch_idx * stride_dqb + head_idx * stride_dqh + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
        
        if DIVISIBLE_S:
            tl.store(dq_ptrs, dq.to(q.dtype))
        else:
            tl.store(dq_ptrs, dq.to(q.dtype), mask=mask_m[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Computes the causal multi-head attention backward pass.
    Writes the gradients to the preallocated dQ, dK, and dV tensors.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    # Enable fast paths if S is exactly divisible by max autotune BLOCK size (128)
    divisible_s = (S % 128 == 0)
    
    def grid(META):
        num_kv_tiles = triton.cdiv(S, META['BLOCK_N_DKDV'])
        num_q_tiles = triton.cdiv(S, META['BLOCK_M_DQ'])
        return (num_kv_tiles + num_q_tiles, B * H)
    
    bwd_kernel[grid](
        Q, K, V, O, dO, L, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        scale, S, H, divisible_s,
        BLOCK_D=d,
    )