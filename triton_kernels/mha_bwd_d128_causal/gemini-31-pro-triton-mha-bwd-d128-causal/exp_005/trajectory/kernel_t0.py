import math
import torch
import triton
import triton.language as tl

@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, dQ, L,
    seq_len,
    scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_lb, stride_lh, stride_ls,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2) 
    
    i_start = pid_s * BLOCK_S
    i_offs = i_start + tl.arange(0, BLOCK_S)
    i_mask = i_offs < seq_len
    
    offs_d = tl.arange(0, HEAD_DIM)
    
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + i_offs[:, None] * stride_qs + offs_d[None, :] * stride_qd
    do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + i_offs[:, None] * stride_dos + offs_d[None, :] * stride_dod
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + i_offs[:, None] * stride_os + offs_d[None, :] * stride_od
    
    q = tl.load(q_ptrs, mask=i_mask[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=i_mask[:, None], other=0.0)
    o = tl.load(o_ptrs, mask=i_mask[:, None], other=0.0)
    
    l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + i_offs * stride_ls
    l = tl.load(l_ptrs, mask=i_mask, other=0.0)
    
    # D_i = sum(O_i * dO_i), computed efficiently on-the-fly per block
    d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    
    dq = tl.zeros((BLOCK_S, HEAD_DIM), dtype=tl.float32)
    
    # j can at most reach up to i's sequence position for causal relations
    num_j_blocks = tl.cdiv(tl.minimum(i_start + BLOCK_S, seq_len), BLOCK_C)
    
    for j_blk in range(0, num_j_blocks):
        j_start = j_blk * BLOCK_C
        j_offs = j_start + tl.arange(0, BLOCK_C)
        j_mask = j_offs < seq_len
        
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + j_offs[:, None] * stride_ks + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + j_offs[:, None] * stride_vs + offs_d[None, :] * stride_vd
        
        k = tl.load(k_ptrs, mask=j_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=j_mask[:, None], other=0.0)
        
        # S_ij = Q_i * K_j^T
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        # Causal mask bounded & sequence limits mapping
        mask_causal = i_offs[:, None] >= j_offs[None, :]
        mask_valid = mask_causal & i_mask[:, None] & j_mask[None, :]
        s = tl.where(mask_valid, s, float('-inf'))
        
        p = tl.exp(s - l[:, None])
        p = tl.where(mask_valid, p, 0.0)  # Suppresses eventual NaNs propagated by (-inf) - (-inf) sequences
        
        # dP_ij = dO_i * V_j^T
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        # dS_ij = P_ij * (dP_ij - D_i) * scale
        ds = p * (dp - d_val[:, None]) * scale
        ds = ds.to(q.dtype)
        
        # dQ_i += dS_ij * K_j
        dq += tl.dot(ds, k, out_dtype=tl.float32)
        
    dq_ptrs = dQ + pid_b * stride_dqb + pid_h * stride_dqh + i_offs[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(dQ.dtype.element_ty), mask=i_mask[:, None])


@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, dK, dV, L,
    seq_len,
    scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_lb, stride_lh, stride_ls,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    j_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    j_mask = j_offs < seq_len
    
    offs_d = tl.arange(0, HEAD_DIM)
    
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + j_offs[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + j_offs[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    k = tl.load(k_ptrs, mask=j_mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=j_mask[:, None], other=0.0)
    
    dk = tl.zeros((BLOCK_C, HEAD_DIM), dtype=tl.float32)
    dv = tl.zeros((BLOCK_C, HEAD_DIM), dtype=tl.float32)
    
    start_i_block = (pid_c * BLOCK_C) // BLOCK_S
    num_i_blocks = tl.cdiv(seq_len, BLOCK_S)
    
    for i_blk in range(start_i_block, num_i_blocks):
        i_start = i_blk * BLOCK_S
        i_offs = i_start + tl.arange(0, BLOCK_S)
        i_mask = i_offs < seq_len
        
        q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + i_offs[:, None] * stride_qs + offs_d[None, :] * stride_qd
        do_ptrs = dO + pid_b * stride_dob + pid_h * stride_doh + i_offs[:, None] * stride_dos + offs_d[None, :] * stride_dod
        o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + i_offs[:, None] * stride_os + offs_d[None, :] * stride_od
        
        q = tl.load(q_ptrs, mask=i_mask[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=i_mask[:, None], other=0.0)
        o = tl.load(o_ptrs, mask=i_mask[:, None], other=0.0)
        
        l_ptrs = L + pid_b * stride_lb + pid_h * stride_lh + i_offs * stride_ls
        l = tl.load(l_ptrs, mask=i_mask, other=0.0)
        
        # D_i re-computed locally for block `i` ensuring isolation compliance
        d_val = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        
        mask_causal = i_offs[:, None] >= j_offs[None, :]
        mask_valid = mask_causal & i_mask[:, None] & j_mask[None, :]
        s = tl.where(mask_valid, s, float('-inf'))
        
        p = tl.exp(s - l[:, None])
        p = tl.where(mask_valid, p, 0.0)
        
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32)
        
        ds = p * (dp - d_val[:, None]) * scale
        ds = ds.to(q.dtype)
        p = p.to(q.dtype)
        
        # dK_j += dS_ij^T * Q_i  |  dV_j += P_ij^T * dO_i
        dk += tl.dot(tl.trans(ds), q, out_dtype=tl.float32)
        dv += tl.dot(tl.trans(p), do, out_dtype=tl.float32)
        
    dk_ptrs = dK + pid_b * stride_dkb + pid_h * stride_dkh + j_offs[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dv_ptrs = dV + pid_b * stride_dvb + pid_h * stride_dvh + j_offs[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    
    tl.store(dk_ptrs, dk.to(dK.dtype.element_ty), mask=j_mask[:, None])
    tl.store(dv_ptrs, dv.to(dV.dtype.element_ty), mask=j_mask[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    with torch.cuda.device(Q.device):
        B, H, seq_len, HEAD_DIM = Q.shape
        scale = 1.0 / math.sqrt(HEAD_DIM)
        
        # Fixed sizing aligned nicely for Tensor Cores without straining register constraints on H100
        BLOCK_S = 64
        BLOCK_C = 64
        
        # Process and resolve backwards gradients for dQ exclusively
        grid_dq = (B, H, triton.cdiv(seq_len, BLOCK_S))
        bwd_dq_kernel[grid_dq](
            Q, K, V, O, dO, dQ, L,
            seq_len, scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C, HEAD_DIM=HEAD_DIM,
            num_warps=4, num_stages=3
        )
        
        # Process and resolve backwards gradients for dK and dV exclusively
        grid_dkdv = (B, H, triton.cdiv(seq_len, BLOCK_C))
        bwd_dk_dv_kernel[grid_dkdv](
            Q, K, V, O, dO, dK, dV, L,
            seq_len, scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            L.stride(0), L.stride(1), L.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C, HEAD_DIM=HEAD_DIM,
            num_warps=4, num_stages=3
        )