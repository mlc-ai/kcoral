import math
import torch
import triton
import triton.language as tl

@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr
):
    pid = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    m_start = pid * BLOCK_M
    off_m = m_start + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D)
    
    q_ptrs = Q + b_idx * stride_q_b + h_idx * stride_q_h + off_m[:, None] * stride_q_s + off_d[None, :] * stride_q_d
    o_ptrs = O + b_idx * stride_o_b + h_idx * stride_o_h + off_m[:, None] * stride_o_s + off_d[None, :] * stride_o_d
    do_ptrs = dO + b_idx * stride_do_b + h_idx * stride_do_h + off_m[:, None] * stride_do_s + off_d[None, :] * stride_do_d
    l_ptrs = L + b_idx * stride_l_b + h_idx * stride_l_h + off_m * stride_l_s
    
    mask_m = off_m < S
    
    # Completely eliminate masked loads when block perfectly divides sequence length
    if EVEN_M:
        q_tile = tl.load(q_ptrs)
        o_tile = tl.load(o_ptrs)
        do_tile = tl.load(do_ptrs)
        l_tile = tl.load(l_ptrs)
    else:
        q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o_tile = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do_tile = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l_tile = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
    delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
    dq_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    off_n_init = tl.arange(0, BLOCK_N)
    k_ptrs = K + b_idx * stride_k_b + h_idx * stride_k_h + off_n_init[:, None] * stride_k_s + off_d[None, :] * stride_k_d
    v_ptrs = V + b_idx * stride_v_b + h_idx * stride_v_h + off_n_init[:, None] * stride_v_s + off_d[None, :] * stride_v_d
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n_idx in tl.range(0, num_n_blocks, num_stages=3):
        if not EVEN_N:
            mask_n = (n_idx * BLOCK_N + off_n_init) < S
            
        if EVEN_N:
            k_tile = tl.load(k_ptrs)
            v_tile = tl.load(v_ptrs)
        else:
            k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            
        s_mat = tl.dot(q_tile, k_tile.T, out_dtype=tl.float32) * scale
        
        if not (EVEN_M and EVEN_N):
            valid = mask_m[:, None] & mask_n[None, :]
            s_mat = tl.where(valid, s_mat, -float("inf"))
            
        p_mat = tl.exp(s_mat - l_tile[:, None])
        
        dp_mat = tl.dot(do_tile, v_tile.T, out_dtype=tl.float32)
        ds_mat = p_mat * (dp_mat - delta[:, None]) * scale
        
        if not (EVEN_M and EVEN_N):
            ds_mat = tl.where(valid, ds_mat, 0.0)
            
        dq_acc = tl.dot(ds_mat.to(q_tile.dtype), k_tile, acc=dq_acc)
        
        # Advance pointers for pipelining efficiency
        k_ptrs += BLOCK_N * stride_k_s
        v_ptrs += BLOCK_N * stride_v_s
        
    dq_ptrs = dQ + b_idx * stride_dq_b + h_idx * stride_dq_h + off_m[:, None] * stride_dq_s + off_d[None, :] * stride_dq_d
    if EVEN_M:
        tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty))
    else:
        tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def bwd_dk_dv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    B, H, S, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr
):
    pid = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    n_start = pid * BLOCK_N
    off_n = n_start + tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, D)
    
    k_ptrs = K + b_idx * stride_k_b + h_idx * stride_k_h + off_n[:, None] * stride_k_s + off_d[None, :] * stride_k_d
    v_ptrs = V + b_idx * stride_v_b + h_idx * stride_v_h + off_n[:, None] * stride_v_s + off_d[None, :] * stride_v_d
    
    mask_n = off_n < S
    
    if EVEN_N:
        k_tile = tl.load(k_ptrs)
        v_tile = tl.load(v_ptrs)
    else:
        k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
    dk_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    
    off_m_init = tl.arange(0, BLOCK_M)
    q_ptrs = Q + b_idx * stride_q_b + h_idx * stride_q_h + off_m_init[:, None] * stride_q_s + off_d[None, :] * stride_q_d
    o_ptrs = O + b_idx * stride_o_b + h_idx * stride_o_h + off_m_init[:, None] * stride_o_s + off_d[None, :] * stride_o_d
    do_ptrs = dO + b_idx * stride_do_b + h_idx * stride_do_h + off_m_init[:, None] * stride_do_s + off_d[None, :] * stride_do_d
    l_ptrs = L + b_idx * stride_l_b + h_idx * stride_l_h + off_m_init * stride_l_s
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for m_idx in tl.range(0, num_m_blocks, num_stages=3):
        if not EVEN_M:
            mask_m = (m_idx * BLOCK_M + off_m_init) < S
            
        if EVEN_M:
            q_tile = tl.load(q_ptrs)
            o_tile = tl.load(o_ptrs)
            do_tile = tl.load(do_ptrs)
            l_tile = tl.load(l_ptrs)
        else:
            q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            o_tile = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
            do_tile = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
            l_tile = tl.load(l_ptrs, mask=mask_m, other=0.0)
            
        delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
        
        s_mat_T = tl.dot(k_tile, q_tile.T, out_dtype=tl.float32) * scale
        
        if not (EVEN_M and EVEN_N):
            valid = mask_n[:, None] & mask_m[None, :]
            s_mat_T = tl.where(valid, s_mat_T, -float("inf"))
            
        p_mat_T = tl.exp(s_mat_T - l_tile[None, :])
        
        dp_mat_T = tl.dot(v_tile, do_tile.T, out_dtype=tl.float32)
        ds_mat_T = p_mat_T * (dp_mat_T - delta[None, :]) * scale
        
        if not (EVEN_M and EVEN_N):
            ds_mat_T = tl.where(valid, ds_mat_T, 0.0)
            
        dk_acc = tl.dot(ds_mat_T.to(q_tile.dtype), q_tile, acc=dk_acc)
        dv_acc = tl.dot(p_mat_T.to(q_tile.dtype), do_tile, acc=dv_acc)
        
        q_ptrs += BLOCK_M * stride_q_s
        o_ptrs += BLOCK_M * stride_o_s
        do_ptrs += BLOCK_M * stride_do_s
        l_ptrs += BLOCK_M * stride_l_s
        
    dk_ptrs = dK + b_idx * stride_dk_b + h_idx * stride_dk_h + off_n[:, None] * stride_dk_s + off_d[None, :] * stride_dk_d
    dv_ptrs = dV + b_idx * stride_dv_b + h_idx * stride_dv_h + off_n[:, None] * stride_dv_s + off_d[None, :] * stride_dv_d
    
    if EVEN_N:
        tl.store(dk_ptrs, dk_acc.to(dK.dtype.element_ty))
        tl.store(dv_ptrs, dv_acc.to(dV.dtype.element_ty))
    else:
        tl.store(dk_ptrs, dk_acc.to(dK.dtype.element_ty), mask=mask_n[:, None])
        tl.store(dv_ptrs, dv_acc.to(dV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Highly Optimized Pointer-Based Flash-Attention Backward using split-ownership.
    Provides ideal instruction pipeline behavior safely avoiding unaligned TMA TMEM mapping layouts.
    Uses dynamic compile-time mask exclusion (EVEN_S) maximizing hardware utilization natively.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    # Kernel 1 Pipeline (Compute dQ): Outer-loop streams over M (queries)
    BLOCK_M_1 = 128
    BLOCK_N_1 = 64
    grid_dq = (triton.cdiv(S, BLOCK_M_1), B * H)
    
    # Determine execution time specialization for mask avoidance
    even_m_1 = (S % BLOCK_M_1 == 0)
    even_n_1 = (S % BLOCK_N_1 == 0)
    
    bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H, S, scale,
        BLOCK_M=BLOCK_M_1, BLOCK_N=BLOCK_N_1, D=D,
        EVEN_M=even_m_1, EVEN_N=even_n_1,
        num_warps=8, num_stages=3
    )
    
    # Kernel 2 Pipeline (Compute dK, dV): Outer-loop streams over N (keys/values)
    BLOCK_M_2 = 64
    BLOCK_N_2 = 128
    grid_dk_dv = (triton.cdiv(S, BLOCK_N_2), B * H)
    
    even_m_2 = (S % BLOCK_M_2 == 0)
    even_n_2 = (S % BLOCK_N_2 == 0)
    
    bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, O, dO, L, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, S, scale,
        BLOCK_M=BLOCK_M_2, BLOCK_N=BLOCK_N_2, D=D,
        EVEN_M=even_m_2, EVEN_N=even_n_2,
        num_warps=8, num_stages=3
    )