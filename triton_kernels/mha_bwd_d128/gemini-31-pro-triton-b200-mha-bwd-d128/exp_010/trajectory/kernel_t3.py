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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    pid = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    # Swizzle program instances for better L2 data reuse
    grid_m = tl.cdiv(S, BLOCK_M)
    GROUP_SIZE = 8
    group_id = pid // GROUP_SIZE
    first_pid_m = group_id * GROUP_SIZE
    group_size_m = min(grid_m - first_pid_m, GROUP_SIZE)
    pid_m = first_pid_m + (pid % group_size_m)
    
    m_start = pid_m * BLOCK_M
    is_last_m = m_start + BLOCK_M > S
    
    off_m = m_start + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D)
    mask_m = off_m < S
    
    # Setup M-centric base pointers
    q_ptrs = Q + b_idx * stride_q_b + h_idx * stride_q_h + off_m[:, None] * stride_q_s + off_d[None, :] * stride_q_d
    o_ptrs = O + b_idx * stride_o_b + h_idx * stride_o_h + off_m[:, None] * stride_o_s + off_d[None, :] * stride_o_d
    do_ptrs = dO + b_idx * stride_do_b + h_idx * stride_do_h + off_m[:, None] * stride_do_s + off_d[None, :] * stride_do_d
    l_ptrs = L + b_idx * stride_l_b + h_idx * stride_l_h + off_m * stride_l_s
    
    # Load completely stationary tensors
    q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o_tile = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do_tile = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l_tile = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Pre-calculate delta for probabilities scaling inside KV-loop
    delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
    dq_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    # Iterate across Key and Value sequences
    k_ptrs = K + b_idx * stride_k_b + h_idx * stride_k_h + off_d[None, :] * stride_k_d
    v_ptrs = V + b_idx * stride_v_b + h_idx * stride_v_h + off_d[None, :] * stride_v_d
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n_idx in tl.range(0, num_n_blocks, num_stages=3):
        n_start = n_idx * BLOCK_N
        is_last_n = n_start + BLOCK_N > S
        need_mask = is_last_m | is_last_n
        
        off_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = off_n < S
        
        # Current stream sequence pointers
        curr_k_ptrs = k_ptrs + off_n[:, None] * stride_k_s
        curr_v_ptrs = v_ptrs + off_n[:, None] * stride_v_s
        
        k_tile = tl.load(curr_k_ptrs, mask=mask_n[:, None], other=0.0)
        v_tile = tl.load(curr_v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # S = Q @ K^T
        s_mat = tl.dot(q_tile, k_tile.T, out_dtype=tl.float32) * scale
        
        # Avoid masking overhead dynamically for full interior blocks 
        valid = mask_m[:, None] & mask_n[None, :]
        if need_mask:
            s_mat = tl.where(valid, s_mat, -float("inf"))
        
        # P = softmax(S)
        p_mat = tl.exp(s_mat - l_tile[:, None])
        
        # dP = dO @ V^T
        dp_mat = tl.dot(do_tile, v_tile.T, out_dtype=tl.float32)
        
        # dS = P * (dP - delta)
        ds_mat = p_mat * (dp_mat - delta[:, None]) * scale
        
        if need_mask:
            ds_mat = tl.where(valid, ds_mat, 0.0)
            
        # dQ = dS @ K
        dq_acc = tl.dot(ds_mat.to(q_tile.dtype), k_tile, acc=dq_acc)
        
    # Store finalized dQ to HBM
    dq_ptrs = dQ + b_idx * stride_dq_b + h_idx * stride_dq_h + off_m[:, None] * stride_dq_s + off_d[None, :] * stride_dq_d
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    pid = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    # Swizzle N-blocks
    grid_n = tl.cdiv(S, BLOCK_N)
    GROUP_SIZE = 8
    group_id = pid // GROUP_SIZE
    first_pid_n = group_id * GROUP_SIZE
    group_size_n = min(grid_n - first_pid_n, GROUP_SIZE)
    pid_n = first_pid_n + (pid % group_size_n)
    
    n_start = pid_n * BLOCK_N
    is_last_n = n_start + BLOCK_N > S
    
    off_n = n_start + tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, D)
    mask_n = off_n < S
    
    # Load stationary K and V chunks from DRAM
    k_ptrs = K + b_idx * stride_k_b + h_idx * stride_k_h + off_n[:, None] * stride_k_s + off_d[None, :] * stride_k_d
    v_ptrs = V + b_idx * stride_v_b + h_idx * stride_v_h + off_n[:, None] * stride_v_s + off_d[None, :] * stride_v_d
    
    k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    dk_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    
    # Stream over query-centric variables
    q_ptrs = Q + b_idx * stride_q_b + h_idx * stride_q_h + off_d[None, :] * stride_q_d
    o_ptrs = O + b_idx * stride_o_b + h_idx * stride_o_h + off_d[None, :] * stride_o_d
    do_ptrs = dO + b_idx * stride_do_b + h_idx * stride_do_h + off_d[None, :] * stride_do_d
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for m_idx in tl.range(0, num_m_blocks, num_stages=3):
        m_start = m_idx * BLOCK_M
        is_last_m = m_start + BLOCK_M > S
        need_mask = is_last_m | is_last_n
        
        off_m = m_start + tl.arange(0, BLOCK_M)
        mask_m = off_m < S
        
        curr_q_ptrs = q_ptrs + off_m[:, None] * stride_q_s
        curr_o_ptrs = o_ptrs + off_m[:, None] * stride_o_s
        curr_do_ptrs = do_ptrs + off_m[:, None] * stride_do_s
        
        q_tile = tl.load(curr_q_ptrs, mask=mask_m[:, None], other=0.0)
        o_tile = tl.load(curr_o_ptrs, mask=mask_m[:, None], other=0.0)
        do_tile = tl.load(curr_do_ptrs, mask=mask_m[:, None], other=0.0)
        
        l_ptr = L + b_idx * stride_l_b + h_idx * stride_l_h + off_m * stride_l_s
        l_tile = tl.load(l_ptr, mask=mask_m, other=0.0)
        
        # Log Sum Exp scaling metric
        delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
        
        # S^T = K @ Q^T
        s_mat_T = tl.dot(k_tile, q_tile.T, out_dtype=tl.float32) * scale
        
        valid = mask_n[:, None] & mask_m[None, :]
        if need_mask:
            s_mat_T = tl.where(valid, s_mat_T, -float("inf"))
        
        # P^T = softmax(S^T)
        p_mat_T = tl.exp(s_mat_T - l_tile[None, :])
        
        # dP^T = V @ dO^T
        dp_mat_T = tl.dot(v_tile, do_tile.T, out_dtype=tl.float32)
        
        # dS^T = P^T * (dP^T - delta)
        ds_mat_T = p_mat_T * (dp_mat_T - delta[None, :]) * scale
        
        if need_mask:
            ds_mat_T = tl.where(valid, ds_mat_T, 0.0)
            
        # Accumulate corresponding sequence slices
        dk_acc = tl.dot(ds_mat_T.to(q_tile.dtype), q_tile, acc=dk_acc)
        dv_acc = tl.dot(p_mat_T.to(q_tile.dtype), do_tile, acc=dv_acc)
        
    dk_ptrs = dK + b_idx * stride_dk_b + h_idx * stride_dk_h + off_n[:, None] * stride_dk_s + off_d[None, :] * stride_dk_d
    dv_ptrs = dV + b_idx * stride_dv_b + h_idx * stride_dv_h + off_n[:, None] * stride_dv_s + off_d[None, :] * stride_dv_d
    
    tl.store(dk_ptrs, dk_acc.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv_acc.to(dV.dtype.element_ty), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Split-Ownership Attention Backward implemented securely via standard pointer logic. 
    It eliminates structural race conditions by isolating distinct dQ and dK/dV accumulation phases
    without invoking Triton atomics, and ensures TMA footprint safety via tightly configured block boundaries. 
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    # Kernel 1 Pipeline (Compute dQ): Optimal mapping over Queries 
    BLOCK_M_1 = 128
    BLOCK_N_1 = 64
    grid_dq = (triton.cdiv(S, BLOCK_M_1), B * H)
    
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
        num_warps=8, num_stages=3
    )
    
    # Kernel 2 Pipeline (Compute dK, dV): Optimal mapping over Keys/Values
    BLOCK_N_2 = 128
    BLOCK_M_2 = 64
    grid_dk_dv = (triton.cdiv(S, BLOCK_N_2), B * H)
    
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
        num_warps=8, num_stages=3
    )