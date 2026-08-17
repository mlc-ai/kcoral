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
    B, H, S,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    # Program identity mapping
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    # Generate query offsets
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D)
    
    # Initialize base pointers for queries (M dimension blocks)
    q_ptrs = Q + b_idx * stride_q_b + h_idx * stride_q_h + off_m[:, None] * stride_q_s + off_d[None, :] * stride_q_d
    o_ptrs = O + b_idx * stride_o_b + h_idx * stride_o_h + off_m[:, None] * stride_o_s + off_d[None, :] * stride_o_d
    do_ptrs = dO + b_idx * stride_do_b + h_idx * stride_do_h + off_m[:, None] * stride_do_s + off_d[None, :] * stride_do_d
    l_ptrs = L + b_idx * stride_l_b + h_idx * stride_l_h + off_m * stride_l_s
    
    mask_m = off_m < S
    
    # Pre-load query-related local tensor tiles
    q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    o_tile = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do_tile = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
    l_tile = tl.load(l_ptrs, mask=mask_m, other=0.0)
    
    # Compute rowwise scale coefficient delta = sum(dO * O, axis=1)
    delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
    
    # Preallocate dQ FP32 accumulator
    dq_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    # Set up key/value base pointers 
    off_n = tl.arange(0, BLOCK_N)
    k_ptrs = K + b_idx * stride_k_b + h_idx * stride_k_h + off_n[:, None] * stride_k_s + off_d[None, :] * stride_k_d
    v_ptrs = V + b_idx * stride_v_b + h_idx * stride_v_h + off_n[:, None] * stride_v_s + off_d[None, :] * stride_v_d
    
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n_idx in range(num_n_blocks):
        curr_n = n_idx * BLOCK_N + off_n
        mask_n = curr_n < S
        
        # Load local K and V patches
        k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Protect against non-rectangular padding and infinite exp scaling bugs
        valid = mask_m[:, None] & mask_n[None, :]
        
        # Forward pass components
        s_mat = tl.dot(q_tile, k_tile.T, out_dtype=tl.float32) * scale
        s_mat = tl.where(valid, s_mat, -float("inf"))
        
        p_mat = tl.exp(s_mat - l_tile[:, None])
        
        # Backward partial derivatives
        dp_mat = tl.dot(do_tile, v_tile.T, out_dtype=tl.float32)
        ds_mat = p_mat * (dp_mat - delta[:, None]) * scale
        
        # Downcast for Tensor Cores and combine into dQ tile accumulator
        dq_acc = tl.dot(ds_mat.to(q_tile.dtype), k_tile, acc=dq_acc)
        
        # Advance pointers toward next stream KV-block
        k_ptrs += BLOCK_N * stride_k_s
        v_ptrs += BLOCK_N * stride_v_s

    # Store fully computed dQ values
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
    B, H, S,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    # This kernel iterates over N-blocks (KV), keeping K/V patches stationary in registers
    # and streaming over M-blocks (Q, O, dO, L) mapping out dK and dV outputs without Atomics.
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    # Generate static KV offsets
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, D)
    mask_n = off_n < S
    
    # KV pointers 
    k_ptrs = K + b_idx * stride_k_b + h_idx * stride_k_h + off_n[:, None] * stride_k_s + off_d[None, :] * stride_k_d
    v_ptrs = V + b_idx * stride_v_b + h_idx * stride_v_h + off_n[:, None] * stride_v_s + off_d[None, :] * stride_v_d
    
    # Read invariant Key and Value sub-chunks
    k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    
    # Base accumulators mapped for FP32 outputs
    dk_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    
    # Initial dynamic Q-related pointers 
    off_m = tl.arange(0, BLOCK_M)
    q_ptrs = Q + b_idx * stride_q_b + h_idx * stride_q_h + off_m[:, None] * stride_q_s + off_d[None, :] * stride_q_d
    o_ptrs = O + b_idx * stride_o_b + h_idx * stride_o_h + off_m[:, None] * stride_o_s + off_d[None, :] * stride_o_d
    do_ptrs = dO + b_idx * stride_do_b + h_idx * stride_do_h + off_m[:, None] * stride_do_s + off_d[None, :] * stride_do_d
    l_ptrs = L + b_idx * stride_l_b + h_idx * stride_l_h + off_m * stride_l_s
    
    num_m_blocks = tl.cdiv(S, BLOCK_M)
    for m_idx in range(num_m_blocks):
        curr_m = m_idx * BLOCK_M + off_m
        mask_m = curr_m < S
        
        # Pipelined Query-centric block reading 
        q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        o_tile = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
        do_tile = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
        l_tile = tl.load(l_ptrs, mask=mask_m, other=0.0)
        
        delta = tl.sum(do_tile.to(tl.float32) * o_tile.to(tl.float32), axis=1)
        
        valid = mask_n[:, None] & mask_m[None, :]
        
        # Calculate transposed matching scores logic since this computes from perspective of N->M interactions
        s_mat_T = tl.dot(k_tile, q_tile.T, out_dtype=tl.float32) * scale
        s_mat_T = tl.where(valid, s_mat_T, -float("inf"))
        
        p_mat_T = tl.exp(s_mat_T - l_tile[None, :])
        
        dp_mat_T = tl.dot(v_tile, do_tile.T, out_dtype=tl.float32)
        ds_mat_T = p_mat_T * (dp_mat_T - delta[None, :]) * scale
        
        # Aggregate localized partial results
        dk_acc = tl.dot(ds_mat_T.to(q_tile.dtype), q_tile, acc=dk_acc)
        dv_acc = tl.dot(p_mat_T.to(q_tile.dtype), do_tile, acc=dv_acc)
        
        # Advance sequence loop for M coordinates 
        q_ptrs += BLOCK_M * stride_q_s
        o_ptrs += BLOCK_M * stride_o_s
        do_ptrs += BLOCK_M * stride_do_s
        l_ptrs += BLOCK_M * stride_l_s
        
    dk_ptrs = dK + b_idx * stride_dk_b + h_idx * stride_dk_h + off_n[:, None] * stride_dk_s + off_d[None, :] * stride_dk_d
    dv_ptrs = dV + b_idx * stride_dv_b + h_idx * stride_dv_h + off_n[:, None] * stride_dv_s + off_d[None, :] * stride_dv_d
    
    # Store finalized localized deterministic dK and dV tiles 
    tl.store(dk_ptrs, dk_acc.to(dK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv_acc.to(dV.dtype.element_ty), mask=mask_n[:, None])

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Standard Flash-Attention style bidirectional pass using pre-allocated outputs.
    Follows split-ownership execution to circumvent non-deterministic atomic collisions. 
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D_head = Q.shape
    scale = 1.0 / math.sqrt(D_head)
    
    # Kernel tile sizing for 128 embedding widths
    BLOCK_M = 64
    BLOCK_N = 64
    
    # 1. dQ Kernel Pipeline 
    grid_dq = (triton.cdiv(S, BLOCK_M), B * H)
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
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=128,
        num_warps=4, num_stages=2
    )
    
    # 2. dK and dV Kernel Pipeline 
    grid_dk_dv = (triton.cdiv(S, BLOCK_N), B * H)
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
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=128,
        num_warps=4, num_stages=2
    )