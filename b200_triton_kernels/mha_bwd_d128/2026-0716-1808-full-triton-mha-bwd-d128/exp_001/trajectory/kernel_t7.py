import torch
import triton
import triton.language as tl
import math


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    S_seq, scale, H_heads,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dq_b, stride_dq_h, stride_dq_s, stride_dq_d,
    BLOCK_S: tl.constexpr,
):
    pid_i = tl.program_id(0)
    bh = tl.program_id(1)
    
    batch_id = bh // H_heads
    head_id = bh % H_heads
    
    i_start = pid_i * BLOCK_S
    
    row_idx = tl.arange(0, BLOCK_S)
    col_idx = tl.arange(0, 64)
    
    mask_i = (i_start + row_idx) < S_seq
    
    off_bh = batch_id * stride_q_b + head_id * stride_q_h
    
    s_Q0 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_Q1 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_dO0 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_dO1 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_K0 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_K1 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_V0 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_V1 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_O0 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_O1 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_dP = tl.empty((BLOCK_S, BLOCK_S), dtype=tl.bfloat16, shared=True)
    s_E = tl.empty((BLOCK_S, 1), dtype=tl.float32, shared=True)
    
    q0 = tl.load(Q + off_bh + (i_start * stride_q_s) + (row_idx[:, None] * stride_q_s) + (col_idx[None, :] * stride_q_d), mask=mask_i[:, None], other=0.0)
    q1 = tl.load(Q + off_bh + (i_start * stride_q_s) + (row_idx[:, None] * stride_q_s) + ((col_idx + 64)[None, :] * stride_q_d), mask=mask_i[:, None], other=0.0)
    
    do0 = tl.load(dO + (batch_id * stride_do_b + head_id * stride_do_h) + (i_start * stride_do_s) + (row_idx[:, None] * stride_do_s) + (col_idx[None, :] * stride_do_d), mask=mask_i[:, None], other=0.0)
    do1 = tl.load(dO + (batch_id * stride_do_b + head_id * stride_do_h) + (i_start * stride_do_s) + (row_idx[:, None] * stride_do_s) + ((col_idx + 64)[None, :] * stride_do_d), mask=mask_i[:, None], other=0.0)
    
    o0 = tl.load(O + (batch_id * stride_o_b + head_id * stride_o_h) + (i_start * stride_o_s) + (row_idx[:, None] * stride_o_s) + (col_idx[None, :] * stride_o_d), mask=mask_i[:, None], other=0.0)
    o1 = tl.load(O + (batch_id * stride_o_b + head_id * stride_o_h) + (i_start * stride_o_s) + (row_idx[:, None] * stride_o_s) + ((col_idx + 64)[None, :] * stride_o_d), mask=mask_i[:, None], other=0.0)
    
    tl.store(s_Q0, q0)
    tl.store(s_Q1, q1)
    tl.store(s_dO0, do0)
    tl.store(s_dO1, do1)
    tl.store(s_O0, o0)
    tl.store(s_O1, o1)
    
    E = (s_dO0 * s_O0).sum(1, keep_dims=True) + (s_dO1 * s_O1).sum(1, keep_dims=True)
    tl.store(s_E, E)
    
    dQ0_acc = tl.zeros((BLOCK_S, 64), dtype=tl.float32)
    dQ1_acc = tl.zeros((BLOCK_S, 64), dtype=tl.float32)
    
    num_blocks = tl.cdiv(S_seq, BLOCK_S)
    
    for j_blk in range(num_blocks):
        j_start = j_blk * BLOCK_S
        mask_j = (j_start + row_idx) < S_seq
        
        off_bh_k = batch_id * stride_k_b + head_id * stride_k_h
        off_bh_v = batch_id * stride_v_b + head_id * stride_v_h
        
        k0 = tl.load(K + off_bh_k + (j_start * stride_k_s) + (row_idx[:, None] * stride_k_s) + (col_idx[None, :] * stride_k_d), mask=mask_j[:, None], other=0.0)
        k1 = tl.load(K + off_bh_k + (j_start * stride_k_s) + (row_idx[:, None] * stride_k_s) + ((col_idx + 64)[None, :] * stride_k_d), mask=mask_j[:, None], other=0.0)
        
        v0 = tl.load(V + off_bh_v + (j_start * stride_v_s) + (row_idx[:, None] * stride_v_s) + (col_idx[None, :] * stride_v_d), mask=mask_j[:, None], other=0.0)
        v1 = tl.load(V + off_bh_v + (j_start * stride_v_s) + (row_idx[:, None] * stride_v_s) + ((col_idx + 64)[None, :] * stride_v_d), mask=mask_j[:, None], other=0.0)
        
        tl.store(s_K0, k0)
        tl.store(s_K1, k1)
        tl.store(s_V0, v0)
        tl.store(s_V1, v1)
        
        S_acc = tl.dot(s_Q0, s_K0.T)
        S_acc = tl.dot(s_Q1, s_K1.T, acc=S_acc)
        
        D_acc = tl.dot(s_dO0, s_V0.T)
        D_acc = tl.dot(s_dO1, s_V1.T, acc=D_acc)
        
        off_bh_l = batch_id * stride_l_b + head_id * stride_l_h
        l_row = tl.load(L + off_bh_l + (i_start * stride_l_s) + (row_idx * stride_l_s), mask=mask_i, other=0.0)
        
        P = tl.exp(S_acc * scale - l_row[:, None])
        P = P * mask_j[None, :].to(tl.float32)
        
        dP = (D_acc - s_E) * P
        
        ds = dP.to(tl.bfloat16)
        tl.store(s_dP, ds)
        
        dQ0_acc = tl.dot(s_dP, s_K0, acc=dQ0_acc)
        dQ1_acc = tl.dot(s_dP, s_K1, acc=dQ1_acc)
        
    dQ0_acc = dQ0_acc * scale
    dQ1_acc = dQ1_acc * scale
    
    off_bh_dq = batch_id * stride_dq_b + head_id * stride_dq_h
    
    out_ptr0 = dQ + off_bh_dq + (i_start * stride_dq_s) + (row_idx[:, None] * stride_dq_s) + (col_idx[None, :] * stride_dq_d)
    out_ptr1 = dQ + off_bh_dq + (i_start * stride_dq_s) + (row_idx[:, None] * stride_dq_s) + ((col_idx + 64)[None, :] * stride_dq_d)
    
    tl.store(out_ptr0, dQ0_acc.to(tl.bfloat16), mask=mask_i[:, None])
    tl.store(out_ptr1, dQ1_acc.to(tl.bfloat16), mask=mask_i[:, None])


@triton.jit
def _bwd_dkv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    S_seq, scale, H_heads,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_h, stride_o_s, stride_o_d,
    stride_do_b, stride_do_h, stride_do_s, stride_do_d,
    stride_l_b, stride_l_h, stride_l_s,
    stride_dk_b, stride_dk_h, stride_dk_s, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_s, stride_dv_d,
    BLOCK_S: tl.constexpr,
):
    pid_j = tl.program_id(0)
    bh = tl.program_id(1)
    
    batch_id = bh // H_heads
    head_id = bh % H_heads
    
    j_start = pid_j * BLOCK_S
    
    row_idx = tl.arange(0, BLOCK_S)
    col_idx = tl.arange(0, 64)
    
    mask_j = (j_start + row_idx) < S_seq
    
    off_bh_k = batch_id * stride_k_b + head_id * stride_k_h
    off_bh_v = batch_id * stride_v_b + head_id * stride_v_h
    
    s_K0 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_K1 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_V0 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_V1 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_Q0 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_Q1 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_dO0 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_dO1 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_O0 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_O1 = tl.empty((BLOCK_S, 64), dtype=tl.bfloat16, shared=True)
    s_dP = tl.empty((BLOCK_S, BLOCK_S), dtype=tl.bfloat16, shared=True)
    s_P = tl.empty((BLOCK_S, BLOCK_S), dtype=tl.bfloat16, shared=True)
    s_E = tl.empty((BLOCK_S, 1), dtype=tl.float32, shared=True)
    
    k0 = tl.load(K + off_bh_k + (j_start * stride_k_s) + (row_idx[:, None] * stride_k_s) + (col_idx[None, :] * stride_k_d), mask=mask_j[:, None], other=0.0)
    k1 = tl.load(K + off_bh_k + (j_start * stride_k_s) + (row_idx[:, None] * stride_k_s) + ((col_idx + 64)[None, :] * stride_k_d), mask=mask_j[:, None], other=0.0)
    
    v0 = tl.load(V + off_bh_v + (j_start * stride_v_s) + (row_idx[:, None] * stride_v_s) + (col_idx[None, :] * stride_v_d), mask=mask_j[:, None], other=0.0)
    v1 = tl.load(V + off_bh_v + (j_start * stride_v_s) + (row_idx[:, None] * stride_v_s) + ((col_idx + 64)[None, :] * stride_v_d), mask=mask_j[:, None], other=0.0)
    
    tl.store(s_K0, k0)
    tl.store(s_K1, k1)
    tl.store(s_V0, v0)
    tl.store(s_V1, v1)
    
    dK0_acc = tl.zeros((BLOCK_S, 64), dtype=tl.float32)
    dK1_acc = tl.zeros((BLOCK_S, 64), dtype=tl.float32)
    dV0_acc = tl.zeros((BLOCK_S, 64), dtype=tl.float32)
    dV1_acc = tl.zeros((BLOCK_S, 64), dtype=tl.float32)
    
    num_blocks = tl.cdiv(S_seq, BLOCK_S)
    
    for i_blk in range(num_blocks):
        i_start = i_blk * BLOCK_S
        mask_i = (i_start + row_idx) < S_seq
        
        off_bh_q = batch_id * stride_q_b + head_id * stride_q_h
        off_bh_do = batch_id * stride_do_b + head_id * stride_do_h
        off_bh_o = batch_id * stride_o_b + head_id * stride_o_h
        
        q0 = tl.load(Q + off_bh_q + (i_start * stride_q_s) + (row_idx[:, None] * stride_q_s) + (col_idx[None, :] * stride_q_d), mask=mask_i[:, None], other=0.0)
        q1 = tl.load(Q + off_bh_q + (i_start * stride_q_s) + (row_idx[:, None] * stride_q_s) + ((col_idx + 64)[None, :] * stride_q_d), mask=mask_i[:, None], other=0.0)
        
        do0 = tl.load(dO + off_bh_do + (i_start * stride_do_s) + (row_idx[:, None] * stride_do_s) + (col_idx[None, :] * stride_do_d), mask=mask_i[:, None], other=0.0)
        do1 = tl.load(dO + off_bh_do + (i_start * stride_do_s) + (row_idx[:, None] * stride_do_s) + ((col_idx + 64)[None, :] * stride_do_d), mask=mask_i[:, None], other=0.0)
        
        o0 = tl.load(O + off_bh_o + (i_start * stride_o_s) + (row_idx[:, None] * stride_o_s) + (col_idx[None, :] * stride_o_d), mask=mask_i[:, None], other=0.0)
        o1 = tl.load(O + off_bh_o + (i_start * stride_o_s) + (row_idx[:, None] * stride_o_s) + ((col_idx + 64)[None, :] * stride_o_d), mask=mask_i[:, None], other=0.0)
        
        tl.store(s_Q0, q0)
        tl.store(s_Q1, q1)
        tl.store(s_dO0, do0)
        tl.store(s_dO1, do1)
        tl.store(s_O0, o0)
        tl.store(s_O1, o1)
        
        S_acc = tl.dot(s_Q0, s_K0.T)
        S_acc = tl.dot(s_Q1, s_K1.T, acc=S_acc)
        
        D_acc = tl.dot(s_dO0, s_V0.T)
        D_acc = tl.dot(s_dO1, s_V1.T, acc=D_acc)
        
        E = (s_dO0 * s_O0).sum(1, keep_dims=True) + (s_dO1 * s_O1).sum(1, keep_dims=True)
        tl.store(s_E, E)
        
        off_bh_l = batch_id * stride_l_b + head_id * stride_l_h
        l_row = tl.load(L + off_bh_l + (i_start * stride_l_s) + (row_idx * stride_l_s), mask=mask_i, other=0.0)
        
        P = tl.exp(S_acc * scale - l_row[:, None])
        P = P * mask_i[None, :].to(tl.float32)
        
        dP = (D_acc - s_E) * P
        
        ds = dP.to(tl.bfloat16)
        p = P.to(tl.bfloat16)
        tl.store(s_dP, ds)
        tl.store(s_P, p)
        
        dK0_acc = tl.dot(s_dP.T, s_Q0, acc=dK0_acc)
        dK1_acc = tl.dot(s_dP.T, s_Q1, acc=dK1_acc)
        
        dV0_acc = tl.dot(s_P.T, s_dO0, acc=dV0_acc)
        dV1_acc = tl.dot(s_P.T, s_dO1, acc=dV1_acc)
        
    dK0_acc = dK0_acc * scale
    dK1_acc = dK1_acc * scale
    
    off_bh_dk = batch_id * stride_dk_b + head_id * stride_dk_h
    off_bh_dv = batch_id * stride_dv_b + head_id * stride_dv_h
    
    out_ptr_dk0 = dK + off_bh_dk + (j_start * stride_dk_s) + (row_idx[:, None] * stride_dk_s) + (col_idx[None, :] * stride_dk_d)
    out_ptr_dk1 = dK + off_bh_dk + (j_start * stride_dk_s) + (row_idx[:, None] * stride_dk_s) + ((col_idx + 64)[None, :] * stride_dk_d)
    
    tl.store(out_ptr_dk0, dK0_acc.to(tl.bfloat16), mask=mask_j[:, None])
    tl.store(out_ptr_dk1, dK1_acc.to(tl.bfloat16), mask=mask_j[:, None])
    
    out_ptr_dv0 = dV + off_bh_dv + (j_start * stride_dv_s) + (row_idx[:, None] * stride_dv_s) + (col_idx[None, :] * stride_dv_d)
    out_ptr_dv1 = dV + off_bh_dv + (j_start * stride_dv_s) + (row_idx[:, None] * stride_dv_s) + ((col_idx + 64)[None, :] * stride_dv_d)
    
    tl.store(out_ptr_dv0, dV0_acc.to(tl.bfloat16), mask=mask_j[:, None])
    tl.store(out_ptr_dv1, dV1_acc.to(tl.bfloat16), mask=mask_j[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward ``dQ, dK, dV`` in place."""
    torch.cuda.set_device(Q.device)
    S_seq = Q.shape[-2]
    scale = 1.0 / math.sqrt(128.0)
    
    bh_total = Q.shape[0] * Q.shape[1]
    H_heads = Q.shape[1]
    BLOCK_S = 128
    
    num_blocks = triton.cdiv(S_seq, BLOCK_S)
    
    grid_dq = (num_blocks, bh_total)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ, S_seq, scale, H_heads,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=2,
        num_ctas=4,
    )
    
    grid_dkv = (num_blocks, bh_total)
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV, S_seq, scale, H_heads,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=2,
        num_ctas=4,
    )