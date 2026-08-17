import torch
import triton
import triton.language as tl
import math


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    S_seq, scale,
    stride_q_bh, stride_q_s, stride_q_d,
    stride_k_bh, stride_k_s, stride_k_d,
    stride_v_bh, stride_v_s, stride_v_d,
    stride_o_bh, stride_o_s, stride_o_d,
    stride_do_bh, stride_do_s, stride_do_d,
    stride_l_bh, stride_l_s,
    stride_dq_bh, stride_dq_s, stride_dq_d,
):
    pid_i = tl.program_id(0)
    bh = tl.program_id(1)
    
    i_start = pid_i * 128
    
    row_idx = tl.arange(0, 128)
    d_idx_0 = tl.arange(0, 64)
    d_idx_1 = d_idx_0 + 64
    
    Q0 = tl.load(Q + bh * stride_q_bh + (i_start + row_idx[:, None]) * stride_q_s + d_idx_0[None, :] * stride_q_d, mask=((i_start + row_idx[:, None]) < S_seq), other=0.0)
    Q1 = tl.load(Q + bh * stride_q_bh + (i_start + row_idx[:, None]) * stride_q_s + d_idx_1[None, :] * stride_q_d, mask=((i_start + row_idx[:, None]) < S_seq), other=0.0)
    
    dO0 = tl.load(dO + bh * stride_do_bh + (i_start + row_idx[:, None]) * stride_do_s + d_idx_0[None, :] * stride_do_d, mask=((i_start + row_idx[:, None]) < S_seq), other=0.0)
    dO1 = tl.load(dO + bh * stride_do_bh + (i_start + row_idx[:, None]) * stride_do_s + d_idx_1[None, :] * stride_do_d, mask=((i_start + row_idx[:, None]) < S_seq), other=0.0)
    
    O0 = tl.load(O + bh * stride_o_bh + (i_start + row_idx[:, None]) * stride_o_s + d_idx_0[None, :] * stride_o_d, mask=((i_start + row_idx[:, None]) < S_seq), other=0.0)
    O1 = tl.load(O + bh * stride_o_bh + (i_start + row_idx[:, None]) * stride_o_s + d_idx_1[None, :] * stride_o_d, mask=((i_start + row_idx[:, None]) < S_seq), other=0.0)
    
    E = (dO0 * O0).sum(1, keep_dims=True) + (dO1 * O1).sum(1, keep_dims=True)
    
    dQ0_acc = tl.zeros((128, 64), dtype=tl.float32)
    dQ1_acc = tl.zeros((128, 64), dtype=tl.float32)
    
    num_blocks = tl.cdiv(S_seq, 128)
    col_idx = tl.arange(0, 128)
    
    for j_blk in range(num_blocks):
        j_start = j_blk * 128
        
        K0 = tl.load(K + bh * stride_k_bh + (j_start + row_idx[:, None]) * stride_k_s + d_idx_0[None, :] * stride_k_d, mask=((j_start + row_idx[:, None]) < S_seq), other=0.0)
        K1 = tl.load(K + bh * stride_k_bh + (j_start + row_idx[:, None]) * stride_k_s + d_idx_1[None, :] * stride_k_d, mask=((j_start + row_idx[:, None]) < S_seq), other=0.0)
        
        V0 = tl.load(V + bh * stride_v_bh + (j_start + row_idx[:, None]) * stride_v_s + d_idx_0[None, :] * stride_v_d, mask=((j_start + row_idx[:, None]) < S_seq), other=0.0)
        V1 = tl.load(V + bh * stride_v_bh + (j_start + row_idx[:, None]) * stride_v_s + d_idx_1[None, :] * stride_v_d, mask=((j_start + row_idx[:, None]) < S_seq), other=0.0)
        
        S_acc = tl.dot(Q0, K0.T)
        S_acc = tl.dot(Q1, K1.T, acc=S_acc)
        
        D_acc = tl.dot(dO0, V0.T)
        D_acc = tl.dot(dO1, V1.T, acc=D_acc)
        
        l_row = tl.load(L + bh * stride_l_bh + (i_start + row_idx) * stride_l_s, mask=(i_start + row_idx < S_seq), other=0.0)
        
        P = tl.exp(S_acc * scale - l_row[:, None])
        P = P * ((j_start + col_idx) < S_seq).to(tl.float32)[None, :]
        
        dP = (D_acc - E) * P
        
        dQ0_acc = tl.dot(dP, K0, acc=dQ0_acc)
        dQ1_acc = tl.dot(dP, K1, acc=dQ1_acc)
        
    dQ0_acc = dQ0_acc * scale
    dQ1_acc = dQ1_acc * scale
    
    out_ptr0 = dQ + bh * stride_dq_bh + (i_start + row_idx[:, None]) * stride_dq_s + d_idx_0[None, :] * stride_dq_d
    out_ptr1 = dQ + bh * stride_dq_bh + (i_start + row_idx[:, None]) * stride_dq_s + d_idx_1[None, :] * stride_dq_d
    
    tl.store(out_ptr0, dQ0_acc.to(tl.bfloat16), mask=((i_start + row_idx[:, None]) < S_seq))
    tl.store(out_ptr1, dQ1_acc.to(tl.bfloat16), mask=((i_start + row_idx[:, None]) < S_seq))


@triton.jit
def _bwd_dkv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    S_seq, scale,
    stride_q_bh, stride_q_s, stride_q_d,
    stride_k_bh, stride_k_s, stride_k_d,
    stride_v_bh, stride_v_s, stride_v_d,
    stride_o_bh, stride_o_s, stride_o_d,
    stride_do_bh, stride_do_s, stride_do_d,
    stride_l_bh, stride_l_s,
    stride_dk_bh, stride_dk_s, stride_dk_d,
    stride_dv_bh, stride_dv_s, stride_dv_d,
):
    pid_j = tl.program_id(0)
    bh = tl.program_id(1)
    
    j_start = pid_j * 128
    
    row_idx = tl.arange(0, 128)
    d_idx_0 = tl.arange(0, 64)
    d_idx_1 = d_idx_0 + 64
    
    K0 = tl.load(K + bh * stride_k_bh + (j_start + row_idx[:, None]) * stride_k_s + d_idx_0[None, :] * stride_k_d, mask=((j_start + row_idx[:, None]) < S_seq), other=0.0)
    K1 = tl.load(K + bh * stride_k_bh + (j_start + row_idx[:, None]) * stride_k_s + d_idx_1[None, :] * stride_k_d, mask=((j_start + row_idx[:, None]) < S_seq), other=0.0)
    
    V0 = tl.load(V + bh * stride_v_bh + (j_start + row_idx[:, None]) * stride_v_s + d_idx_0[None, :] * stride_v_d, mask=((j_start + row_idx[:, None]) < S_seq), other=0.0)
    V1 = tl.load(V + bh * stride_v_bh + (j_start + row_idx[:, None]) * stride_v_s + d_idx_1[None, :] * stride_v_d, mask=((j_start + row_idx[:, None]) < S_seq), other=0.0)
    
    dK0_acc = tl.zeros((128, 64), dtype=tl.float32)
    dK1_acc = tl.zeros((128, 64), dtype=tl.float32)
    dV0_acc = tl.zeros((128, 64), dtype=tl.float32)
    dV1_acc = tl.zeros((128, 64), dtype=tl.float32)
    
    num_blocks = tl.cdiv(S_seq, 128)
    col_idx = tl.arange(0, 128)
    
    for i_blk in range(num_blocks):
        i_start = i_blk * 128
        
        Q0 = tl.load(Q + bh * stride_q_bh + (i_start + row_idx[:, None]) * stride_q_s + d_idx_0[None, :] * stride_q_d, mask=((i_start + row_idx[:, None]) < S_seq), other=0.0)
        Q1 = tl.load(Q + bh * stride_q_bh + (i_start + row_idx[:, None]) * stride_q_s + d_idx_1[None, :] * stride_q_d, mask=((i_start + row_idx[:, None]) < S_seq), other=0.0)
        
        dO0 = tl.load(dO + bh * stride_do_bh + (i_start + row_idx[:, None]) * stride_do_s + d_idx_0[None, :] * stride_do_d, mask=((i_start + row_idx[:, None]) < S_seq), other=0.0)
        dO1 = tl.load(dO + bh * stride_do_bh + (i_start + row_idx[:, None]) * stride_do_s + d_idx_1[None, :] * stride_do_d, mask=((i_start + row_idx[:, None]) < S_seq), other=0.0)
        
        O0 = tl.load(O + bh * stride_o_bh + (i_start + row_idx[:, None]) * stride_o_s + d_idx_0[None, :] * stride_o_d, mask=((i_start + row_idx[:, None]) < S_seq), other=0.0)
        O1 = tl.load(O + bh * stride_o_bh + (i_start + row_idx[:, None]) * stride_o_s + d_idx_1[None, :] * stride_o_d, mask=((i_start + row_idx[:, None]) < S_seq), other=0.0)
        
        S_acc = tl.dot(Q0, K0.T)
        S_acc = tl.dot(Q1, K1.T, acc=S_acc)
        
        D_acc = tl.dot(dO0, V0.T)
        D_acc = tl.dot(dO1, V1.T, acc=D_acc)
        
        E = (dO0 * O0).sum(1, keep_dims=True) + (dO1 * O1).sum(1, keep_dims=True)
        
        l_row = tl.load(L + bh * stride_l_bh + (i_start + row_idx) * stride_l_s, mask=(i_start + row_idx < S_seq), other=0.0)
        
        P = tl.exp(S_acc * scale - l_row[:, None])
        P = P * ((i_start + col_idx) < S_seq).to(tl.float32)[None, :]
        
        dP = (D_acc - E) * P
        
        dK0_acc = tl.dot(dP.T, Q0, acc=dK0_acc)
        dK1_acc = tl.dot(dP.T, Q1, acc=dK1_acc)
        
        dV0_acc = tl.dot(P.T, dO0, acc=dV0_acc)
        dV1_acc = tl.dot(P.T, dO1, acc=dV1_acc)
        
    dK0_acc = dK0_acc * scale
    dK1_acc = dK1_acc * scale
    
    out_ptr_dk0 = dK + bh * stride_dk_bh + (j_start + row_idx[:, None]) * stride_dk_s + d_idx_0[None, :] * stride_dk_d
    out_ptr_dk1 = dK + bh * stride_dk_bh + (j_start + row_idx[:, None]) * stride_dk_s + d_idx_1[None, :] * stride_dk_d
    
    tl.store(out_ptr_dk0, dK0_acc.to(tl.bfloat16), mask=((j_start + row_idx[:, None]) < S_seq))
    tl.store(out_ptr_dk1, dK1_acc.to(tl.bfloat16), mask=((j_start + row_idx[:, None]) < S_seq))
    
    out_ptr_dv0 = dV + bh * stride_dv_bh + (j_start + row_idx[:, None]) * stride_dv_s + d_idx_0[None, :] * stride_dv_d
    out_ptr_dv1 = dV + bh * stride_dv_bh + (j_start + row_idx[:, None]) * stride_dv_s + d_idx_1[None, :] * stride_dv_d
    
    tl.store(out_ptr_dv0, dV0_acc.to(tl.bfloat16), mask=((j_start + row_idx[:, None]) < S_seq))
    tl.store(out_ptr_dv1, dV1_acc.to(tl.bfloat16), mask=((j_start + row_idx[:, None]) < S_seq))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward ``dQ, dK, dV`` in place."""
    torch.cuda.set_device(Q.device)
    S_seq = Q.shape[-2]
    scale = 1.0 / math.sqrt(128.0)
    
    bh_total = Q.shape[0] * Q.shape[1]
    num_blocks = triton.cdiv(S_seq, 128)
    
    grid_dq = (num_blocks, bh_total)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ, S_seq, scale,
        Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(1), K.stride(2), K.stride(3),
        V.stride(1), V.stride(2), V.stride(3),
        O.stride(1), O.stride(2), O.stride(3),
        dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(1), L.stride(2),
        dQ.stride(1), dQ.stride(2), dQ.stride(3),
        num_warps=8,
        num_stages=4,
    )
    
    grid_dkv = (num_blocks, bh_total)
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV, S_seq, scale,
        Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(1), K.stride(2), K.stride(3),
        V.stride(1), V.stride(2), V.stride(3),
        O.stride(1), O.stride(2), O.stride(3),
        dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(1), L.stride(2),
        dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(1), dV.stride(2), dV.stride(3),
        num_warps=8,
        num_stages=4,
    )