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
    
    i_start = pid_i * 64
    i_idx = tl.arange(0, 64)
    d_idx = tl.arange(0, 64)
    
    mask_i = ((i_start + i_idx[:, None]) < S_seq) & ((d_idx[None, :]) < 128)
    mask_i_1 = ((i_start + i_idx[:, None]) < S_seq) & (((64 + d_idx)[None, :]) < 128)
    
    Q_i0 = tl.load(Q + bh * stride_q_bh + (i_start + i_idx[:, None]) * stride_q_s + d_idx[None, :] * stride_q_d, mask=mask_i, other=0.0)
    Q_i1 = tl.load(Q + bh * stride_q_bh + (i_start + i_idx[:, None]) * stride_q_s + (64 + d_idx)[None, :] * stride_q_d, mask=mask_i_1, other=0.0)
    
    dO_i0 = tl.load(dO + bh * stride_do_bh + (i_start + i_idx[:, None]) * stride_do_s + d_idx[None, :] * stride_do_d, mask=mask_i, other=0.0)
    dO_i1 = tl.load(dO + bh * stride_do_bh + (i_start + i_idx[:, None]) * stride_do_s + (64 + d_idx)[None, :] * stride_do_d, mask=mask_i_1, other=0.0)
    
    O_i0 = tl.load(O + bh * stride_o_bh + (i_start + i_idx[:, None]) * stride_o_s + d_idx[None, :] * stride_o_d, mask=mask_i, other=0.0)
    O_i1 = tl.load(O + bh * stride_o_bh + (i_start + i_idx[:, None]) * stride_o_s + (64 + d_idx)[None, :] * stride_o_d, mask=mask_i_1, other=0.0)
    
    E = (dO_i0 * O_i0).sum(1, keep_dims=True) + (dO_i1 * O_i1).sum(1, keep_dims=True)
    
    dQ0_acc = tl.zeros((64, 64), dtype=tl.float32)
    dQ1_acc = tl.zeros((64, 64), dtype=tl.float32)
    
    num_blocks = triton.cdiv(S_seq, 64)
    
    for j_blk in range(num_blocks):
        j_start = j_blk * 64
        
        mask_j = ((j_start + i_idx[:, None]) < S_seq) & ((d_idx[None, :]) < 128)
        mask_j_1 = ((j_start + i_idx[:, None]) < S_seq) & (((64 + d_idx)[None, :]) < 128)
        
        k0 = tl.load(K + bh * stride_k_bh + (j_start + i_idx[:, None]) * stride_k_s + d_idx[None, :] * stride_k_d, mask=mask_j, other=0.0)
        k1 = tl.load(K + bh * stride_k_bh + (j_start + i_idx[:, None]) * stride_k_s + (64 + d_idx)[None, :] * stride_k_d, mask=mask_j_1, other=0.0)
        
        v0 = tl.load(V + bh * stride_v_bh + (j_start + i_idx[:, None]) * stride_v_s + d_idx[None, :] * stride_v_d, mask=mask_j, other=0.0)
        v1 = tl.load(V + bh * stride_v_bh + (j_start + i_idx[:, None]) * stride_v_s + (64 + d_idx)[None, :] * stride_v_d, mask=mask_j_1, other=0.0)
        
        S_acc = tl.zeros((64, 64), dtype=tl.float32)
        S_acc = tl.dot(Q_i0, k0.T, acc=S_acc)
        S_acc = tl.dot(Q_i1, k1.T, acc=S_acc)
        
        D_acc = tl.zeros((64, 64), dtype=tl.float32)
        D_acc = tl.dot(dO_i0, v0.T, acc=D_acc)
        D_acc = tl.dot(dO_i1, v1.T, acc=D_acc)
        
        l_row = tl.load(L + bh * stride_l_bh + (i_start + i_idx) * stride_l_s, 
                        mask=(i_start + i_idx < S_seq), other=0.0)
        
        P = tl.exp(S_acc * scale - l_row[:, None])
        
        P = P * ((j_start + i_idx) < S_seq).to(tl.float32)[None, :]
        
        dP = (D_acc - E) * P
        
        dQ0_acc = tl.dot(dP, k0, acc=dQ0_acc)
        dQ1_acc = tl.dot(dP, k1, acc=dQ1_acc)
        
    dQ0_acc = dQ0_acc * scale
    dQ1_acc = dQ1_acc * scale
    
    out_ptr0 = dQ + bh * stride_dq_bh + (i_start + i_idx[:, None]) * stride_dq_s + d_idx[None, :] * stride_dq_d
    out_ptr1 = dQ + bh * stride_dq_bh + (i_start + i_idx[:, None]) * stride_dq_s + (64 + d_idx)[None, :] * stride_dq_d
    
    tl.store(out_ptr0, dQ0_acc.to(tl.bfloat16), mask=((i_start + i_idx[:, None]) < S_seq))
    tl.store(out_ptr1, dQ1_acc.to(tl.bfloat16), mask=((i_start + i_idx[:, None]) < S_seq))


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
    
    j_start = pid_j * 64
    j_idx = tl.arange(0, 64)
    d_idx = tl.arange(0, 64)
    
    mask_j = ((j_start + j_idx[:, None]) < S_seq) & ((d_idx[None, :]) < 128)
    mask_j_1 = ((j_start + j_idx[:, None]) < S_seq) & (((64 + d_idx)[None, :]) < 128)
    
    K_j0 = tl.load(K + bh * stride_k_bh + (j_start + j_idx[:, None]) * stride_k_s + d_idx[None, :] * stride_k_d, mask=mask_j, other=0.0)
    K_j1 = tl.load(K + bh * stride_k_bh + (j_start + j_idx[:, None]) * stride_k_s + (64 + d_idx)[None, :] * stride_k_d, mask=mask_j_1, other=0.0)
    
    V_j0 = tl.load(V + bh * stride_v_bh + (j_start + j_idx[:, None]) * stride_v_s + d_idx[None, :] * stride_v_d, mask=mask_j, other=0.0)
    V_j1 = tl.load(V + bh * stride_v_bh + (j_start + j_idx[:, None]) * stride_v_s + (64 + d_idx)[None, :] * stride_v_d, mask=mask_j_1, other=0.0)
    
    dK0_acc = tl.zeros((64, 64), dtype=tl.float32)
    dK1_acc = tl.zeros((64, 64), dtype=tl.float32)
    dV0_acc = tl.zeros((64, 64), dtype=tl.float32)
    dV1_acc = tl.zeros((64, 64), dtype=tl.float32)
    
    num_blocks = triton.cdiv(S_seq, 64)
    
    for i_blk in range(num_blocks):
        i_start = i_blk * 64
        
        mask_i = ((i_start + j_idx[:, None]) < S_seq) & ((d_idx[None, :]) < 128)
        mask_i_1 = ((i_start + j_idx[:, None]) < S_seq) & (((64 + d_idx)[None, :]) < 128)
        
        Q_i0 = tl.load(Q + bh * stride_q_bh + (i_start + j_idx[:, None]) * stride_q_s + d_idx[None, :] * stride_q_d, mask=mask_i, other=0.0)
        Q_i1 = tl.load(Q + bh * stride_q_bh + (i_start + j_idx[:, None]) * stride_q_s + (64 + d_idx)[None, :] * stride_q_d, mask=mask_i_1, other=0.0)
        
        dO_i0 = tl.load(dO + bh * stride_do_bh + (i_start + j_idx[:, None]) * stride_do_s + d_idx[None, :] * stride_do_d, mask=mask_i, other=0.0)
        dO_i1 = tl.load(dO + bh * stride_do_bh + (i_start + j_idx[:, None]) * stride_do_s + (64 + d_idx)[None, :] * stride_do_d, mask=mask_i_1, other=0.0)
        
        O_i0 = tl.load(O + bh * stride_o_bh + (i_start + j_idx[:, None]) * stride_o_s + d_idx[None, :] * stride_o_d, mask=mask_i, other=0.0)
        O_i1 = tl.load(O + bh * stride_o_bh + (i_start + j_idx[:, None]) * stride_o_s + (64 + d_idx)[None, :] * stride_o_d, mask=mask_i_1, other=0.0)
        
        S_acc = tl.zeros((64, 64), dtype=tl.float32)
        S_acc = tl.dot(Q_i0, K_j0.T, acc=S_acc)
        S_acc = tl.dot(Q_i1, K_j1.T, acc=S_acc)
        
        D_acc = tl.zeros((64, 64), dtype=tl.float32)
        D_acc = tl.dot(dO_i0, V_j0.T, acc=D_acc)
        D_acc = tl.dot(dO_i1, V_j1.T, acc=D_acc)
        
        E = (dO_i0 * O_i0).sum(1, keep_dims=True) + (dO_i1 * O_i1).sum(1, keep_dims=True)
        
        l_row = tl.load(L + bh * stride_l_bh + (i_start + j_idx) * stride_l_s, 
                        mask=(i_start + j_idx < S_seq), other=0.0)
        
        P = tl.exp(S_acc * scale - l_row[:, None])
        
        P = P * ((i_start + j_idx) < S_seq).to(tl.float32)[None, :]
        
        dP = (D_acc - E) * P
        
        dP_T = dP.T
        
        dK0_acc = tl.dot(dP_T, Q_i0, acc=dK0_acc)
        dK1_acc = tl.dot(dP_T, Q_i1, acc=dK1_acc)
        
        P_T = P.T
        
        dV0_acc = tl.dot(P_T, dO_i0, acc=dV0_acc)
        dV1_acc = tl.dot(P_T, dO_i1, acc=dV1_acc)
        
    dK0_acc = dK0_acc * scale
    dK1_acc = dK1_acc * scale
    
    out_ptr_dk0 = dK + bh * stride_dk_bh + (j_start + j_idx[:, None]) * stride_dk_s + d_idx[None, :] * stride_dk_d
    out_ptr_dk1 = dK + bh * stride_dk_bh + (j_start + j_idx[:, None]) * stride_dk_s + (64 + d_idx)[None, :] * stride_dk_d
    
    tl.store(out_ptr_dk0, dK0_acc.to(tl.bfloat16), mask=((j_start + j_idx[:, None]) < S_seq))
    tl.store(out_ptr_dk1, dK1_acc.to(tl.bfloat16), mask=((j_start + j_idx[:, None]) < S_seq))
    
    out_ptr_dv0 = dV + bh * stride_dv_bh + (j_start + j_idx[:, None]) * stride_dv_s + d_idx[None, :] * stride_dv_d
    out_ptr_dv1 = dV + bh * stride_dv_bh + (j_start + j_idx[:, None]) * stride_dv_s + (64 + d_idx)[None, :] * stride_dv_d
    
    tl.store(out_ptr_dv0, dV0_acc.to(tl.bfloat16), mask=((j_start + j_idx[:, None]) < S_seq))
    tl.store(out_ptr_dv1, dV1_acc.to(tl.bfloat16), mask=((j_start + j_idx[:, None]) < S_seq))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward ``dQ, dK, dV`` in place."""
    torch.cuda.set_device(Q.device)
    S_seq = Q.shape[-2]
    scale = 1.0 / math.sqrt(128.0)
    
    bh_total = Q.shape[0] * Q.shape[1]
    num_blocks = triton.cdiv(S_seq, 64)
    
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
        num_stages=2,
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
        num_stages=2,
    )