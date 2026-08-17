import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ,
    S_seq,
    stride_q_bh, stride_q_s, stride_q_d,
    stride_k_bh, stride_k_s, stride_k_d,
    stride_v_bh, stride_v_s, stride_v_d,
    stride_o_bh, stride_o_s, stride_o_d,
    stride_do_bh, stride_do_s, stride_do_d,
    stride_l_bh, stride_l_s,
    stride_dq_bh, stride_dq_s, stride_dq_d,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    
    i_offsets = pid_m * 32 + tl.arange(0, 32)
    d_offsets = tl.arange(0, 128)
    
    Q_tile = tl.load(Q + bh * stride_q_bh + i_offsets[:, None] * stride_q_s + d_offsets[None, :] * stride_q_d, 
                     mask=((i_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)), other=0.0).to(tl.float32)
    
    dO_tile = tl.load(dO + bh * stride_do_bh + i_offsets[:, None] * stride_do_s + d_offsets[None, :] * stride_do_d, 
                      mask=((i_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)), other=0.0).to(tl.float32)
    
    dQ_acc = tl.zeros((32, 128), tl.float32)
    
    num_blocks = S_seq // 128
    scale = 1.0 / tl.sqrt(128.0)
    
    for j_blk in range(num_blocks):
        j_start = j_blk * 128
        j_offsets = j_start + tl.arange(0, 128)
        
        K_block = tl.load(K + bh * stride_k_bh + j_offsets[:, None] * stride_k_s + d_offsets[None, :] * stride_k_d,
                          mask=((j_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)), other=0.0).to(tl.float32)
        
        V_block = tl.load(V + bh * stride_v_bh + j_offsets[:, None] * stride_v_s + d_offsets[None, :] * stride_v_d,
                          mask=((j_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)), other=0.0).to(tl.float32)
        
        S = tl.zeros((32, 128), tl.float32)
        S = tl.dot(Q_tile, K_block.T, acc=S)
        S = S * scale
        
        D_full = tl.zeros((32, 128), tl.float32)
        D_full = tl.dot(dO_tile, V_block.T, acc=D_full)
        
        l_col = tl.load(L + bh * stride_l_bh + i_offsets[:, None] * stride_l_s, 
                        mask=((i_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)), other=0.0)
        
        P = tl.exp(S - l_col)
        dP = D_full * P
        
        dQ_acc = tl.dot(dP, K_block, acc=dQ_acc)
        dQ_acc = dQ_acc * scale
    
    out_ptr = dQ + bh * stride_dq_bh + i_offsets[:, None] * stride_dq_s + d_offsets[None, :] * stride_dq_d
    tl.store(out_ptr, dQ_acc.to(tl.bfloat16), mask=((i_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)))


@triton.jit
def _bwd_dkv_kernel(
    Q, K, V, O, dO, L, dK, dV,
    S_seq,
    stride_q_bh, stride_q_s, stride_q_d,
    stride_k_bh, stride_k_s, stride_k_d,
    stride_v_bh, stride_v_s, stride_v_d,
    stride_o_bh, stride_o_s, stride_o_d,
    stride_do_bh, stride_do_s, stride_do_d,
    stride_l_bh, stride_l_s,
    stride_dk_bh, stride_dk_s, stride_dk_d,
    stride_dv_bh, stride_dv_s, stride_dv_d,
):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    
    j_offsets = pid_n * 128 + tl.arange(0, 128)
    d_offsets = tl.arange(0, 128)
    
    K_tile = tl.load(K + bh * stride_k_bh + j_offsets[:, None] * stride_k_s + d_offsets[None, :] * stride_k_d, 
                     mask=((j_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)), other=0.0).to(tl.float32)
    
    V_tile = tl.load(V + bh * stride_v_bh + j_offsets[:, None] * stride_v_s + d_offsets[None, :] * stride_v_d,
                     mask=((j_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)), other=0.0).to(tl.float32)
    
    dK_acc = tl.zeros((128, 128), tl.float32)
    dV_acc = tl.zeros((128, 128), tl.float32)
    
    num_blocks = S_seq // 128
    scale = 1.0 / tl.sqrt(128.0)
    
    for i_blk in range(num_blocks):
        i_start = i_blk * 128
        i_offsets = i_start + tl.arange(0, 128)
        
        Q_block = tl.load(Q + bh * stride_q_bh + i_offsets[:, None] * stride_q_s + d_offsets[None, :] * stride_q_d,
                          mask=((i_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)), other=0.0).to(tl.float32)
        
        dO_block = tl.load(dO + bh * stride_do_bh + i_offsets[:, None] * stride_do_s + d_offsets[None, :] * stride_do_d,
                           mask=((i_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)), other=0.0).to(tl.float32)
        
        O_block = tl.load(O + bh * stride_o_bh + i_offsets[:, None] * stride_o_s + d_offsets[None, :] * stride_o_d,
                          mask=((i_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)), other=0.0).to(tl.float32)
        
        S = tl.zeros((128, 128), tl.float32)
        S = tl.dot(Q_block, K_tile.T, acc=S)
        S = S * scale
        
        D_partial = tl.zeros((128, 128), tl.float32)
        D_partial = tl.dot(dO_block, V_tile.T, acc=D_partial)
        
        E = (dO_block * O_block)
        
        D = D_partial - E[:, None]
        
        l_col = tl.load(L + bh * stride_l_bh + i_offsets[:, None] * stride_l_s, 
                        mask=((i_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)), other=0.0)
        
        P = tl.exp(S - l_col)
        
        P_T = P.T
        D_T = D.T
        
        dP_T = D_T * P_T
        
        dK_acc = tl.dot(dP_T, Q_block, acc=dK_acc)
        dK_acc = dK_acc * scale
        
        dV_acc = tl.dot(P_T, dO_block, acc=dV_acc)
    
    out_ptr_dk = dK + bh * stride_dk_bh + j_offsets[:, None] * stride_dk_s + d_offsets[None, :] * stride_dk_d
    tl.store(out_ptr_dk, dK_acc.to(tl.bfloat16), mask=((j_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)))
    
    out_ptr_dv = dV + bh * stride_dv_bh + j_offsets[:, None] * stride_dv_s + d_offsets[None, :] * stride_dv_d
    tl.store(out_ptr_dv, dV_acc.to(tl.bfloat16), mask=((j_offsets[:, None] < S_seq) & (d_offsets[None, :] < 128)))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward ``dQ, dK, dV`` in place."""
    torch.cuda.set_device(Q.device)
    S_seq = Q.shape[-2]
    
    grid_dq = (S_seq // 32, Q.shape[0] * Q.shape[1])
    _bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ, S_seq,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        num_warps=8,
        num_stages=2,
    )
    
    grid_dkv = (S_seq // 128, K.shape[0] * K.shape[1])
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV, S_seq,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2),
        L.stride(0), L.stride(1),
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        num_warps=8,
        num_stages=2,
    )