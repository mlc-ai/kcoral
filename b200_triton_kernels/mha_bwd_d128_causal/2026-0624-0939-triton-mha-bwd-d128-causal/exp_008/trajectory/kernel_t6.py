import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _preprocess_D_kernel(
    dO_ptr, O_ptr, D_ptr,
    S_len,
    STRIDE_BH, STRIDE_S, STRIDE_D,
    BLOCK_SIZE: tl.constexpr,
):
    bh = tl.program_id(0)
    start_s = tl.program_id(1) * BLOCK_SIZE
    
    if start_s >= S_len:
        return
    
    s_idx = start_s + tl.arange(0, BLOCK_SIZE)
    d_idx_0 = tl.arange(0, 64)
    d_idx_1 = tl.arange(0, 64)
    
    valid_s = s_idx < S_len
    
    do_base_0 = bh * STRIDE_BH + s_idx[:, None] * STRIDE_S + d_idx_0[None, :]
    do_0 = tl.load(dO_ptr + do_base_0, mask=valid_s[:, None], other=0.0)
    
    do_base_1 = bh * STRIDE_BH + s_idx[:, None] * STRIDE_S + 64 + d_idx_1[None, :]
    do_1 = tl.load(dO_ptr + do_base_1, mask=valid_s[:, None], other=0.0)
    
    o_base_0 = bh * STRIDE_BH + s_idx[:, None] * STRIDE_S + d_idx_0[None, :]
    o_0 = tl.load(O_ptr + o_base_0, mask=valid_s[:, None], other=0.0)
    
    o_base_1 = bh * STRIDE_BH + s_idx[:, None] * STRIDE_S + 64 + d_idx_1[None, :]
    o_1 = tl.load(O_ptr + o_base_1, mask=valid_s[:, None], other=0.0)
    
    D = tl.sum(do_0.to(tl.float32) * o_0.to(tl.float32) + do_1.to(tl.float32) * o_1.to(tl.float32), axis=1)
    
    D = tl.where(valid_s, D, 0.0)
    
    d_base = bh * STRIDE_BH + s_idx
    tl.store(D_ptr + d_base, D, mask=valid_s)


@triton.jit
def _bwd_dKdV_kernel(
    q_desc, k_desc, v_desc, do_desc, l_desc, d_desc, dk_desc, dv_desc,
    S_len, scale, num_blocks,
):
    bh = tl.program_id(0)
    j = tl.program_id(1)
    
    if j >= num_blocks:
        return
    
    start_n_j = j * 128
    K_j_0_raw = k_desc.load([bh, start_n_j, 0])
    K_j_0 = tl.squeeze(K_j_0_raw, 0)
    K_j_1_raw = k_desc.load([bh, start_n_j, 64])
    K_j_1 = tl.squeeze(K_j_1_raw, 0)
    V_j_0_raw = v_desc.load([bh, start_n_j, 0])
    V_j_0 = tl.squeeze(V_j_0_raw, 0)
    V_j_1_raw = v_desc.load([bh, start_n_j, 64])
    V_j_1 = tl.squeeze(V_j_1_raw, 0)
    
    dK_acc_0 = tl.zeros((128, 64), tl.float32)
    dK_acc_1 = tl.zeros((128, 64), tl.float32)
    dV_acc_0 = tl.zeros((128, 64), tl.float32)
    dV_acc_1 = tl.zeros((128, 64), tl.float32)
    
    row_idx = tl.arange(0, 128)
    col_idx = tl.arange(0, 128)
    
    for i in range(j, num_blocks):
        start_n_i = i * 128
        Q_i_0_raw = q_desc.load([bh, start_n_i, 0])
        Q_i_0 = tl.squeeze(Q_i_0_raw, 0)
        Q_i_1_raw = q_desc.load([bh, start_n_i, 64])
        Q_i_1 = tl.squeeze(Q_i_1_raw, 0)
        
        dO_i_0_raw = do_desc.load([bh, start_n_i, 0])
        dO_i_0 = tl.squeeze(dO_i_0_raw, 0)
        dO_i_1_raw = do_desc.load([bh, start_n_i, 64])
        dO_i_1 = tl.squeeze(dO_i_1_raw, 0)
        
        L_i_raw = l_desc.load([bh, start_n_i])
        L_i = tl.squeeze(L_i_raw, 0)
        
        D_i_raw = d_desc.load([bh, start_n_i])
        D_i = tl.squeeze(D_i_raw, 0)
        
        S = tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)
        dP = tl.dot(dO_i_0, V_j_0.T) + tl.dot(dO_i_1, V_j_1.T)
        
        query_idx = (i * 128 + row_idx)
        key_idx = (j * 128 + col_idx)
        valid_S = (query_idx[:, None] < S_len) & (key_idx[None, :] < S_len) & (query_idx[:, None] >= key_idx[None, :])
        
        S = tl.where(valid_S, S, 0.0)
        dP = tl.where(valid_S, dP, 0.0)
        
        diff = S * scale - L_i[:, None]
        exp_val = tl.exp(diff)
        
        P = tl.where(valid_S, exp_val, 0.0)
        
        dS = P * (dP - D_i[:, None]) * scale
        
        dS_bf16 = dS.to(tl.bfloat16)
        P_bf16 = P.to(tl.bfloat16)
        
        dV_acc_0 = tl.dot(P_bf16.T, dO_i_0, dV_acc_0)
        dV_acc_1 = tl.dot(P_bf16.T, dO_i_1, dV_acc_1)
        dK_acc_0 = tl.dot(dS_bf16.T, Q_i_0, dK_acc_0)
        dK_acc_1 = tl.dot(dS_bf16.T, Q_i_1, dK_acc_1)
        
    dk_desc.store([bh, start_n_j, 0], dK_acc_0.to(tl.bfloat16))
    dk_desc.store([bh, start_n_j, 64], dK_acc_1.to(tl.bfloat16))
    dv_desc.store([bh, start_n_j, 0], dV_acc_0.to(tl.bfloat16))
    dv_desc.store([bh, start_n_j, 64], dV_acc_1.to(tl.bfloat16))


@triton.jit
def _bwd_dQ_kernel(
    q_desc, k_desc, v_desc, do_desc, l_desc, d_desc, dq_desc,
    S_len, scale, num_blocks,
):
    bh = tl.program_id(0)
    i = tl.program_id(1)
    
    if i >= num_blocks:
        return
    
    start_n_i = i * 128
    Q_i_0_raw = q_desc.load([bh, start_n_i, 0])
    Q_i_0 = tl.squeeze(Q_i_0_raw, 0)
    Q_i_1_raw = q_desc.load([bh, start_n_i, 64])
    Q_i_1 = tl.squeeze(Q_i_1_raw, 0)
    
    dO_i_0_raw = do_desc.load([bh, start_n_i, 0])
    dO_i_0 = tl.squeeze(dO_i_0_raw, 0)
    dO_i_1_raw = do_desc.load([bh, start_n_i, 64])
    dO_i_1 = tl.squeeze(dO_i_1_raw, 0)
    
    L_i_raw = l_desc.load([bh, start_n_i])
    L_i = tl.squeeze(L_i_raw, 0)
    
    D_i_raw = d_desc.load([bh, start_n_i])
    D_i = tl.squeeze(D_i_raw, 0)
    
    dQ_acc_0 = tl.zeros((128, 64), tl.float32)
    dQ_acc_1 = tl.zeros((128, 64), tl.float32)
    
    row_idx = tl.arange(0, 128)
    col_idx = tl.arange(0, 128)
    query_idx = (i * 128 + row_idx)
    
    for j in range(0, i + 1):
        start_n_j = j * 128
        K_j_0_raw = k_desc.load([bh, start_n_j, 0])
        K_j_0 = tl.squeeze(K_j_0_raw, 0)
        K_j_1_raw = k_desc.load([bh, start_n_j, 64])
        K_j_1 = tl.squeeze(K_j_1_raw, 0)
        
        V_j_0_raw = v_desc.load([bh, start_n_j, 0])
        V_j_0 = tl.squeeze(V_j_0_raw, 0)
        V_j_1_raw = v_desc.load([bh, start_n_j, 64])
        V_j_1 = tl.squeeze(V_j_1_raw, 0)
        
        S = tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)
        dP = tl.dot(dO_i_0, V_j_0.T) + tl.dot(dO_i_1, V_j_1.T)
        
        key_idx = (j * 128 + col_idx)
        valid_S = (query_idx[:, None] < S_len) & (key_idx[None, :] < S_len) & (query_idx[:, None] >= key_idx[None, :])
        
        S = tl.where(valid_S, S, 0.0)
        dP = tl.where(valid_S, dP, 0.0)
        
        diff = S * scale - L_i[:, None]
        exp_val = tl.exp(diff)
        
        P = tl.where(valid_S, exp_val, 0.0)
        
        dS = P * (dP - D_i[:, None]) * scale
        
        dS_bf16 = dS.to(tl.bfloat16)
        
        dQ_acc_0 = tl.dot(dS_bf16, K_j_0, dQ_acc_0)
        dQ_acc_1 = tl.dot(dS_bf16, K_j_1, dQ_acc_1)
        
    dq_desc.store([bh, start_n_i, 0], dQ_acc_0.to(tl.bfloat16))
    dq_desc.store([bh, start_n_i, 64], dQ_acc_1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    num_blocks = triton.cdiv(S, 128)
    grid = (B * H, num_blocks)
    
    scale = 1.0 / math.sqrt(d)
    
    q_desc = TensorDescriptor.from_tensor(Q.reshape(B * H, S, 128), [1, 128, 64])
    q_desc.padding_option = "zero"
    k_desc = TensorDescriptor.from_tensor(K.reshape(B * H, S, 128), [1, 128, 64])
    k_desc.padding_option = "zero"
    v_desc = TensorDescriptor.from_tensor(V.reshape(B * H, S, 128), [1, 128, 64])
    v_desc.padding_option = "zero"
    do_desc = TensorDescriptor.from_tensor(dO.reshape(B * H, S, 128), [1, 128, 64])
    do_desc.padding_option = "zero"
    l_desc = TensorDescriptor.from_tensor(L.reshape(B * H, S), [1, 128])
    l_desc.padding_option = "zero"
    d_desc = TensorDescriptor.from_tensor(D.reshape(B * H, S), [1, 128])
    d_desc.padding_option = "zero"
    dq_desc = TensorDescriptor.from_tensor(dQ.reshape(B * H, S, 128), [1, 128, 64])
    dk_desc = TensorDescriptor.from_tensor(dK.reshape(B * H, S, 128), [1, 128, 64])
    dv_desc = TensorDescriptor.from_tensor(dV.reshape(B * H, S, 128), [1, 128, 64])
    
    STRIDE_BH = Q.stride()[1]
    STRIDE_S = Q.stride()[2]
    STRIDE_D = Q.stride()[3]
    
    _bwd_dKdV_kernel[grid](
        q_desc, k_desc, v_desc, do_desc, l_desc, d_desc, dk_desc, dv_desc,
        S, scale, num_blocks,
        num_warps=8,
        num_stages=2,
    )
    
    _bwd_dQ_kernel[grid](
        q_desc, k_desc, v_desc, do_desc, l_desc, d_desc, dq_desc,
        S, scale, num_blocks,
        num_warps=8,
        num_stages=2,
    )