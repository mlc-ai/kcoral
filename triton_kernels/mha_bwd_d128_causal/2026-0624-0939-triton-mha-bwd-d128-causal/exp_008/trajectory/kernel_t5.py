import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


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
    K_j_raw = k_desc.load([bh, start_n_j, 0])
    K_j = tl.squeeze(K_j_raw, 0)
    V_j_raw = v_desc.load([bh, start_n_j, 0])
    V_j = tl.squeeze(V_j_raw, 0)
    
    dK_acc = tl.zeros((128, 128), tl.float32)
    dV_acc = tl.zeros((128, 128), tl.float32)
    
    row_idx = tl.arange(0, 128)
    col_idx = tl.arange(0, 128)
    
    for i in tl.range(j, num_blocks, 1, num_stages=2):
        start_n_i = i * 128
        Q_i_raw = q_desc.load([bh, start_n_i, 0])
        Q_i = tl.squeeze(Q_i_raw, 0)
        
        dO_i_raw = do_desc.load([bh, start_n_i, 0])
        dO_i = tl.squeeze(dO_i_raw, 0)
        
        L_i_raw = l_desc.load([bh, start_n_i])
        L_i = tl.squeeze(L_i_raw, 0)
        
        D_i_raw = d_desc.load([bh, start_n_i])
        D_i = tl.squeeze(D_i_raw, 0)
        
        S = tl.dot(Q_i, K_j.T)
        dP = tl.dot(dO_i, V_j.T)
        
        diff = S * scale - L_i[:, None]
        exp_val = tl.exp(diff)
        
        query_idx = (i * 128 + row_idx)
        key_idx = (j * 128 + col_idx)
        valid_S = (query_idx[:, None] < S_len) & (key_idx[None, :] < S_len) & (query_idx[:, None] >= key_idx[None, :])
        
        S = tl.where(valid_S, S, 0.0)
        dP = tl.where(valid_S, dP, 0.0)
        
        P = tl.where(valid_S, exp_val, 0.0)
        
        dS = P * (dP - D_i[:, None]) * scale
        
        dS_bf16 = dS.to(tl.bfloat16)
        P_bf16 = P.to(tl.bfloat16)
        
        dV_acc = tl.dot(P_bf16.T, dO_i, dV_acc)
        dK_acc = tl.dot(dS_bf16.T, Q_i, dK_acc)
        
    dk_desc.store([bh, start_n_j, 0], dK_acc.to(tl.bfloat16))
    dv_desc.store([bh, start_n_j, 0], dV_acc.to(tl.bfloat16))


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
    Q_i_raw = q_desc.load([bh, start_n_i, 0])
    Q_i = tl.squeeze(Q_i_raw, 0)
    
    dO_i_raw = do_desc.load([bh, start_n_i, 0])
    dO_i = tl.squeeze(dO_i_raw, 0)
    
    L_i_raw = l_desc.load([bh, start_n_i])
    L_i = tl.squeeze(L_i_raw, 0)
    
    D_i_raw = d_desc.load([bh, start_n_i])
    D_i = tl.squeeze(D_i_raw, 0)
    
    dQ_acc = tl.zeros((128, 128), tl.float32)
    
    row_idx = tl.arange(0, 128)
    col_idx = tl.arange(0, 128)
    query_idx = (i * 128 + row_idx)
    
    for j in tl.range(0, i + 1, 1, num_stages=2):
        start_n_j = j * 128
        K_j_raw = k_desc.load([bh, start_n_j, 0])
        K_j = tl.squeeze(K_j_raw, 0)
        
        V_j_raw = v_desc.load([bh, start_n_j, 0])
        V_j = tl.squeeze(V_j_raw, 0)
        
        S = tl.dot(Q_i, K_j.T)
        dP = tl.dot(dO_i, V_j.T)
        
        diff = S * scale - L_i[:, None]
        exp_val = tl.exp(diff)
        
        key_idx = (j * 128 + col_idx)
        valid_S = (query_idx[:, None] < S_len) & (key_idx[None, :] < S_len) & (query_idx[:, None] >= key_idx[None, :])
        
        S = tl.where(valid_S, S, 0.0)
        dP = tl.where(valid_S, dP, 0.0)
        
        P = tl.where(valid_S, exp_val, 0.0)
        
        dS = P * (dP - D_i[:, None]) * scale
        
        dS_bf16 = dS.to(tl.bfloat16)
        
        dQ_acc = tl.dot(dS_bf16, K_j, dQ_acc)
        
    dq_desc.store([bh, start_n_i, 0], dQ_acc.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    num_blocks = triton.cdiv(S, 128)
    grid = (B * H, num_blocks)
    
    scale = 1.0 / math.sqrt(d)
    
    D = torch.sum(dO * O, dim=-1, dtype=torch.float32)
    
    q_desc = TensorDescriptor.from_tensor(Q.reshape(B * H, S, 128), [1, 128, 128])
    q_desc.padding_option = "zero"
    k_desc = TensorDescriptor.from_tensor(K.reshape(B * H, S, 128), [1, 128, 128])
    k_desc.padding_option = "zero"
    v_desc = TensorDescriptor.from_tensor(V.reshape(B * H, S, 128), [1, 128, 128])
    v_desc.padding_option = "zero"
    do_desc = TensorDescriptor.from_tensor(dO.reshape(B * H, S, 128), [1, 128, 128])
    do_desc.padding_option = "zero"
    l_desc = TensorDescriptor.from_tensor(L.reshape(B * H, S), [1, 128])
    l_desc.padding_option = "zero"
    d_desc = TensorDescriptor.from_tensor(D.reshape(B * H, S), [1, 128])
    d_desc.padding_option = "zero"
    dq_desc = TensorDescriptor.from_tensor(dQ.reshape(B * H, S, 128), [1, 128, 128])
    dk_desc = TensorDescriptor.from_tensor(dK.reshape(B * H, S, 128), [1, 128, 128])
    dv_desc = TensorDescriptor.from_tensor(dV.reshape(B * H, S, 128), [1, 128, 128])
    
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