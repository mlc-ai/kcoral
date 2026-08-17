import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dKdV_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_desc, dk_desc, dv_desc,
    S_len, scale, num_blocks,
):
    bh = tl.program_id(0)
    j = tl.program_id(1)
    
    start_n_j = j * 128
    K_j = k_desc.load([bh, start_n_j, 0])
    K_j = tl.squeeze(K_j, 0)
    V_j = v_desc.load([bh, start_n_j, 0])
    V_j = tl.squeeze(V_j, 0)
    
    dK_acc = tl.zeros((128, 128), tl.float32)
    dV_acc = tl.zeros((128, 128), tl.float32)
    
    row_idx = tl.arange(0, 128)
    col_idx = tl.arange(0, 128)
    
    for i in range(j, num_blocks):
        start_n_i = i * 128
        Q_i = q_desc.load([bh, start_n_i, 0])
        Q_i = tl.squeeze(Q_i, 0)
        dO_i = do_desc.load([bh, start_n_i, 0])
        dO_i = tl.squeeze(dO_i, 0)
        O_i = o_desc.load([bh, start_n_i, 0])
        O_i = tl.squeeze(O_i, 0)
        
        L_i_raw = l_desc.load([bh, start_n_i])
        L_i = tl.squeeze(L_i_raw, 0)
        
        D_i = tl.sum(dO_i.to(tl.float32) * O_i.to(tl.float32), axis=1)
        
        S = tl.dot(Q_i, K_j.T)
        dP = tl.dot(dO_i, V_j.T)
        
        diff = S * scale - L_i[:, None]
        diff = tl.clamp(diff, -20.0, 20.0)
        exp_val = tl.exp(diff)
        
        query_idx = (i * 128 + row_idx)
        key_idx = (j * 128 + col_idx)
        causal_mask = query_idx[:, None] >= key_idx[None, :]
        valid = causal_mask & (query_idx[:, None] < S_len) & (key_idx[None, :] < S_len)
        P = tl.where(valid, exp_val, 0.0)
        
        dS = P * (dP - D_i[:, None]) * scale
        
        P_bf16 = P.to(tl.bfloat16)
        dS_bf16 = dS.to(tl.bfloat16)
        
        dV_acc = tl.dot(P_bf16.T, dO_i, dV_acc)
        dK_acc = tl.dot(dS_bf16.T, Q_i, dK_acc)
    
    dk_desc.store([bh, start_n_j, 0], dK_acc.to(tl.bfloat16))
    dv_desc.store([bh, start_n_j, 0], dV_acc.to(tl.bfloat16))


@triton.jit
def _bwd_dQ_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_desc, dq_desc,
    S_len, scale, num_blocks,
):
    bh = tl.program_id(0)
    i = tl.program_id(1)
    
    start_n_i = i * 128
    Q_i = q_desc.load([bh, start_n_i, 0])
    Q_i = tl.squeeze(Q_i, 0)
    dO_i = do_desc.load([bh, start_n_i, 0])
    dO_i = tl.squeeze(dO_i, 0)
    O_i = o_desc.load([bh, start_n_i, 0])
    O_i = tl.squeeze(O_i, 0)
    
    L_i_raw = l_desc.load([bh, start_n_i])
    L_i = tl.squeeze(L_i_raw, 0)
    
    D_i = tl.sum(dO_i.to(tl.float32) * O_i.to(tl.float32), axis=1)
    
    dQ_acc = tl.zeros((128, 128), tl.float32)
    
    row_idx = tl.arange(0, 128)
    col_idx = tl.arange(0, 128)
    query_idx = (i * 128 + row_idx)
    
    for j in range(0, i + 1):
        start_n_j = j * 128
        K_j = k_desc.load([bh, start_n_j, 0])
        K_j = tl.squeeze(K_j, 0)
        V_j = v_desc.load([bh, start_n_j, 0])
        V_j = tl.squeeze(V_j, 0)
        
        S = tl.dot(Q_i, K_j.T)
        dP = tl.dot(dO_i, V_j.T)
        
        diff = S * scale - L_i[:, None]
        diff = tl.clamp(diff, -20.0, 20.0)
        exp_val = tl.exp(diff)
        
        key_idx = (j * 128 + col_idx)
        causal_mask = query_idx[:, None] >= key_idx[None, :]
        valid = causal_mask & (query_idx[:, None] < S_len) & (key_idx[None, :] < S_len)
        P = tl.where(valid, exp_val, 0.0)
        
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
    
    q_desc = TensorDescriptor.from_tensor(Q.reshape(B * H, S, d), [1, 128, 128], padding_option="zero")
    k_desc = TensorDescriptor.from_tensor(K.reshape(B * H, S, d), [1, 128, 128], padding_option="zero")
    v_desc = TensorDescriptor.from_tensor(V.reshape(B * H, S, d), [1, 128, 128], padding_option="zero")
    o_desc = TensorDescriptor.from_tensor(O.reshape(B * H, S, d), [1, 128, 128], padding_option="zero")
    do_desc = TensorDescriptor.from_tensor(dO.reshape(B * H, S, d), [1, 128, 128], padding_option="zero")
    l_desc = TensorDescriptor.from_tensor(L.reshape(B * H, S), [1, 128], padding_option="zero")
    dq_desc = TensorDescriptor.from_tensor(dQ.reshape(B * H, S, d), [1, 128, 128])
    dk_desc = TensorDescriptor.from_tensor(dK.reshape(B * H, S, d), [1, 128, 128])
    dv_desc = TensorDescriptor.from_tensor(dV.reshape(B * H, S, d), [1, 128, 128])
    
    _bwd_dKdV_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, do_desc, l_desc, dk_desc, dv_desc,
        S, scale, num_blocks,
        num_warps=8,
        num_stages=2,
    )
    
    _bwd_dQ_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, do_desc, l_desc, dq_desc,
        S, scale, num_blocks,
        num_warps=8,
        num_stages=2,
    )