import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dKdV_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
    S, tau,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    j = tl.program_id(2)
    
    T_r = tl.cdiv(S, BLOCK_S)
    
    if j >= T_r:
        return
    
    dK_acc_0 = tl.zeros((BLOCK_S, 64), tl.float32)
    dK_acc_1 = tl.zeros((BLOCK_S, 64), tl.float32)
    dV_acc_0 = tl.zeros((BLOCK_S, 64), tl.float32)
    dV_acc_1 = tl.zeros((BLOCK_S, 64), tl.float32)
    
    K_j_0 = K_desc.load([b, h, j * BLOCK_S, 0])
    K_j_1 = K_desc.load([b, h, j * BLOCK_S, 64])
    V_j_0 = V_desc.load([b, h, j * BLOCK_S, 0])
    V_j_1 = V_desc.load([b, h, j * BLOCK_S, 64])
    
    for i in range(j, T_r):
        Q_i_0 = Q_desc.load([b, h, i * BLOCK_S, 0])
        Q_i_1 = Q_desc.load([b, h, i * BLOCK_S, 64])
        dO_i_0 = dO_desc.load([b, h, i * BLOCK_S, 0])
        dO_i_1 = dO_desc.load([b, h, i * BLOCK_S, 64])
        O_i_0 = O_desc.load([b, h, i * BLOCK_S, 0])
        O_i_1 = O_desc.load([b, h, i * BLOCK_S, 64])
        
        D_i = tl.sum(O_i_0 * dO_i_0 + O_i_1 * dO_i_1, axis=1)
        
        L_i = L_desc.load([b, h, i * BLOCK_S])
        
        S_ij = tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)
        S_ij = S_ij * tau
        
        if i == j:
            rows_q = tl.arange(0, BLOCK_S)
            cols_k = tl.arange(0, BLOCK_S)
            q_seq_idx = i * BLOCK_S + rows_q
            k_seq_idx = j * BLOCK_S + cols_k
            causal_mask = k_seq_idx[None, :] <= q_seq_idx[:, None]
            S_ij = tl.where(causal_mask, S_ij, -float('inf'))
        
        P_ij = tl.exp(S_ij - L_i[:, None])
        
        if i == j:
            P_ij = tl.where(causal_mask, P_ij, 0.0)
        
        dP_ij = tl.dot(dO_i_0, V_j_0.T) + tl.dot(dO_i_1, V_j_1.T)
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * tau
        
        if i == j:
            dS_ij = tl.where(causal_mask, dS_ij, 0.0)
        
        dV_acc_0 = tl.dot(P_ij.T, dO_i_0, acc=dV_acc_0)
        dV_acc_1 = tl.dot(P_ij.T, dO_i_1, acc=dV_acc_1)
        dK_acc_0 = tl.dot(dS_ij.T, Q_i_0, acc=dK_acc_0)
        dK_acc_1 = tl.dot(dS_ij.T, Q_i_1, acc=dK_acc_1)
    
    dK_desc.store([b, h, j * BLOCK_S, 0], dK_acc_0.to(tl.bfloat16))
    dK_desc.store([b, h, j * BLOCK_S, 64], dK_acc_1.to(tl.bfloat16))
    dV_desc.store([b, h, j * BLOCK_S, 0], dV_acc_0.to(tl.bfloat16))
    dV_desc.store([b, h, j * BLOCK_S, 64], dV_acc_1.to(tl.bfloat16))


@triton.jit
def _bwd_dQ_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc,
    S, tau,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)
    
    T_c = tl.cdiv(S, BLOCK_S)
    
    if i >= T_c:
        return
    
    dQ_acc_0 = tl.zeros((BLOCK_S, 64), tl.float32)
    dQ_acc_1 = tl.zeros((BLOCK_S, 64), tl.float32)
    
    Q_i_0 = Q_desc.load([b, h, i * BLOCK_S, 0])
    Q_i_1 = Q_desc.load([b, h, i * BLOCK_S, 64])
    dO_i_0 = dO_desc.load([b, h, i * BLOCK_S, 0])
    dO_i_1 = dO_desc.load([b, h, i * BLOCK_S, 64])
    O_i_0 = O_desc.load([b, h, i * BLOCK_S, 0])
    O_i_1 = O_desc.load([b, h, i * BLOCK_S, 64])
    
    L_i = L_desc.load([b, h, i * BLOCK_S])
    
    D_i = tl.sum(O_i_0 * dO_i_0 + O_i_1 * dO_i_1, axis=1)
    
    for j in range(0, i + 1):
        K_j_0 = K_desc.load([b, h, j * BLOCK_S, 0])
        K_j_1 = K_desc.load([b, h, j * BLOCK_S, 64])
        V_j_0 = V_desc.load([b, h, j * BLOCK_S, 0])
        V_j_1 = V_desc.load([b, h, j * BLOCK_S, 64])
        
        S_ij = tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)
        S_ij = S_ij * tau
        
        if i == j:
            rows_q = tl.arange(0, BLOCK_S)
            cols_k = tl.arange(0, BLOCK_S)
            q_seq_idx = i * BLOCK_S + rows_q
            k_seq_idx = j * BLOCK_S + cols_k
            causal_mask = k_seq_idx[None, :] <= q_seq_idx[:, None]
            S_ij = tl.where(causal_mask, S_ij, -float('inf'))
        
        P_ij = tl.exp(S_ij - L_i[:, None])
        
        if i == j:
            P_ij = tl.where(causal_mask, P_ij, 0.0)
        
        dP_ij = tl.dot(dO_i_0, V_j_0.T) + tl.dot(dO_i_1, V_j_1.T)
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * tau
        
        if i == j:
            dS_ij = tl.where(causal_mask, dS_ij, 0.0)
        
        dQ_acc_0 = tl.dot(dS_ij, K_j_0, acc=dQ_acc_0)
        dQ_acc_1 = tl.dot(dS_ij, K_j_1, acc=dQ_acc_1)
    
    dQ_desc.store([b, h, i * BLOCK_S, 0], dQ_acc_0.to(tl.bfloat16))
    dQ_desc.store([b, h, i * BLOCK_S, 64], dQ_acc_1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    tau = 1.0 / math.sqrt(d)
    
    BLOCK_S = 64
    T_r = triton.cdiv(S, BLOCK_S)
    
    stride_b = Q.stride(0)
    stride_h = Q.stride(1)
    stride_s = Q.stride(2)
    stride_d = Q.stride(3)
    
    Q_desc = TensorDescriptor(Q, shape=[B, H, S, 64], strides=[H*S*64, S*64, 64, 1], block_shape=[1, 1, 64, 64])
    K_desc = TensorDescriptor(K, shape=[B, H, S, 64], strides=[H*S*64, S*64, 64, 1], block_shape=[1, 1, 64, 64])
    V_desc = TensorDescriptor(V, shape=[B, H, S, 64], strides=[H*S*64, S*64, 64, 1], block_shape=[1, 1, 64, 64])
    O_desc = TensorDescriptor(O, shape=[B, H, S, 64], strides=[H*S*64, S*64, 64, 1], block_shape=[1, 1, 64, 64])
    dO_desc = TensorDescriptor(dO, shape=[B, H, S, 64], strides=[H*S*64, S*64, 64, 1], block_shape=[1, 1, 64, 64])
    
    L_desc = TensorDescriptor(L, shape=[B, H, S], strides=[H*S, S, 1], block_shape=[1, 1, BLOCK_S])
    
    dQ_desc = TensorDescriptor(dQ, shape=[B, H, S, 64], strides=[H*S*64, S*64, 64, 1], block_shape=[1, 1, 64, 64])
    dK_desc = TensorDescriptor(dK, shape=[B, H, S, 64], strides=[H*S*64, S*64, 64, 1], block_shape=[1, 1, 64, 64])
    dV_desc = TensorDescriptor(dV, shape=[B, H, S, 64], strides=[H*S*64, S*64, 64, 1], block_shape=[1, 1, 64, 64])
    
    grid = (B, H, T_r)
    _bwd_dKdV_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
        S, tau, BLOCK_S=BLOCK_S, num_warps=4, num_stages=3
    )
    _bwd_dQ_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc,
        S, tau, BLOCK_S=BLOCK_S, num_warps=4, num_stages=3
    )