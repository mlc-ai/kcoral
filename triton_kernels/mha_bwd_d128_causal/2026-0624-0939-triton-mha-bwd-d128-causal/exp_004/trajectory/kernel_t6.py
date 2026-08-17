import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dKdV_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
    S, H, tau,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    j = tl.program_id(2)
    
    T_r = tl.cdiv(S, BLOCK_S)
    
    if j >= T_r:
        return
    
    dK_acc = tl.zeros((BLOCK_S, 128), tl.float32)
    dV_acc = tl.zeros((BLOCK_S, 128), tl.float32)
    
    K_j = tl.reshape(K_desc.load([b, h, j * BLOCK_S, 0]), [BLOCK_S, 128])
    V_j = tl.reshape(V_desc.load([b, h, j * BLOCK_S, 0]), [BLOCK_S, 128])
    
    rows_k = tl.arange(0, BLOCK_S)
    k_seq_idx = j * BLOCK_S + rows_k
    
    for i in range(j, T_r):
        Q_i = tl.reshape(Q_desc.load([b, h, i * BLOCK_S, 0]), [BLOCK_S, 128])
        dO_i = tl.reshape(dO_desc.load([b, h, i * BLOCK_S, 0]), [BLOCK_S, 128])
        O_i = tl.reshape(O_desc.load([b, h, i * BLOCK_S, 0]), [BLOCK_S, 128])
        
        D_i = tl.sum(O_i * dO_i, axis=1)
        
        L_i = tl.reshape(L_desc.load([b, h, i * BLOCK_S]), [BLOCK_S])
        
        rows_q = tl.arange(0, BLOCK_S)
        q_seq_idx = i * BLOCK_S + rows_q
        
        S_ij = tl.dot(Q_i, K_j.T) * tau
        
        if i == j:
            causal_mask = k_seq_idx[None, :] <= q_seq_idx[:, None]
            S_ij = tl.where(causal_mask, S_ij, -float('inf'))
        
        P_ij = tl.exp(S_ij - L_i[:, None])
        
        if i == j:
            P_ij = tl.where(causal_mask, P_ij, 0.0)
        
        dP_ij = tl.dot(dO_i, V_j.T)
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * tau
        
        if i == j:
            dS_ij = tl.where(causal_mask, dS_ij, 0.0)
        
        dV_acc = tl.dot(P_ij.T, dO_i, acc=dV_acc)
        dK_acc = tl.dot(dS_ij.T, Q_i, acc=dK_acc)
    
    dK_desc.store([b, h, j * BLOCK_S, 0], dK_acc.to(tl.bfloat16))
    dV_desc.store([b, h, j * BLOCK_S, 0], dV_acc.to(tl.bfloat16))


@triton.jit
def _bwd_dQ_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc,
    S, H, tau,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)
    
    T_c = tl.cdiv(S, BLOCK_S)
    
    if i >= T_c:
        return
    
    dQ_acc = tl.zeros((BLOCK_S, 128), tl.float32)
    
    Q_i = tl.reshape(Q_desc.load([b, h, i * BLOCK_S, 0]), [BLOCK_S, 128])
    dO_i = tl.reshape(dO_desc.load([b, h, i * BLOCK_S, 0]), [BLOCK_S, 128])
    O_i = tl.reshape(O_desc.load([b, h, i * BLOCK_S, 0]), [BLOCK_S, 128])
    
    rows_q = tl.arange(0, BLOCK_S)
    q_seq_idx = i * BLOCK_S + rows_q
    
    L_i = tl.reshape(L_desc.load([b, h, i * BLOCK_S]), [BLOCK_S])
    
    D_i = tl.sum(O_i * dO_i, axis=1)
    
    for j in range(0, i + 1):
        K_j = tl.reshape(K_desc.load([b, h, j * BLOCK_S, 0]), [BLOCK_S, 128])
        V_j = tl.reshape(V_desc.load([b, h, j * BLOCK_S, 0]), [BLOCK_S, 128])
        
        S_ij = tl.dot(Q_i, K_j.T) * tau
        
        if i == j:
            rows_k = tl.arange(0, BLOCK_S)
            k_seq_idx = j * BLOCK_S + rows_k
            causal_mask = k_seq_idx[None, :] <= q_seq_idx[:, None]
            S_ij = tl.where(causal_mask, S_ij, -float('inf'))
        
        P_ij = tl.exp(S_ij - L_i[:, None])
        
        if i == j:
            rows_k = tl.arange(0, BLOCK_S)
            k_seq_idx = j * BLOCK_S + rows_k
            causal_mask = k_seq_idx[None, :] <= q_seq_idx[:, None]
            P_ij = tl.where(causal_mask, P_ij, 0.0)
        
        dP_ij = tl.dot(dO_i, V_j.T)
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * tau
        
        if i == j:
            rows_k = tl.arange(0, BLOCK_S)
            k_seq_idx = j * BLOCK_S + rows_k
            causal_mask = k_seq_idx[None, :] <= q_seq_idx[:, None]
            dS_ij = tl.where(causal_mask, dS_ij, 0.0)
        
        dQ_acc = tl.dot(dS_ij, K_j, acc=dQ_acc)
    
    dQ_desc.store([b, h, i * BLOCK_S, 0], dQ_acc.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    tau = 1.0 / math.sqrt(d)
    
    BLOCK_S = 64
    T_r = triton.cdiv(S, BLOCK_S)
    
    Q_flat = Q.view(B * H, S, d)
    K_flat = K.view(B * H, S, d)
    V_flat = V.view(B * H, S, d)
    O_flat = O.view(B * H, S, d)
    dO_flat = dO.view(B * H, S, d)
    dQ_flat = dQ.view(B * H, S, d)
    dK_flat = dK.view(B * H, S, d)
    dV_flat = dV.view(B * H, S, d)
    
    Q_desc = TensorDescriptor.from_tensor(Q_flat, [1, BLOCK_S, d])
    K_desc = TensorDescriptor.from_tensor(K_flat, [1, BLOCK_S, d])
    V_desc = TensorDescriptor.from_tensor(V_flat, [1, BLOCK_S, d])
    O_desc = TensorDescriptor.from_tensor(O_flat, [1, BLOCK_S, d])
    dO_desc = TensorDescriptor.from_tensor(dO_flat, [1, BLOCK_S, d])
    
    L_flat = L.view(B * H, S)
    L_desc = TensorDescriptor.from_tensor(L_flat, [1, BLOCK_S])
    
    dQ_desc = TensorDescriptor.from_tensor(dQ_flat, [1, BLOCK_S, d])
    dK_desc = TensorDescriptor.from_tensor(dK_flat, [1, BLOCK_S, d])
    dV_desc = TensorDescriptor.from_tensor(dV_flat, [1, BLOCK_S, d])
    
    grid = (B, H, T_r)
    _bwd_dKdV_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
        S, H, tau, BLOCK_S=BLOCK_S, num_warps=4, num_stages=3
    )
    _bwd_dQ_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc,
        S, H, tau, BLOCK_S=BLOCK_S, num_warps=4, num_stages=3
    )