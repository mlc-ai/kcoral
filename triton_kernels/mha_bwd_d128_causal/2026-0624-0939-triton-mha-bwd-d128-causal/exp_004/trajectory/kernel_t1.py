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
    
    start_k = b * H * S + h * S + j * BLOCK_S
    K_j = K_desc.load([start_k, 0])
    V_j = V_desc.load([start_k, 0])
    
    rows_k = tl.arange(0, BLOCK_S)
    k_seq_idx = j * BLOCK_S + rows_k
    
    for i in range(j, T_r):
        start_q = b * H * S + h * S + i * BLOCK_S
        Q_i = Q_desc.load([start_q, 0])
        dO_i = dO_desc.load([start_q, 0])
        O_i = O_desc.load([start_q, 0])
        
        start_l = b * H * S + h * S + i * BLOCK_S
        L_i = L_desc.load(start_l + rows_q)
        
        rows_q = tl.arange(0, BLOCK_S)
        q_seq_idx = i * BLOCK_S + rows_q
        
        D_i = tl.sum(O_i * dO_i, axis=1)
        
        S_ij = tl.dot(Q_i, K_j.T) * tau
        
        # Exp safety: clamp prevents exp domain issues
        S_ij_capped = tl.maximum(S_ij, -14.0)
        P_ij = tl.exp(S_ij_capped - L_i[:, None])
        
        if i == j:
            causal_mask = k_seq_idx[None, :] <= q_seq_idx[:, None]
            P_ij = tl.where(causal_mask, P_ij, 0.0)
        
        dP_ij = tl.dot(dO_i, V_j.T)
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * tau
        
        dV_acc = tl.dot(P_ij.T, dO_i, acc=dV_acc)
        dK_acc = tl.dot(dS_ij.T, Q_i, acc=dK_acc)
    
    dK_desc.store([start_k, 0], dK_acc.to(torch.bfloat16))
    dV_desc.store([start_k, 0], dV_acc.to(torch.bfloat16))


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
    
    start_q = b * H * S + h * S + i * BLOCK_S
    Q_i = Q_desc.load([start_q, 0])
    dO_i = dO_desc.load([start_q, 0])
    O_i = O_desc.load([start_q, 0])
    
    start_l = b * H * S + h * S + i * BLOCK_S
    L_i = L_desc.load(start_l + rows_q)
    
    rows_q = tl.arange(0, BLOCK_S)
    q_seq_idx = i * BLOCK_S + rows_q
    
    D_i = tl.sum(O_i * dO_i, axis=1)
    
    for j in range(0, i + 1):
        start_k = b * H * S + h * S + j * BLOCK_S
        K_j = K_desc.load([start_k, 0])
        V_j = V_desc.load([start_k, 0])
        
        rows_k = tl.arange(0, BLOCK_S)
        k_seq_idx = j * BLOCK_S + rows_k
        
        S_ij = tl.dot(Q_i, K_j.T) * tau
        
        # Exp safety: clamp prevents exp domain issues
        S_ij_capped = tl.maximum(S_ij, -14.0)
        P_ij = tl.exp(S_ij_capped - L_i[:, None])
        
        if i == j:
            causal_mask = k_seq_idx[None, :] <= q_seq_idx[:, None]
            P_ij = tl.where(causal_mask, P_ij, 0.0)
        
        dP_ij = tl.dot(dO_i, V_j.T)
        dS_ij = P_ij * (dP_ij - D_i[:, None]) * tau
        
        dQ_acc = tl.dot(dS_ij, K_j, acc=dQ_acc)
    
    dQ_desc.store([start_q, 0], dQ_acc.to(torch.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    tau = 1.0 / math.sqrt(d)
    
    # Explicitly track strides to resolve tensor layout mapping accurately
    stride_b = Q.stride(-3)
    stride_h = Q.stride(-2)
    stride_s = Q.stride(-1)
    
    BLOCK_S = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_S, d])
    K_desc = TensorDescriptor.from_tensor(K, [BLOCK_S, d])
    V_desc = TensorDescriptor.from_tensor(V, [BLOCK_S, d])
    O_desc = TensorDescriptor.from_tensor(O, [BLOCK_S, d])
    dO_desc = TensorDescriptor.from_tensor(dO, [BLOCK_S, d])
    L_desc = TensorDescriptor.from_tensor(L, [BLOCK_S])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [BLOCK_S, d])
    dK_desc = TensorDescriptor.from_tensor(dK, [BLOCK_S, d])
    dV_desc = TensorDescriptor.from_tensor(dV, [BLOCK_S, d])
    
    T_r = triton.cdiv(S, BLOCK_S)
    grid = (B, H, T_r)
    
    _bwd_dKdV_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
        S, H, tau, BLOCK_S=BLOCK_S, num_warps=4, num_stages=3
    )
    _bwd_dQ_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc,
        S, H, tau, BLOCK_S=BLOCK_S, num_warps=4, num_stages=3
    )