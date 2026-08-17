import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dq_kernel(
    descs_Q, descs_K, descs_V, descs_dO, descs_O, descs_dQ,
    L_ptr, S_len, tau,
    STRIDE_L_B: tl.constexpr, STRIDE_L_H: tl.constexpr,
):
    i = tl.program_id(0)
    b_idx = tl.program_id(1)
    h_idx = tl.program_id(2)
    off_i = i * 64
    
    desc_Q = descs_Q[b_idx][h_idx]
    desc_K = descs_K[b_idx][h_idx]
    desc_V = descs_V[b_idx][h_idx]
    desc_dO = descs_dO[b_idx][h_idx]
    desc_O = descs_O[b_idx][h_idx]
    desc_dQ = descs_dQ[b_idx][h_idx]
    
    Q_i = desc_Q.load([off_i, 0])
    dO_i = desc_dO.load([off_i, 0])
    O_i = desc_O.load([off_i, 0])
    
    D_i = tl.sum(dO_i * O_i, axis=1)  
    
    row_idx = tl.arange(0, 64)
    global_row = off_i + row_idx
    lse_ptr = L_ptr + b_idx * STRIDE_L_B + h_idx * STRIDE_L_H + global_row
    L_i = tl.load(lse_ptr, mask=global_row < S_len, other=0.0)  
    
    dQ_acc = tl.zeros((64, 128), tl.float32)
    
    num_blocks = tl.cdiv(S_len, 64)
    
    for j in range(0, i + 1):
        off_j = j * 64
        
        K_j = desc_K.load([off_j, 0])
        V_j = desc_V.load([off_j, 0])
        
        S_val = tl.dot(Q_i, K_j.T)
        dP = tl.dot(dO_i, V_j.T)
        
        col_idx = tl.arange(0, 64)
        global_row_2d = global_row[:, None]
        global_col_2d = off_j + col_idx[None, :]
        mask = (global_row_2d >= global_col_2d) & (global_row_2d < S_len) & (global_col_2d < S_len)
        
        S_val = S_val * tau
        P = tl.exp(S_val - L_i[:, None])
        P = tl.where(mask, P, 0.0)
        
        dS = P * (dP - D_i[:, None]) * tau
        dS = tl.where(mask, dS, 0.0)
        
        dS_bf16 = dS.to(tl.bfloat16)
        
        dQ_acc = tl.dot(dS_bf16, K_j, dQ_acc)
        
    desc_dQ.store([off_i, 0], dQ_acc.to(tl.bfloat16))


@triton.jit
def _bwd_dkdV_kernel(
    descs_Q, descs_K, descs_V, descs_dO, descs_O, descs_dK, descs_dV,
    L_ptr, S_len, tau,
    STRIDE_L_B: tl.constexpr, STRIDE_L_H: tl.constexpr,
):
    j = tl.program_id(0)
    b_idx = tl.program_id(1)
    h_idx = tl.program_id(2)
    off_j = j * 64
    
    desc_Q = descs_Q[b_idx][h_idx]
    desc_K = descs_K[b_idx][h_idx]
    desc_V = descs_V[b_idx][h_idx]
    desc_dO = descs_dO[b_idx][h_idx]
    desc_O = descs_O[b_idx][h_idx]
    desc_dK = descs_dK[b_idx][h_idx]
    desc_dV = descs_dV[b_idx][h_idx]
    
    K_j = desc_K.load([off_j, 0])
    V_j = desc_V.load([off_j, 0])
    
    dK_acc = tl.zeros((64, 128), tl.float32)
    dV_acc = tl.zeros((64, 128), tl.float32)
    
    num_blocks = tl.cdiv(S_len, 64)
    
    col_idx = tl.arange(0, 64)
    
    for i in range(j, num_blocks):
        off_i = i * 64
        
        Q_i = desc_Q.load([off_i, 0])
        dO_i = desc_dO.load([off_i, 0])
        O_i = desc_O.load([off_i, 0])
        
        D_i = tl.sum(dO_i * O_i, axis=1) 
        
        row_idx = tl.arange(0, 64)
        global_row = off_i + row_idx
        lse_ptr = L_ptr + b_idx * STRIDE_L_B + h_idx * STRIDE_L_H + global_row
        L_i = tl.load(lse_ptr, mask=global_row < S_len, other=0.0)
        
        S_val = tl.dot(Q_i, K_j.T)
        dP = tl.dot(dO_i, V_j.T)
        
        global_row_2d = global_row[:, None]
        global_col_2d = off_j + col_idx[None, :]
        mask = (global_row_2d >= global_col_2d) & (global_row_2d < S_len) & (global_col_2d < S_len)
        
        S_val = S_val * tau
        P = tl.exp(S_val - L_i[:, None])
        P = tl.where(mask, P, 0.0)
        
        dS = P * (dP - D_i[:, None]) * tau
        dS = tl.where(mask, dS, 0.0)
        
        dS_bf16 = dS.to(tl.bfloat16)
        P_bf16 = P.to(tl.bfloat16)
        
        dV_acc = tl.dot(P_bf16.T, dO_i, dV_acc)
        dK_acc = tl.dot(dS_bf16.T, Q_i, dK_acc)
        
    desc_dK.store([off_j, 0], dK_acc.to(tl.bfloat16))
    desc_dV.store([off_j, 0], dV_acc.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    tau = 1.0 / (d ** 0.5)
    
    descs_Q = []
    descs_K = []
    descs_V = []
    descs_O = []
    descs_dO = []
    descs_dQ = []
    descs_dK = []
    descs_dV = []
    
    for b in range(B):
        descs_Q_b, descs_K_b, descs_V_b = [], [], []
        descs_O_b, descs_dO_b, descs_dQ_b = [], [], []
        descs_dK_b, descs_dV_b = [], []
        for h in range(H):
            descs_Q_b.append(TensorDescriptor.from_tensor(Q[b, h], [64, 128]))
            descs_K_b.append(TensorDescriptor.from_tensor(K[b, h], [64, 128]))
            descs_V_b.append(TensorDescriptor.from_tensor(V[b, h], [64, 128]))
            descs_O_b.append(TensorDescriptor.from_tensor(O[b, h], [64, 128]))
            descs_dO_b.append(TensorDescriptor.from_tensor(dO[b, h], [64, 128]))
            descs_dQ_b.append(TensorDescriptor.from_tensor(dQ[b, h], [64, 128]))
            descs_dK_b.append(TensorDescriptor.from_tensor(dK[b, h], [64, 128]))
            descs_dV_b.append(TensorDescriptor.from_tensor(dV[b, h], [64, 128]))
        descs_Q.append(descs_Q_b)
        descs_K.append(descs_K_b)
        descs_V.append(descs_V_b)
        descs_O.append(descs_O_b)
        descs_dO.append(descs_dO_b)
        descs_dQ.append(descs_dQ_b)
        descs_dK.append(descs_dK_b)
        descs_dV.append(descs_dV_b)
        
    grid = (triton.cdiv(S, 64), B, H)
    
    STRIDE_L_B = H * S
    STRIDE_L_H = S
    
    _bwd_dq_kernel[grid](
        descs_Q, descs_K, descs_V, descs_dO, descs_O, descs_dQ,
        L, S, tau,
        STRIDE_L_B, STRIDE_L_H,
        num_warps=8,
        num_stages=2,
    )
    
    _bwd_dkdV_kernel[grid](
        descs_Q, descs_K, descs_V, descs_dO, descs_O, descs_dK, descs_dV,
        L, S, tau,
        STRIDE_L_B, STRIDE_L_H,
        num_warps=8,
        num_stages=2,
    )