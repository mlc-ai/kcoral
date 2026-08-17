import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dQ,
    L_ptr, S_len, B, H, tau,
    STRIDE_L_B: tl.constexpr, STRIDE_L_H: tl.constexpr,
):
    i = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    bh_idx = b_idx * H + h_idx
    off_i = i * 128
    
    Q_i = desc_Q.load([bh_idx, off_i, 0])
    dO_i = desc_dO.load([bh_idx, off_i, 0])
    O_i = desc_O.load([bh_idx, off_i, 0])
    
    D_i = tl.sum(dO_i * O_i, axis=1, keepdims=False)  
    
    row_idx = tl.arange(0, 128)
    lse_ptr = L_ptr + b_idx * STRIDE_L_B + h_idx * STRIDE_L_H + (off_i + row_idx)
    L_i = tl.load(lse_ptr, mask=(off_i + row_idx) < S_len, other=0.0)  
    
    dQ_acc = tl.zeros((128, 128), tl.float32)
    
    num_blocks = tl.cdiv(S_len, 128)
    
    for j in range(0, i + 1):
        off_j = j * 128
        
        K_j = desc_K.load([bh_idx, off_j, 0])
        V_j = desc_V.load([bh_idx, off_j, 0])
        
        S_val = tl.dot(Q_i, K_j.T)
        dP = tl.dot(dO_i, V_j.T)
        
        row_idx_2d = row_idx[:, None]
        col_idx_2d = row_idx[None, :]
        global_row = off_i + row_idx_2d
        global_col = off_j + col_idx_2d
        mask = (global_row >= global_col) & (global_row < S_len) & (global_col < S_len)
        
        S_val = S_val * tau
        P = tl.exp(S_val - L_i[:, None])
        P = tl.where(mask, P, 0.0)
        
        dS = P * (dP - D_i[:, None]) * tau
        dS = tl.where(mask, dS, 0.0)
        
        dS_bf16 = dS.to(tl.bfloat16)
        
        dQ_acc = tl.dot(dS_bf16, K_j, dQ_acc)
        
    desc_dQ.store([bh_idx, off_i, 0], dQ_acc.to(tl.bfloat16))


@triton.jit
def _bwd_dkdV_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dK, desc_dV,
    L_ptr, S_len, B, H, tau,
    STRIDE_L_B: tl.constexpr, STRIDE_L_H: tl.constexpr,
):
    j = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    bh_idx = b_idx * H + h_idx
    off_j = j * 128
    
    K_j = desc_K.load([bh_idx, off_j, 0])
    V_j = desc_V.load([bh_idx, off_j, 0])
    
    dK_acc = tl.zeros((128, 128), tl.float32)
    dV_acc = tl.zeros((128, 128), tl.float32)
    
    num_blocks = tl.cdiv(S_len, 128)
    
    row_idx = tl.arange(0, 128)
    
    for i in range(j, num_blocks):
        off_i = i * 128
        
        Q_i = desc_Q.load([bh_idx, off_i, 0])
        dO_i = desc_dO.load([bh_idx, off_i, 0])
        O_i = desc_O.load([bh_idx, off_i, 0])
        
        D_i = tl.sum(dO_i * O_i, axis=1, keepdims=False) 
        
        lse_ptr = L_ptr + b_idx * STRIDE_L_B + h_idx * STRIDE_L_H + (off_i + row_idx)
        L_i = tl.load(lse_ptr, mask=(off_i + row_idx) < S_len, other=0.0)
        
        S_val = tl.dot(Q_i, K_j.T)
        dP = tl.dot(dO_i, V_j.T)
        
        row_idx_2d = row_idx[:, None]
        col_idx_2d = row_idx[None, :]
        global_row = off_i + row_idx_2d
        global_col = off_j + col_idx_2d
        mask = (global_row >= global_col) & (global_row < S_len) & (global_col < S_len)
        
        S_val = S_val * tau
        P = tl.exp(S_val - L_i[:, None])
        P = tl.where(mask, P, 0.0)
        
        dS = P * (dP - D_i[:, None]) * tau
        dS = tl.where(mask, dS, 0.0)
        
        dS_bf16 = dS.to(tl.bfloat16)
        P_bf16 = P.to(tl.bfloat16)
        
        dV_acc = tl.dot(P_bf16.T, dO_i, dV_acc)
        
        dK_acc = tl.dot(dS_bf16.T, Q_i, dK_acc)
        
    desc_dK.store([bh_idx, off_j, 0], dK_acc.to(tl.bfloat16))
    
    desc_dV.store([bh_idx, off_j, 0], dV_acc.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    tau = 1.0 / (d ** 0.5)
    
    Q_3d = Q.view(B * H, S, d)
    K_3d = K.view(B * H, S, d)
    V_3d = V.view(B * H, S, d)
    O_3d = O.view(B * H, S, d)
    dO_3d = dO.view(B * H, S, d)
    dQ_3d = dQ.view(B * H, S, d)
    dK_3d = dK.view(B * H, S, d)
    dV_3d = dV.view(B * H, S, d)
    
    desc_Q = TensorDescriptor.from_tensor(Q_3d, [1, 128, 128])
    desc_K = TensorDescriptor.from_tensor(K_3d, [1, 128, 128])
    desc_V = TensorDescriptor.from_tensor(V_3d, [1, 128, 128])
    desc_O = TensorDescriptor.from_tensor(O_3d, [1, 128, 128])
    desc_dO = TensorDescriptor.from_tensor(dO_3d, [1, 128, 128])
    desc_dQ = TensorDescriptor.from_tensor(dQ_3d, [1, 128, 128])
    desc_dK = TensorDescriptor.from_tensor(dK_3d, [1, 128, 128])
    desc_dV = TensorDescriptor.from_tensor(dV_3d, [1, 128, 128])
    
    grid = (triton.cdiv(S, 128), H, B)
    
    STRIDE_L_B = H * S
    STRIDE_L_H = S
    
    _bwd_dq_kernel[grid](
        desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dQ,
        L, S, B, H, tau,
        STRIDE_L_B, STRIDE_L_H,
        num_warps=8,
        num_stages=2,
    )
    
    _bwd_dkdV_kernel[grid](
        desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dK, desc_dV,
        L, S, B, H, tau,
        STRIDE_L_B, STRIDE_L_H,
        num_warps=8,
        num_stages=2,
    )