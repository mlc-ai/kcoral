import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dQ,
    L_ptr, S_len, H, tau,
    STRIDE_L_B: tl.constexpr, STRIDE_L_H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    bh_idx = tl.program_id(1)
    b_idx = bh_idx // H
    h_idx = bh_idx % H
    off_i = i * BLOCK
    
    Q_0 = desc_Q.load([bh_idx, off_i, 0])
    Q_1 = desc_Q.load([bh_idx, off_i, 64])
    dO_0 = desc_dO.load([bh_idx, off_i, 0])
    dO_1 = desc_dO.load([bh_idx, off_i, 64])
    O_0 = desc_O.load([bh_idx, off_i, 0])
    O_1 = desc_O.load([bh_idx, off_i, 64])
    
    D_i = tl.sum(dO_0 * O_0 + dO_1 * O_1, axis=-1, keep_dims=True)  
    
    row_idx = tl.arange(0, BLOCK)
    lse_ptr = L_ptr + b_idx * STRIDE_L_B + h_idx * STRIDE_L_H + (off_i + row_idx)
    L_i = tl.load(lse_ptr, mask=(off_i + row_idx) < S_len, other=0.0)  
    
    dQ_acc_0 = tl.zeros((1, 64, 64), tl.float32)
    dQ_acc_1 = tl.zeros((1, 64, 64), tl.float32)
    
    num_blocks = tl.cdiv(S_len, BLOCK)
    
    for j in range(0, i + 1):
        off_j = j * BLOCK
        
        K_0 = desc_K.load([bh_idx, off_j, 0])
        K_1 = desc_K.load([bh_idx, off_j, 64])
        V_0 = desc_V.load([bh_idx, off_j, 0])
        V_1 = desc_V.load([bh_idx, off_j, 64])
        
        S_val = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        global_row = (off_i + row_idx)[:, None]
        global_col = (off_j + row_idx)[None, :]
        mask = (global_row >= global_col) & (global_row < S_len) & (global_col < S_len)
        
        S_val = S_val * tau
        P = tl.exp(S_val - L_i[:, None])
        P = tl.where(mask, P, 0.0)
        
        dS = P * (dP - D_i) * tau
        dS = tl.where(mask, dS, 0.0)
        
        dS_bf16 = dS.to(tl.bfloat16)
        
        dQ_acc_0 = tl.dot(dS_bf16, K_0, dQ_acc_0)
        dQ_acc_1 = tl.dot(dS_bf16, K_1, dQ_acc_1)
        
    desc_dQ.store([bh_idx, off_i, 0], dQ_acc_0.to(tl.bfloat16))
    desc_dQ.store([bh_idx, off_i, 64], dQ_acc_1.to(tl.bfloat16))


@triton.jit
def _bwd_dkdV_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dK, desc_dV,
    L_ptr, S_len, H, tau,
    STRIDE_L_B: tl.constexpr, STRIDE_L_H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    j = tl.program_id(0)
    bh_idx = tl.program_id(1)
    b_idx = bh_idx // H
    h_idx = bh_idx % H
    off_j = j * BLOCK
    
    K_0 = desc_K.load([bh_idx, off_j, 0])
    K_1 = desc_K.load([bh_idx, off_j, 64])
    V_0 = desc_V.load([bh_idx, off_j, 0])
    V_1 = desc_V.load([bh_idx, off_j, 64])
    
    dK_acc_0 = tl.zeros((1, 64, 64), tl.float32)
    dK_acc_1 = tl.zeros((1, 64, 64), tl.float32)
    dV_acc_0 = tl.zeros((1, 64, 64), tl.float32)
    dV_acc_1 = tl.zeros((1, 64, 64), tl.float32)
    
    num_blocks = tl.cdiv(S_len, BLOCK)
    
    row_idx = tl.arange(0, BLOCK)
    
    for i in range(j, num_blocks):
        off_i = i * BLOCK
        
        Q_0 = desc_Q.load([bh_idx, off_i, 0])
        Q_1 = desc_Q.load([bh_idx, off_i, 64])
        dO_0 = desc_dO.load([bh_idx, off_i, 0])
        dO_1 = desc_dO.load([bh_idx, off_i, 64])
        O_0 = desc_O.load([bh_idx, off_i, 0])
        O_1 = desc_O.load([bh_idx, off_i, 64])
        
        D_i = tl.sum(dO_0 * O_0 + dO_1 * O_1, axis=-1, keep_dims=True) 
        
        lse_ptr = L_ptr + b_idx * STRIDE_L_B + h_idx * STRIDE_L_H + (off_i + row_idx)
        L_i = tl.load(lse_ptr, mask=(off_i + row_idx) < S_len, other=0.0)
        
        S_val = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        global_row = (off_i + row_idx)[:, None]
        global_col = (off_j + row_idx)[None, :]
        mask = (global_row >= global_col) & (global_row < S_len) & (global_col < S_len)
        
        S_val = S_val * tau
        P = tl.exp(S_val - L_i[:, None])
        P = tl.where(mask, P, 0.0)
        
        dS = P * (dP - D_i) * tau
        dS = tl.where(mask, dS, 0.0)
        
        dS_bf16 = dS.to(tl.bfloat16)
        P_bf16 = P.to(tl.bfloat16)
        
        dV_acc_0 = tl.dot(P_bf16.T, dO_0, dV_acc_0)
        dV_acc_1 = tl.dot(P_bf16.T, dO_1, dV_acc_1)
        
        dK_acc_0 = tl.dot(dS_bf16.T, Q_0, dK_acc_0)
        dK_acc_1 = tl.dot(dS_bf16.T, Q_1, dK_acc_1)
        
    desc_dK.store([bh_idx, off_j, 0], dK_acc_0.to(tl.bfloat16))
    desc_dK.store([bh_idx, off_j, 64], dK_acc_1.to(tl.bfloat16))
    
    desc_dV.store([bh_idx, off_j, 0], dV_acc_0.to(tl.bfloat16))
    desc_dV.store([bh_idx, off_j, 64], dV_acc_1.to(tl.bfloat16))


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
    
    desc_Q = TensorDescriptor.from_tensor(Q_3d, [1, 64, 64])
    desc_K = TensorDescriptor.from_tensor(K_3d, [1, 64, 64])
    desc_V = TensorDescriptor.from_tensor(V_3d, [1, 64, 64])
    desc_O = TensorDescriptor.from_tensor(O_3d, [1, 64, 64])
    desc_dO = TensorDescriptor.from_tensor(dO_3d, [1, 64, 64])
    desc_dQ = TensorDescriptor.from_tensor(dQ_3d, [1, 64, 64])
    desc_dK = TensorDescriptor.from_tensor(dK_3d, [1, 64, 64])
    desc_dV = TensorDescriptor.from_tensor(dV_3d, [1, 64, 64])
    
    grid = (triton.cdiv(S, 64), B * H)
    
    STRIDE_L_B = H * S
    STRIDE_L_H = S
    
    _bwd_dq_kernel[grid](
        desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dQ,
        L, S, H, tau,
        STRIDE_L_B, STRIDE_L_H,
        BLOCK=64,
        num_warps=8,
        num_stages=2,
    )
    
    _bwd_dkdV_kernel[grid](
        desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dK, desc_dV,
        L, S, H, tau,
        STRIDE_L_B, STRIDE_L_H,
        BLOCK=64,
        num_warps=8,
        num_stages=2,
    )