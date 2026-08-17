import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dQ,
    L_ptr, S_len, H, tau,
    STRIDE_L_B: tl.constexpr, STRIDE_L_H: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
):
    i = tl.program_id(0)
    bh_idx = tl.program_id(1)
    b_idx = bh_idx // H
    off_i = i * BLOCK_SIZE_S
    
    Q_i = tl.reshape(desc_Q.load([bh_idx, off_i, 0]), [64, 128])
    dO_i = tl.reshape(desc_dO.load([bh_idx, off_i, 0]), [64, 128])
    O_i = tl.reshape(desc_O.load([bh_idx, off_i, 0]), [64, 128])
    
    D_i = tl.sum(dO_i * O_i, axis=1)  
    
    row_idx = tl.arange(0, BLOCK_SIZE_S)
    lse_ptr = L_ptr + b_idx * STRIDE_L_B + (off_i + row_idx)
    L_i = tl.load(lse_ptr, mask=(off_i + row_idx) < S_len, other=0.0)  
    
    dQ_acc = tl.zeros((BLOCK_SIZE_S, 128), tl.float32)
    
    num_blocks = tl.cdiv(S_len, BLOCK_SIZE_S)
    
    for j in range(0, i + 1):
        off_j = j * BLOCK_SIZE_S
        
        K_j = tl.reshape(desc_K.load([bh_idx, off_j, 0]), [64, 128])
        V_j = tl.reshape(desc_V.load([bh_idx, off_j, 0]), [64, 128])
        
        S_val = tl.dot(Q_i, K_j.T)
        dP = tl.dot(dO_i, V_j.T)
        
        row_idx_2d = row_idx[:, None]
        col_idx_2d = row_idx[None, :]
        global_row = off_i + row_idx_2d
        global_col = off_j + col_idx_2d
        mask = global_col <= global_row
        
        S_val = S_val * tau
        P = tl.exp(S_val - L_i[:, None])
        P = tl.where(mask, P, 0.0)
        
        dS = P * (dP - D_i[:, None]) * tau
        dS = tl.where(mask, dS, 0.0)
        
        dQ_acc = tl.dot(dS, K_j, dQ_acc)
        
    dQ_acc_3d = tl.reshape(dQ_acc, [1, 64, 128])
    desc_dQ.store([bh_idx, off_i, 0], dQ_acc_3d.to(tl.bfloat16))


@triton.jit
def _bwd_dkdV_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dK, desc_dV,
    L_ptr, S_len, H, tau,
    STRIDE_L_B: tl.constexpr, STRIDE_L_H: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
):
    j = tl.program_id(0)
    bh_idx = tl.program_id(1)
    b_idx = bh_idx // H
    off_j = j * BLOCK_SIZE_S
    
    K_j = tl.reshape(desc_K.load([bh_idx, off_j, 0]), [64, 128])
    V_j = tl.reshape(desc_V.load([bh_idx, off_j, 0]), [64, 128])
    
    dK_acc = tl.zeros((BLOCK_SIZE_S, 128), tl.float32)
    dV_acc = tl.zeros((BLOCK_SIZE_S, 128), tl.float32)
    
    num_blocks = tl.cdiv(S_len, BLOCK_SIZE_S)
    
    for i in range(j, num_blocks):
        off_i = i * BLOCK_SIZE_S
        
        Q_i = tl.reshape(desc_Q.load([bh_idx, off_i, 0]), [64, 128])
        dO_i = tl.reshape(desc_dO.load([bh_idx, off_i, 0]), [64, 128])
        O_i = tl.reshape(desc_O.load([bh_idx, off_i, 0]), [64, 128])
        
        D_i = tl.sum(dO_i * O_i, axis=1) 
        
        row_idx = tl.arange(0, BLOCK_SIZE_S)
        lse_ptr = L_ptr + b_idx * STRIDE_L_B + (off_i + row_idx)
        L_i = tl.load(lse_ptr, mask=(off_i + row_idx) < S_len, other=0.0)
        
        S_val = tl.dot(Q_i, K_j.T)
        dP = tl.dot(dO_i, V_j.T)
        
        row_idx_2d = row_idx[:, None]
        col_idx_2d = row_idx[None, :]
        global_row = off_i + row_idx_2d
        global_col = off_j + col_idx_2d
        mask = global_col <= global_row
        
        S_val = S_val * tau
        P = tl.exp(S_val - L_i[:, None])
        P = tl.where(mask, P, 0.0)
        
        dS = P * (dP - D_i[:, None]) * tau
        dS = tl.where(mask, dS, 0.0)
        
        dV_acc = tl.dot(P.T, dO_i, dV_acc)
        dK_acc = tl.dot(dS.T, Q_i, dK_acc)
        
    dK_acc_3d = tl.reshape(dK_acc, [1, 64, 128])
    desc_dK.store([bh_idx, off_j, 0], dK_acc_3d.to(tl.bfloat16))
    
    dV_acc_3d = tl.reshape(dV_acc, [1, 64, 128])
    desc_dV.store([bh_idx, off_j, 0], dV_acc_3d.to(tl.bfloat16))


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
    
    desc_Q = TensorDescriptor.from_tensor(Q_3d, [1, 64, 128])
    desc_K = TensorDescriptor.from_tensor(K_3d, [1, 64, 128])
    desc_V = TensorDescriptor.from_tensor(V_3d, [1, 64, 128])
    desc_O = TensorDescriptor.from_tensor(O_3d, [1, 64, 128])
    desc_dO = TensorDescriptor.from_tensor(dO_3d, [1, 64, 128])
    desc_dQ = TensorDescriptor.from_tensor(dQ_3d, [1, 64, 128])
    desc_dK = TensorDescriptor.from_tensor(dK_3d, [1, 64, 128])
    desc_dV = TensorDescriptor.from_tensor(dV_3d, [1, 64, 128])
    
    grid = (triton.cdiv(S, 64), B * H)
    
    STRIDE_L_B = H * S
    STRIDE_L_H = S
    
    _bwd_dq_kernel[grid](
        desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dQ,
        L, S, H, tau,
        STRIDE_L_B, STRIDE_L_H,
        BLOCK_SIZE_S=64,
        num_warps=8,
        num_stages=2,
    )
    
    _bwd_dkdV_kernel[grid](
        desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dK, desc_dV,
        L, S, H, tau,
        STRIDE_L_B, STRIDE_L_H,
        BLOCK_SIZE_S=64,
        num_warps=8,
        num_stages=2,
    )