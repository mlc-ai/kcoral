import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dQ,
    L_ptr, S_len, H, tau,
    STRIDE_L_H: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
):
    i = tl.program_id(0)
    bh_idx = tl.program_id(1)
    off_i = i * BLOCK_SIZE_S
    
    Q_i_0 = desc_Q.load([bh_idx, off_i, 0])
    Q_i_1 = desc_Q.load([bh_idx, off_i, 64])
    dO_i_0 = desc_dO.load([bh_idx, off_i, 0])
    dO_i_1 = desc_dO.load([bh_idx, off_i, 64])
    O_i_0 = desc_O.load([bh_idx, off_i, 0])
    O_i_1 = desc_O.load([bh_idx, off_i, 64])
    
    D_i = tl.sum(dO_i_0 * O_i_0 + dO_i_1 * O_i_1, axis=1)  
    
    row_idx = tl.arange(0, BLOCK_SIZE_S)
    lse_ptr = L_ptr + bh_idx * STRIDE_L_H + (off_i + row_idx)
    L_i = tl.load(lse_ptr, mask=(off_i + row_idx) < S_len, other=0.0)  
    
    dQ_acc_0 = tl.zeros((BLOCK_SIZE_S, 64), tl.float32)
    dQ_acc_1 = tl.zeros((BLOCK_SIZE_S, 64), tl.float32)
    
    num_blocks = tl.cdiv(S_len, BLOCK_SIZE_S)
    
    for j in range(0, i + 1):
        off_j = j * BLOCK_SIZE_S
        
        K_j_0 = desc_K.load([bh_idx, off_j, 0])
        K_j_1 = desc_K.load([bh_idx, off_j, 64])
        V_j_0 = desc_V.load([bh_idx, off_j, 0])
        V_j_1 = desc_V.load([bh_idx, off_j, 64])
        
        S_val = tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)
        dP = tl.dot(dO_i_0, V_j_0.T) + tl.dot(dO_i_1, V_j_1.T)
        
        if i > j:
            mask = tl.full((128, 128), True, dtype=tl.int1)
        elif i == j:
            idx = tl.arange(0, 128)
            mask = (idx[None, :] <= idx[:, None]).to(tl.int1)
        else:
            mask = tl.full((128, 128), False, dtype=tl.int1)
            
        if i == num_blocks - 1:
            row_idx_2d = row_idx[:, None]
            boundary_mask = (row_idx_2d < S_len) & (~row_idx_2d)
            mask = mask & boundary_mask
        if j == num_blocks - 1:
            col_idx_2d = row_idx[None, :]
            boundary_mask = (col_idx_2d < S_len) & (~col_idx_2d)
            mask = mask & boundary_mask
        
        S_val = S_val * tau
        P = tl.exp(S_val - L_i[:, None])
        P = tl.where(mask, P, 0.0)
        
        dS = P * (dP - D_i[:, None]) * tau
        dS = tl.where(mask, dS, 0.0)
        
        dS_bf16 = dS.to(tl.bfloat16)
        
        dQ_acc_0 = tl.dot(dS_bf16, K_j_0, dQ_acc_0)
        dQ_acc_1 = tl.dot(dS_bf16, K_j_1, dQ_acc_1)
        
    desc_dQ.store([bh_idx, off_i, 0], dQ_acc_0.to(tl.bfloat16))
    desc_dQ.store([bh_idx, off_i, 64], dQ_acc_1.to(tl.bfloat16))


@triton.jit
def _bwd_dkdV_kernel(
    desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dK, desc_dV,
    L_ptr, S_len, H, tau,
    STRIDE_L_H: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
):
    j = tl.program_id(0)
    bh_idx = tl.program_id(1)
    off_j = j * BLOCK_SIZE_S
    
    K_j_0 = desc_K.load([bh_idx, off_j, 0])
    K_j_1 = desc_K.load([bh_idx, off_j, 64])
    V_j_0 = desc_V.load([bh_idx, off_j, 0])
    V_j_1 = desc_V.load([bh_idx, off_j, 64])
    
    dK_acc_0 = tl.zeros((BLOCK_SIZE_S, 64), tl.float32)
    dK_acc_1 = tl.zeros((BLOCK_SIZE_S, 64), tl.float32)
    dV_acc_0 = tl.zeros((BLOCK_SIZE_S, 64), tl.float32)
    dV_acc_1 = tl.zeros((BLOCK_SIZE_S, 64), tl.float32)
    
    num_blocks = tl.cdiv(S_len, BLOCK_SIZE_S)
    
    for i in range(j, num_blocks):
        off_i = i * BLOCK_SIZE_S
        
        Q_i_0 = desc_Q.load([bh_idx, off_i, 0])
        Q_i_1 = desc_Q.load([bh_idx, off_i, 64])
        dO_i_0 = desc_dO.load([bh_idx, off_i, 0])
        dO_i_1 = desc_dO.load([bh_idx, off_i, 64])
        O_i_0 = desc_O.load([bh_idx, off_i, 0])
        O_i_1 = desc_O.load([bh_idx, off_i, 64])
        
        D_i = tl.sum(dO_i_0 * O_i_0 + dO_i_1 * O_i_1, axis=1) 
        
        row_idx = tl.arange(0, BLOCK_SIZE_S)
        lse_ptr = L_ptr + bh_idx * STRIDE_L_H + (off_i + row_idx)
        L_i = tl.load(lse_ptr, mask=(off_i + row_idx) < S_len, other=0.0)
        
        S_val = tl.dot(Q_i_0, K_j_0.T) + tl.dot(Q_i_1, K_j_1.T)
        dP = tl.dot(dO_i_0, V_j_0.T) + tl.dot(dO_i_1, V_j_1.T)
        
        if i > j:
            mask = tl.full((128, 128), True, dtype=tl.int1)
        elif i == j:
            idx = tl.arange(0, 128)
            mask = (idx[None, :] <= idx[:, None]).to(tl.int1)
        else:
            mask = tl.full((128, 128), False, dtype=tl.int1)
            
        if i == num_blocks - 1:
            row_idx_2d = row_idx[:, None]
            boundary_mask = (row_idx_2d < S_len) & (~row_idx_2d)
            mask = mask & boundary_mask
        if j == num_blocks - 1:
            col_idx_2d = row_idx[None, :]
            boundary_mask = (col_idx_2d < S_len) & (~col_idx_2d)
            mask = mask & boundary_mask
        
        S_val = S_val * tau
        P = tl.exp(S_val - L_i[:, None])
        P = tl.where(mask, P, 0.0)
        
        dS = P * (dP - D_i[:, None]) * tau
        dS = tl.where(mask, dS, 0.0)
        
        dS_bf16 = dS.to(tl.bfloat16)
        P_bf16 = P.to(tl.bfloat16)
        
        dV_acc_0 = tl.dot(P_bf16.T, dO_i_0, dV_acc_0)
        dV_acc_1 = tl.dot(P_bf16.T, dO_i_1, dV_acc_1)
        
        dK_acc_0 = tl.dot(dS_bf16.T, Q_i_0, dK_acc_0)
        dK_acc_1 = tl.dot(dS_bf16.T, Q_i_1, dK_acc_1)
        
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
    
    desc_Q = TensorDescriptor.from_tensor(Q_3d, [1, 128, 64])
    desc_K = TensorDescriptor.from_tensor(K_3d, [1, 128, 64])
    desc_V = TensorDescriptor.from_tensor(V_3d, [1, 128, 64])
    desc_O = TensorDescriptor.from_tensor(O_3d, [1, 128, 64])
    desc_dO = TensorDescriptor.from_tensor(dO_3d, [1, 128, 64])
    desc_dQ = TensorDescriptor.from_tensor(dQ_3d, [1, 128, 64])
    desc_dK = TensorDescriptor.from_tensor(dK_3d, [1, 128, 64])
    desc_dV = TensorDescriptor.from_tensor(dV_3d, [1, 128, 64])
    
    grid = (triton.cdiv(S, 128), B * H)
    
    STRIDE_L_H = S
    
    _bwd_dq_kernel[grid](
        desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dQ,
        L, S, H, tau,
        STRIDE_L_H,
        BLOCK_SIZE_S=128,
        num_warps=4,
        num_stages=2,
    )
    
    _bwd_dkdV_kernel[grid](
        desc_Q, desc_K, desc_V, desc_dO, desc_O, desc_dK, desc_dV,
        L, S, H, tau,
        STRIDE_L_H,
        BLOCK_SIZE_S=128,
        num_warps=4,
        num_stages=2,
    )