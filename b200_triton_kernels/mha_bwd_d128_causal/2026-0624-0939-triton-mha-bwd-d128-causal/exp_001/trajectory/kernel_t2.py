import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _kernel_dKV(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc, L_ptr,
    S_len, sqrt_d,
    NUM_BLOCKS: tl.constexpr,
):
    j = tl.program_id(0)
    b_h = tl.program_id(1)
    
    if j >= NUM_BLOCKS: return
        
    K_j = K_desc.load([b_h * S_len + j * 64, 0]).to(tl.float32)
    V_j = V_desc.load([b_h * S_len + j * 64, 0]).to(tl.float32)
    
    dK_acc = tl.zeros((64, 128), tl.float32)
    dV_acc = tl.zeros((64, 128), tl.float32)
    
    row_indices = tl.arange(0, 64)
    col_indices = tl.arange(0, 64)
    
    for i in range(j, NUM_BLOCKS):
        Q_i = Q_desc.load([b_h * S_len + i * 64, 0]).to(tl.float32)
        O_i = O_desc.load([b_h * S_len + i * 64, 0]).to(tl.float32)
        dO_i = dO_desc.load([b_h * S_len + i * 64, 0]).to(tl.float32)
        
        l_mask = (i * 64 + row_indices) < S_len
        L_i = tl.load(L_ptr + b_h * S_len + i * 64 + row_indices, mask=l_mask, other=0.0)
        L_i = L_i.to(tl.float32)
        
        D_i = tl.sum(dO_i * O_i, axis=1)
        
        S = tl.dot(Q_i, K_j.T)
        
        P = tl.exp(S * sqrt_d - L_i[:, None])
        
        global_row = i * 64 + row_indices[:, None]
        global_col = j * 64 + col_indices[None, :]
        valid = (global_row >= global_col) & (global_row < S_len) & (global_col < S_len)
        P = P * valid
        
        dV_acc = tl.dot(P.T, dO_i, dV_acc)
        
        dS_raw = tl.dot(dO_i, V_j.T)
        
        dS = P * (dS_raw - D_i[:, None]) * sqrt_d
        
        dS = tl.where(valid, dS, 0.0)
        
        dK_acc = tl.dot(dS.T, Q_i, dK_acc)
    
    dK_desc.store([b_h * S_len + j * 64, 0], dK_acc.to(tl.bfloat16))
    dV_desc.store([b_h * S_len + j * 64, 0], dV_acc.to(tl.bfloat16))


@triton.jit
def _kernel_dQ(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc, L_ptr,
    S_len, sqrt_d,
    NUM_BLOCKS: tl.constexpr,
):
    i = tl.program_id(0)
    b_h = tl.program_id(1)
    
    if i >= NUM_BLOCKS: return
        
    Q_i = Q_desc.load([b_h * S_len + i * 64, 0]).to(tl.float32)
    O_i = O_desc.load([b_h * S_len + i * 64, 0]).to(tl.float32)
    dO_i = dO_desc.load([b_h * S_len + i * 64, 0]).to(tl.float32)
    
    row_indices = tl.arange(0, 64)
    
    l_mask = (i * 64 + row_indices) < S_len
    L_i = tl.load(L_ptr + b_h * S_len + i * 64 + row_indices, mask=l_mask, other=0.0)
    L_i = L_i.to(tl.float32)
    
    D_i = tl.sum(dO_i * O_i, axis=1)
    
    dQ_acc = tl.zeros((64, 128), tl.float32)
    
    col_indices = tl.arange(0, 64)
    
    for j in range(0, i + 1):
        K_j = K_desc.load([b_h * S_len + j * 64, 0]).to(tl.float32)
        V_j = V_desc.load([b_h * S_len + j * 64, 0]).to(tl.float32)
        
        S = tl.dot(Q_i, K_j.T)
        
        P = tl.exp(S * sqrt_d - L_i[:, None])
        
        global_row = i * 64 + row_indices[:, None]
        global_col = j * 64 + col_indices[None, :]
        valid = (global_row >= global_col) & (global_row < S_len) & (global_col < S_len)
        P = P * valid
        
        dS_raw = tl.dot(dO_i, V_j.T)
        
        dS = P * (dS_raw - D_i[:, None]) * sqrt_d
        
        dS = tl.where(valid, dS, 0.0)
        
        dQ_acc = tl.dot(dS, K_j, dQ_acc)
    
    dQ_desc.store([b_h * S_len + i * 64, 0], dQ_acc.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, d_dim = Q.shape
    
    sqrt_d = 1.0 / math.sqrt(d_dim)
    
    Q_3d = Q.view(B * H * S_len, d_dim)
    K_3d = K.view(B * H * S_len, d_dim)
    V_3d = V.view(B * H * S_len, d_dim)
    O_3d = O.view(B * H * S_len, d_dim)
    dO_3d = dO.view(B * H * S_len, d_dim)
    dQ_3d = dQ.view(B * H * S_len, d_dim)
    dK_3d = dK.view(B * H * S_len, d_dim)
    dV_3d = dV.view(B * H * S_len, d_dim)
    
    Q_desc = TensorDescriptor.from_tensor(Q_3d, [64, 128])
    K_desc = TensorDescriptor.from_tensor(K_3d, [64, 128])
    V_desc = TensorDescriptor.from_tensor(V_3d, [64, 128])
    O_desc = TensorDescriptor.from_tensor(O_3d, [64, 128])
    dO_desc = TensorDescriptor.from_tensor(dO_3d, [64, 128])
    dQ_desc = TensorDescriptor.from_tensor(dQ_3d, [64, 128])
    dK_desc = TensorDescriptor.from_tensor(dK_3d, [64, 128])
    dV_desc = TensorDescriptor.from_tensor(dV_3d, [64, 128])
    
    num_blocks = triton.cdiv(S_len, 64)
    grid = (num_blocks, B * H)
    
    _kernel_dKV[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc, L,
        S_len, sqrt_d,
        NUM_BLOCKS=num_blocks,
        num_warps=8, num_stages=3
    )
    
    _kernel_dQ[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc, L,
        S_len, sqrt_d,
        NUM_BLOCKS=num_blocks,
        num_warps=8, num_stages=3
    )