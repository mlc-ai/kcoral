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
        
    K_j0 = tl.squeeze(K_desc.load([b_h, j * 64, 0])).to(tl.float32)
    K_j1 = tl.squeeze(K_desc.load([b_h, j * 64, 64])).to(tl.float32)
    V_j0 = tl.squeeze(V_desc.load([b_h, j * 64, 0])).to(tl.float32)
    V_j1 = tl.squeeze(V_desc.load([b_h, j * 64, 64])).to(tl.float32)
    
    dK_acc0 = tl.zeros((64, 64), tl.float32)
    dK_acc1 = tl.zeros((64, 64), tl.float32)
    dV_acc0 = tl.zeros((64, 64), tl.float32)
    dV_acc1 = tl.zeros((64, 64), tl.float32)
    
    row_indices = tl.arange(0, 64)
    
    for i in range(j, NUM_BLOCKS):
        Q_i0 = tl.squeeze(Q_desc.load([b_h, i * 64, 0])).to(tl.float32)
        Q_i1 = tl.squeeze(Q_desc.load([b_h, i * 64, 64])).to(tl.float32)
        O_i0 = tl.squeeze(O_desc.load([b_h, i * 64, 0])).to(tl.float32)
        O_i1 = tl.squeeze(O_desc.load([b_h, i * 64, 64])).to(tl.float32)
        dO_i0 = tl.squeeze(dO_desc.load([b_h, i * 64, 0])).to(tl.float32)
        dO_i1 = tl.squeeze(dO_desc.load([b_h, i * 64, 64])).to(tl.float32)
        
        l_mask = (i * 64 + row_indices) < S_len
        L_i = tl.load(L_ptr + b_h * S_len + i * 64 + row_indices, mask=l_mask, other=0.0)
        L_i = L_i.to(tl.float32)
        
        D_i = tl.sum(dO_i0 * O_i0 + dO_i1 * O_i1, axis=1)
        
        S = tl.zeros((64, 64), tl.float32)
        S = tl.dot(Q_i0, K_j0.T, S)
        S = tl.dot(Q_i1, K_j1.T, S)
        
        global_row = i * 64 + row_indices[:, None]
        global_col = j * 64 + row_indices[None, :]
        valid = (global_row >= global_col) & (global_row < S_len) & (global_col < S_len)
        S = tl.where(valid, S, -float('inf'))
        
        P = tl.exp(S * sqrt_d - L_i[:, None])
        
        dV_acc0 = tl.dot(P.T, dO_i0, dV_acc0)
        dV_acc1 = tl.dot(P.T, dO_i1, dV_acc1)
        
        dS_raw = tl.zeros((64, 64), tl.float32)
        dS_raw = tl.dot(dO_i0, V_j0.T, dS_raw)
        dS_raw = tl.dot(dO_i1, V_j1.T, dS_raw)
        
        dS = P * (dS_raw - D_i[:, None]) * sqrt_d
        
        dS = tl.where(valid, dS, 0.0)
        
        dK_acc0 = tl.dot(dS.T, Q_i0, dK_acc0)
        dK_acc1 = tl.dot(dS.T, Q_i1, dK_acc1)
    
    dK_desc.store([b_h, j * 64, 0], dK_acc0.to(tl.bfloat16))
    dK_desc.store([b_h, j * 64, 64], dK_acc1.to(tl.bfloat16))
    dV_desc.store([b_h, j * 64, 0], dV_acc0.to(tl.bfloat16))
    dV_desc.store([b_h, j * 64, 64], dV_acc1.to(tl.bfloat16))


@triton.jit
def _kernel_dQ(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc, L_ptr,
    S_len, sqrt_d,
    NUM_BLOCKS: tl.constexpr,
):
    i = tl.program_id(0)
    b_h = tl.program_id(1)
    
    if i >= NUM_BLOCKS: return
        
    Q_i0 = tl.squeeze(Q_desc.load([b_h, i * 64, 0])).to(tl.float32)
    Q_i1 = tl.squeeze(Q_desc.load([b_h, i * 64, 64])).to(tl.float32)
    O_i0 = tl.squeeze(O_desc.load([b_h, i * 64, 0])).to(tl.float32)
    O_i1 = tl.squeeze(O_desc.load([b_h, i * 64, 64])).to(tl.float32)
    dO_i0 = tl.squeeze(dO_desc.load([b_h, i * 64, 0])).to(tl.float32)
    dO_i1 = tl.squeeze(dO_desc.load([b_h, i * 64, 64])).to(tl.float32)
    
    row_indices = tl.arange(0, 64)
    
    l_mask = (i * 64 + row_indices) < S_len
    L_i = tl.load(L_ptr + b_h * S_len + i * 64 + row_indices, mask=l_mask, other=0.0)
    L_i = L_i.to(tl.float32)
    
    D_i = tl.sum(dO_i0 * O_i0 + dO_i1 * O_i1, axis=1)
    
    dQ_acc0 = tl.zeros((64, 64), tl.float32)
    dQ_acc1 = tl.zeros((64, 64), tl.float32)
    
    for j in range(0, i + 1):
        K_j0 = tl.squeeze(K_desc.load([b_h, j * 64, 0])).to(tl.float32)
        K_j1 = tl.squeeze(K_desc.load([b_h, j * 64, 64])).to(tl.float32)
        V_j0 = tl.squeeze(V_desc.load([b_h, j * 64, 0])).to(tl.float32)
        V_j1 = tl.squeeze(V_desc.load([b_h, j * 64, 64])).to(tl.float32)
        
        S = tl.zeros((64, 64), tl.float32)
        S = tl.dot(Q_i0, K_j0.T, S)
        S = tl.dot(Q_i1, K_j1.T, S)
        
        global_row = i * 64 + row_indices[:, None]
        global_col = j * 64 + row_indices[None, :]
        valid = (global_row >= global_col) & (global_row < S_len) & (global_col < S_len)
        S = tl.where(valid, S, -float('inf'))
        
        P = tl.exp(S * sqrt_d - L_i[:, None])
        
        dS_raw = tl.zeros((64, 64), tl.float32)
        dS_raw = tl.dot(dO_i0, V_j0.T, dS_raw)
        dS_raw = tl.dot(dO_i1, V_j1.T, dS_raw)
        
        dS = P * (dS_raw - D_i[:, None]) * sqrt_d
        dS = tl.where(valid, dS, 0.0)
        
        dQ_acc0 = tl.dot(dS, K_j0, dQ_acc0)
        dQ_acc1 = tl.dot(dS, K_j1, dQ_acc1)
    
    dQ_desc.store([b_h, i * 64, 0], dQ_acc0.to(tl.bfloat16))
    dQ_desc.store([b_h, i * 64, 64], dQ_acc1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, d_dim = Q.shape
    
    sqrt_d = 1.0 / math.sqrt(d_dim)
    
    # Split into smaller contiguous blocks to bypass 2D TMA limitations (<= 65536 elements)
    Q_3d = Q.view(B * H, S_len, 2, 64)
    K_3d = K.view(B * H, S_len, 2, 64)
    V_3d = V.view(B * H, S_len, 2, 64)
    O_3d = O.view(B * H, S_len, 2, 64)
    dO_3d = dO.view(B * H, S_len, 2, 64)
    dQ_3d = dQ.view(B * H, S_len, 2, 64)
    dK_3d = dK.view(B * H, S_len, 2, 64)
    dV_3d = dV.view(B * H, S_len, 2, 64)
    
    Q_desc = TensorDescriptor.from_tensor(Q_3d, [1, 64, 1, 64])
    K_desc = TensorDescriptor.from_tensor(K_3d, [1, 64, 1, 64])
    V_desc = TensorDescriptor.from_tensor(V_3d, [1, 64, 1, 64])
    O_desc = TensorDescriptor.from_tensor(O_3d, [1, 64, 1, 64])
    dO_desc = TensorDescriptor.from_tensor(dO_3d, [1, 64, 1, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ_3d, [1, 64, 1, 64])
    dK_desc = TensorDescriptor.from_tensor(dK_3d, [1, 64, 1, 64])
    dV_desc = TensorDescriptor.from_tensor(dV_3d, [1, 64, 1, 64])
    
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