import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _kernel_dKV(
    Q_desc_0, Q_desc_1, 
    K_desc_0, K_desc_1, 
    V_desc_0, V_desc_1, 
    O_desc_0, O_desc_1, 
    dO_desc_0, dO_desc_1, 
    dK_desc_0, dK_desc_1, 
    dV_desc_0, dV_desc_1, 
    L_ptr,
    S_len, sqrt_d,
    NUM_BLOCKS: tl.constexpr,
):
    j = tl.program_id(0)
    b_h = tl.program_id(1)
    
    if j >= NUM_BLOCKS: return
        
    K_j0 = K_desc_0.load([b_h * S_len + j * 64, 0]).to(tl.float32)
    K_j1 = K_desc_1.load([b_h * S_len + j * 64, 0]).to(tl.float32)
    V_j0 = V_desc_0.load([b_h * S_len + j * 64, 0]).to(tl.float32)
    V_j1 = V_desc_1.load([b_h * S_len + j * 64, 0]).to(tl.float32)
    
    dK_acc0 = tl.zeros((64, 64), tl.float32)
    dK_acc1 = tl.zeros((64, 64), tl.float32)
    dV_acc0 = tl.zeros((64, 64), tl.float32)
    dV_acc1 = tl.zeros((64, 64), tl.float32)
    
    row_indices = tl.arange(0, 64)
    col_indices = tl.arange(0, 64)
    
    for i in range(j, NUM_BLOCKS):
        Q_i0 = Q_desc_0.load([b_h * S_len + i * 64, 0]).to(tl.float32)
        Q_i1 = Q_desc_1.load([b_h * S_len + i * 64, 0]).to(tl.float32)
        O_i0  = O_desc_0.load([b_h * S_len + i * 64, 0]).to(tl.float32)
        O_i1  = O_desc_1.load([b_h * S_len + i * 64, 0]).to(tl.float32)
        dO_i0 = dO_desc_0.load([b_h * S_len + i * 64, 0]).to(tl.float32)
        dO_i1 = dO_desc_1.load([b_h * S_len + i * 64, 0]).to(tl.float32)
        
        l_mask = (i * 64 + row_indices) < S_len
        L_i = tl.load(L_ptr + b_h * S_len + i * 64 + row_indices, mask=l_mask, other=0.0)
        L_i = L_i.to(tl.float32)
        
        D_i = tl.sum(dO_i0 * O_i0 + dO_i1 * O_i1, axis=1)
        
        S = tl.zeros((64, 64), tl.float32)
        S = tl.dot(Q_i0, K_j0.T, S)
        S = tl.dot(Q_i1, K_j1.T, S)
        
        global_row = i * 64 + row_indices[:, None]
        global_col = j * 64 + col_indices[None, :]
        valid = (global_row >= global_col) & (global_row < S_len) & (global_col < S_len)
        
        P = tl.exp(S * sqrt_d - L_i[:, None])
        P = P * valid
        
        dV_acc0 = tl.dot(P.T, dO_i0, dV_acc0)
        dV_acc1 = tl.dot(P.T, dO_i1, dV_acc1)
        
        dS_raw = tl.zeros((64, 64), tl.float32)
        dS_raw = tl.dot(dO_i0, V_j0.T, dS_raw)
        dS_raw = tl.dot(dO_i1, V_j1.T, dS_raw)
        
        dS = P * (dS_raw - D_i[:, None]) * sqrt_d
        
        dS = tl.where(valid, dS, 0.0)
        
        dK_acc0 = tl.dot(dS.T, Q_i0, dK_acc0)
        dK_acc1 = tl.dot(dS.T, Q_i1, dK_acc1)
    
    dK_desc_0.store([b_h * S_len + j * 64, 0], dK_acc0.to(tl.bfloat16))
    dK_desc_1.store([b_h * S_len + j * 64, 0], dK_acc1.to(tl.bfloat16))
    dV_desc_0.store([b_h * S_len + j * 64, 0], dV_acc0.to(tl.bfloat16))
    dV_desc_1.store([b_h * S_len + j * 64, 0], dV_acc1.to(tl.bfloat16))


@triton.jit
def _kernel_dQ(
    Q_desc_0, Q_desc_1, 
    K_desc_0, K_desc_1, 
    V_desc_0, V_desc_1, 
    O_desc_0, O_desc_1, 
    dO_desc_0, dO_desc_1, 
    dQ_desc_0, dQ_desc_1, 
    L_ptr,
    S_len, sqrt_d,
    NUM_BLOCKS: tl.constexpr,
):
    i = tl.program_id(0)
    b_h = tl.program_id(1)
    
    if i >= NUM_BLOCKS: return
        
    Q_i0 = Q_desc_0.load([b_h * S_len + i * 64, 0]).to(tl.float32)
    Q_i1 = Q_desc_1.load([b_h * S_len + i * 64, 0]).to(tl.float32)
    O_i0  = O_desc_0.load([b_h * S_len + i * 64, 0]).to(tl.float32)
    O_i1  = O_desc_1.load([b_h * S_len + i * 64, 0]).to(tl.float32)
    dO_i0 = dO_desc_0.load([b_h * S_len + i * 64, 0]).to(tl.float32)
    dO_i1 = dO_desc_1.load([b_h * S_len + i * 64, 0]).to(tl.float32)
    
    row_indices = tl.arange(0, 64)
    col_indices = tl.arange(0, 64)
    
    l_mask = (i * 64 + row_indices) < S_len
    L_i = tl.load(L_ptr + b_h * S_len + i * 64 + row_indices, mask=l_mask, other=0.0)
    L_i = L_i.to(tl.float32)
    
    D_i = tl.sum(dO_i0 * O_i0 + dO_i1 * O_i1, axis=1)
    
    dQ_acc0 = tl.zeros((64, 64), tl.float32)
    dQ_acc1 = tl.zeros((64, 64), tl.float32)
    
    for j in range(0, i + 1):
        K_j0 = K_desc_0.load([b_h * S_len + j * 64, 0]).to(tl.float32)
        K_j1 = K_desc_1.load([b_h * S_len + j * 64, 0]).to(tl.float32)
        V_j0 = V_desc_0.load([b_h * S_len + j * 64, 0]).to(tl.float32)
        V_j1 = V_desc_1.load([b_h * S_len + j * 64, 0]).to(tl.float32)
        
        S = tl.zeros((64, 64), tl.float32)
        S = tl.dot(Q_i0, K_j0.T, S)
        S = tl.dot(Q_i1, K_j1.T, S)
        
        global_row = i * 64 + row_indices[:, None]
        global_col = j * 64 + col_indices[None, :]
        valid = (global_row >= global_col) & (global_row < S_len) & (global_col < S_len)
        
        P = tl.exp(S * sqrt_d - L_i[:, None])
        P = P * valid
        
        dS_raw = tl.zeros((64, 64), tl.float32)
        dS_raw = tl.dot(dO_i0, V_j0.T, dS_raw)
        dS_raw = tl.dot(dO_i1, V_j1.T, dS_raw)
        
        dS = P * (dS_raw - D_i[:, None]) * sqrt_d
        
        dS = tl.where(valid, dS, 0.0)
        
        dQ_acc0 = tl.dot(dS, K_j0, dQ_acc0)
        dQ_acc1 = tl.dot(dS, K_j1, dQ_acc1)
    
    dQ_desc_0.store([b_h * S_len + i * 64, 0], dQ_acc0.to(tl.bfloat16))
    dQ_desc_1.store([b_h * S_len + i * 64, 0], dQ_acc1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, d_dim = Q.shape
    
    sqrt_d = 1.0 / math.sqrt(d_dim)
    
    Q_3d_0 = Q[:, :, :, :64].contiguous().view(B * H * S_len, 64)
    Q_3d_1 = Q[:, :, :, 64:].contiguous().view(B * H * S_len, 64)
    K_3d_0 = K[:, :, :, :64].contiguous().view(B * H * S_len, 64)
    K_3d_1 = K[:, :, :, 64:].contiguous().view(B * H * S_len, 64)
    V_3d_0 = V[:, :, :, :64].contiguous().view(B * H * S_len, 64)
    V_3d_1 = V[:, :, :, 64:].contiguous().view(B * H * S_len, 64)
    O_3d_0 = O[:, :, :, :64].contiguous().view(B * H * S_len, 64)
    O_3d_1 = O[:, :, :, 64:].contiguous().view(B * H * S_len, 64)
    dO_3d_0 = dO[:, :, :, :64].contiguous().view(B * H * S_len, 64)
    dO_3d_1 = dO[:, :, :, 64:].contiguous().view(B * H * S_len, 64)
    dQ_3d_0 = dQ[:, :, :, :64].contiguous().view(B * H * S_len, 64)
    dQ_3d_1 = dQ[:, :, :, 64:].contiguous().view(B * H * S_len, 64)
    dK_3d_0 = dK[:, :, :, :64].contiguous().view(B * H * S_len, 64)
    dK_3d_1 = dK[:, :, :, 64:].contiguous().view(B * H * S_len, 64)
    dV_3d_0 = dV[:, :, :, :64].contiguous().view(B * H * S_len, 64)
    dV_3d_1 = dV[:, :, :, 64:].contiguous().view(B * H * S_len, 64)
    
    desc_flat = lambda t: TensorDescriptor.from_tensor(t, [64, 64], padding_option="zero")
    
    num_blocks = triton.cdiv(S_len, 64)
    grid = (num_blocks, B * H)
    
    _kernel_dKV[grid](
        desc_flat(Q_3d_0), desc_flat(Q_3d_1),
        desc_flat(K_3d_0), desc_flat(K_3d_1),
        desc_flat(V_3d_0), desc_flat(V_3d_1),
        desc_flat(O_3d_0), desc_flat(O_3d_1),
        desc_flat(dO_3d_0), desc_flat(dO_3d_1),
        desc_flat(dK_3d_0), desc_flat(dK_3d_1),
        desc_flat(dV_3d_0), desc_flat(dV_3d_1),
        L, S_len, sqrt_d,
        NUM_BLOCKS=num_blocks,
        num_warps=8, num_stages=3
    )
    
    _kernel_dQ[grid](
        desc_flat(Q_3d_0), desc_flat(Q_3d_1),
        desc_flat(K_3d_0), desc_flat(K_3d_1),
        desc_flat(V_3d_0), desc_flat(V_3d_1),
        desc_flat(O_3d_0), desc_flat(O_3d_1),
        desc_flat(dO_3d_0), desc_flat(dO_3d_1),
        desc_flat(dQ_3d_0), desc_flat(dQ_3d_1),
        L, S_len, sqrt_d,
        NUM_BLOCKS=num_blocks,
        num_warps=8, num_stages=3
    )