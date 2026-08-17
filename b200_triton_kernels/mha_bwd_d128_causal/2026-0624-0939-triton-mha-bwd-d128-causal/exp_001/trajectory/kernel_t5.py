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
    
    k_offset = b_h * S_len + j * 128
    K_j = K_desc.load([k_offset, 0]).to(tl.float32)
    V_j = V_desc.load([k_offset, 0]).to(tl.float32)
    
    acc_dK = tl.zeros((128, 128), tl.float32)
    acc_dV = tl.zeros((128, 128), tl.float32)
    
    row_indices = tl.arange(0, 128)
    
    for i in range(j, NUM_BLOCKS):
        q_offset = b_h * S_len + i * 128
        Q_i = Q_desc.load([q_offset, 0]).to(tl.float32)
        O_i = O_desc.load([q_offset, 0]).to(tl.float32)
        dO_i = dO_desc.load([q_offset, 0]).to(tl.float32)
        
        l_base = b_h * S_len + i * 128
        l_mask = (i * 128 + row_indices) < S_len
        L_i = tl.load(L_ptr + l_base + row_indices, mask=l_mask, other=0.0)
        L_i = L_i.to(tl.float32)
        
        D_i = tl.sum(dO_i * O_i, axis=1)
        
        S = tl.dot(Q_i, K_j.T)
        
        global_row = i * 128 + row_indices[:, None]
        global_col = j * 128 + row_indices[None, :]
        valid = (global_row >= global_col) & (global_row < S_len) & (global_col < S_len)
        
        P = tl.exp(S * sqrt_d - L_i[:, None])
        P = P * valid
        
        acc_dV = tl.dot(P.T, dO_i, acc_dV)
        
        dS_raw = tl.dot(dO_i, V_j.T)
        
        dS = P * (dS_raw - D_i[:, None]) * sqrt_d
        
        dS = tl.where(valid, dS, 0.0)
        
        acc_dK = tl.dot(dS.T, Q_i, acc_dK)
    
    dK_desc.store([k_offset, 0], acc_dK.to(tl.bfloat16))
    dV_desc.store([k_offset, 0], acc_dV.to(tl.bfloat16))


@triton.jit
def _kernel_dQ(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc, L_ptr,
    S_len, sqrt_d,
    NUM_BLOCKS: tl.constexpr,
):
    i = tl.program_id(0)
    b_h = tl.program_id(1)
    
    if i >= NUM_BLOCKS: return
    
    q_offset = b_h * S_len + i * 128
    Q_i = Q_desc.load([q_offset, 0]).to(tl.float32)
    O_i = O_desc.load([q_offset, 0]).to(tl.float32)
    dO_i = dO_desc.load([q_offset, 0]).to(tl.float32)
    
    row_indices = tl.arange(0, 128)
    
    l_base = b_h * S_len + i * 128
    l_mask = (i * 128 + row_indices) < S_len
    L_i = tl.load(L_ptr + l_base + row_indices, mask=l_mask, other=0.0)
    L_i = L_i.to(tl.float32)
    
    D_i = tl.sum(dO_i * O_i, axis=1)
    
    acc_dQ = tl.zeros((128, 128), tl.float32)
    
    for jti in range(0, i + 1):
        k_offset = b_h * S_len + jti * 128
        K_j = K_desc.load([k_offset, 0]).to(tl.float32)
        V_j = V_desc.load([k_offset, 0]).to(tl.float32)
        
        S = tl.dot(Q_i, K_j.T)
        
        global_row = i * 128 + row_indices[:, None]
        global_col = jti * 128 + row_indices[None, :]
        valid = (global_row >= global_col) & (global_row < S_len) & (global_col < S_len)
        
        P = tl.exp(S * sqrt_d - L_i[:, None])
        P = P * valid
        
        dS_raw = tl.dot(dO_i, V_j.T)
        
        dS = P * (dS_raw - D_i[:, None]) * sqrt_d
        
        dS = tl.where(valid, dS, 0.0)
        
        acc_dQ = tl.dot(dS, K_j, acc_dQ)
    
    dQ_desc.store([q_offset, 0], acc_dQ.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, d_dim = Q.shape
    
    sqrt_d = 1.0 / math.sqrt(d_dim)
    
    Q_2d = Q.view(B * H * S_len, d_dim)
    K_2d = K.view(B * H * S_len, d_dim)
    V_2d = V.view(B * H * S_len, d_dim)
    O_2d = O.view(B * H * S_len, d_dim)
    dO_2d = dO.view(B * H * S_len, d_dim)
    dQ_2d = dQ.view(B * H * S_len, d_dim)
    dK_2d = dK.view(B * H * S_len, d_dim)
    dV_2d = dV.view(B * H * S_len, d_dim)
    
    desc = lambda t: TensorDescriptor.from_tensor(t, [128, 128])
    
    num_blocks = triton.cdiv(S_len, 128)
    grid = (num_blocks, B * H)
    
    _kernel_dKV[grid](
        desc(Q_2d), desc(K_2d), desc(V_2d), desc(O_2d), desc(dO_2d), desc(dK_2d), desc(dV_2d), L,
        S_len, sqrt_d,
        NUM_BLOCKS=num_blocks,
        num_warps=8, num_stages=3
    )
    
    _kernel_dQ[grid](
        desc(Q_2d), desc(K_2d), desc(V_2d), desc(O_2d), desc(dO_2d), desc(dQ_2d), L,
        S_len, sqrt_d,
        NUM_BLOCKS=num_blocks,
        num_warps=8, num_stages=3
    )