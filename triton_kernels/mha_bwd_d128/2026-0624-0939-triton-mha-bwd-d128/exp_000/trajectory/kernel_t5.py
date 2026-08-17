import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dQ,
    L_ptr,
    H, S, scale,
    num_k_steps: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    pid_m = tl.program_id(0)
    m_start = pid_m * 128
    
    Q_tile_4d = desc_Q.load([b, h, m_start, 0])
    Q_tile = tl.reshape(Q_tile_4d, [128, 128])
    
    O_tile_4d = desc_O.load([b, h, m_start, 0])
    O_tile = tl.reshape(O_tile_4d, [128, 128])
    
    dO_tile_4d = desc_dO.load([b, h, m_start, 0])
    dO_tile = tl.reshape(dO_tile_4d, [128, 128])
    
    l_idx = m_start + tl.arange(0, 128)
    L_vec = tl.load(L_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
    
    D_vec = tl.sum(O_tile.to(tl.float32) * dO_tile.to(tl.float32), axis=1)
    
    mask_m = l_idx < S
    
    acc_dQ = tl.zeros((128, 128), tl.float32)
    
    for step in range(num_k_steps):
        n_start = step * 128
        
        K_tile_4d = desc_K.load([b, h, n_start, 0])
        K_tile = tl.reshape(K_tile_4d, [128, 128])
        
        V_tile_4d = desc_V.load([b, h, n_start, 0])
        V_tile = tl.reshape(V_tile_4d, [128, 128])
        
        S_val = tl.dot(Q_tile, K_tile.T) * scale 
        
        P = tl.exp(S_val - L_vec[:, None])
        
        dP = tl.dot(dO_tile, V_tile.T)
        
        dS = P * (dP - D_vec[:, None]) * scale
        
        col_idx = n_start + tl.arange(0, 128)
        mask_n = col_idx < S
        mask = mask_m[:, None] & mask_n[None, :]
        P = P * mask
        dS = dS * mask
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dQ = tl.dot(dS_bf16, K_tile, acc_dQ)
        
    dQ_out_4d = tl.reshape(acc_dQ, [1, 1, 128, 128])
    desc_dQ.store([b, h, m_start, 0], dQ_out_4d)


@triton.jit
def _mha_bwd_dk_dv_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
    L_ptr,
    H, S, scale,
    num_m_steps: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    pid_n = tl.program_id(0)
    n_start = pid_n * 128
    
    K_tile_4d = desc_K.load([b, h, n_start, 0])
    K_tile = tl.reshape(K_tile_4d, [128, 128])
    
    V_tile_4d = desc_V.load([b, h, n_start, 0])
    V_tile = tl.reshape(V_tile_4d, [128, 128])
    
    acc_dK = tl.zeros((128, 128), tl.float32)
    acc_dV = tl.zeros((128, 128), tl.float32)
    
    col_idx = n_start + tl.arange(0, 128)
    mask_n = col_idx < S
    
    for step in range(num_m_steps):
        m_start = step * 64
        
        Q_tile_4d = desc_Q.load([b, h, m_start, 0])
        Q_tile = tl.reshape(Q_tile_4d, [64, 128])
        
        O_tile_4d = desc_O.load([b, h, m_start, 0])
        O_tile = tl.reshape(O_tile_4d, [64, 128])
        
        dO_tile_4d = desc_dO.load([b, h, m_start, 0])
        dO_tile = tl.reshape(dO_tile_4d, [64, 128])
        
        l_idx = m_start + tl.arange(0, 64)
        L_vec = tl.load(L_ptr + b * H * S + h * S + l_idx, mask=(l_idx < S), other=0.0)
        
        D_vec = tl.sum(O_tile.to(tl.float32) * dO_tile.to(tl.float32), axis=1)
        
        mask_m = l_idx < S
        
        S_val = tl.dot(Q_tile, K_tile.T) * scale 
        
        P = tl.exp(S_val - L_vec[:, None])
        
        dP = tl.dot(dO_tile, V_tile.T)
        
        dS = P * (dP - D_vec[:, None]) * scale
        
        mask = mask_m[:, None] & mask_n[None, :]
        P = P * mask
        dS = dS * mask
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dK = tl.dot(dS_bf16.T, Q_tile, acc_dK)
        
        P_bf16 = P.to(tl.bfloat16)
        acc_dV = tl.dot(P_bf16.T, dO_tile, acc_dV)
        
    dK_out_4d = tl.reshape(acc_dK, [1, 1, 128, 128])
    desc_dK.store([b, h, n_start, 0], dK_out_4d)
    
    dV_out_4d = tl.reshape(acc_dV, [1, 1, 128, 128])
    desc_dV.store([b, h, n_start, 0], dV_out_4d)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    desc_Q_128 = TensorDescriptor.from_tensor(Q, [1, 1, 128, 128])
    desc_K_128 = TensorDescriptor.from_tensor(K, [1, 1, 128, 128])
    desc_V_128 = TensorDescriptor.from_tensor(V, [1, 1, 128, 128])
    desc_O_128 = TensorDescriptor.from_tensor(O, [1, 1, 128, 128])
    desc_dO_128 = TensorDescriptor.from_tensor(dO, [1, 1, 128, 128])
    desc_dQ_128 = TensorDescriptor.from_tensor(dQ, [1, 1, 128, 128])
    desc_dK_128 = TensorDescriptor.from_tensor(dK, [1, 1, 128, 128])
    desc_dV_128 = TensorDescriptor.from_tensor(dV, [1, 1, 128, 128])
    
    desc_Q_64 = TensorDescriptor.from_tensor(Q, [1, 1, 64, 128])
    desc_O_64 = TensorDescriptor.from_tensor(O, [1, 1, 64, 128])
    desc_dO_64 = TensorDescriptor.from_tensor(dO, [1, 1, 64, 128])
    desc_dK_64 = TensorDescriptor.from_tensor(dK, [1, 1, 64, 128])
    desc_dV_64 = TensorDescriptor.from_tensor(dV, [1, 1, 64, 128])
    
    num_k_steps = triton.cdiv(S, 128)
    num_m_steps = triton.cdiv(S, 64)
    
    grid_dq = (triton.cdiv(S, 128), H, B)
    grid_dk = (triton.cdiv(S, 128), H, B)
    
    _mha_bwd_dq_kernel[grid_dq](
        desc_Q_128, desc_K_128, desc_V_128, desc_O_128, desc_dO_128, desc_dQ_128,
        L,
        H, S, scale,
        num_k_steps=num_k_steps,
        num_warps=8, num_stages=2,
    )
    
    _mha_bwd_dk_dv_kernel[grid_dk](
        desc_Q_64, desc_K_128, desc_V_128, desc_O_64, desc_dO_64, desc_dK_128, desc_dV_128,
        L,
        H, S, scale,
        num_m_steps=num_m_steps,
        num_warps=8, num_stages=2,
    )