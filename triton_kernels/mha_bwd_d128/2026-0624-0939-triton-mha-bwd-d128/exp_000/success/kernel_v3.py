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
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    pid_m = tl.program_id(0)
    m_start = pid_m * 64
    
    Q_tile = tl.reshape(desc_Q.load([b, h, m_start, 0]), [64, 128])
    O_tile = tl.reshape(desc_O.load([b, h, m_start, 0]), [64, 128])
    dO_tile = tl.reshape(desc_dO.load([b, h, m_start, 0]), [64, 128])
    
    l_idx = m_start + tl.arange(0, 64)
    mask_m = l_idx < S
    L_vec = tl.load(L_ptr + b * H * S + h * S + l_idx, mask=mask_m, other=0.0)
    
    D_vec = tl.sum(O_tile.to(tl.float32) * dO_tile.to(tl.float32), axis=1)
    
    acc_dQ = tl.zeros((64, 128), tl.float32)
    
    num_k_steps = tl.cdiv(S, 64)
    for k_step in range(num_k_steps):
        n_start = k_step * 64
        
        K_tile = tl.reshape(desc_K.load([b, h, n_start, 0]), [64, 128])
        V_tile = tl.reshape(desc_V.load([b, h, n_start, 0]), [64, 128])
        
        S_val = tl.dot(Q_tile, K_tile.T) * scale
        
        P = tl.exp(S_val - L_vec[:, None])
        
        k_idx = n_start + tl.arange(0, 64)
        mask = mask_m[:, None] & (k_idx[None, :] < S)
        P = P * mask
        
        dP = tl.dot(dO_tile, V_tile.T)
        
        dS = P * (dP - D_vec[:, None]) * scale
        dS = dS * mask
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dQ = tl.dot(dS_bf16, K_tile, acc_dQ)
        
    desc_dQ.store([b, h, m_start, 0], tl.reshape(acc_dQ, [1, 1, 64, 128]))


@triton.jit
def _mha_bwd_dk_dv_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
    L_ptr,
    H, S, scale,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    pid_n = tl.program_id(0)
    n_start = pid_n * 64
    
    K_tile = tl.reshape(desc_K.load([b, h, n_start, 0]), [64, 128])
    V_tile = tl.reshape(desc_V.load([b, h, n_start, 0]), [64, 128])
    
    acc_dK = tl.zeros((64, 128), tl.float32)
    acc_dV = tl.zeros((64, 128), tl.float32)
    
    num_m_steps = tl.cdiv(S, 64)
    for m_step in range(num_m_steps):
        m_start = m_step * 64
        
        Q_tile = tl.reshape(desc_Q.load([b, h, m_start, 0]), [64, 128])
        O_tile = tl.reshape(desc_O.load([b, h, m_start, 0]), [64, 128])
        dO_tile = tl.reshape(desc_dO.load([b, h, m_start, 0]), [64, 128])
        
        l_idx = m_start + tl.arange(0, 64)
        mask_m = l_idx < S
        L_vec = tl.load(L_ptr + b * H * S + h * S + l_idx, mask=mask_m, other=0.0)
        
        D_vec = tl.sum(O_tile.to(tl.float32) * dO_tile.to(tl.float32), axis=1)
        
        S_val = tl.dot(Q_tile, K_tile.T) * scale
        
        P = tl.exp(S_val - L_vec[:, None])
        
        k_idx = n_start + tl.arange(0, 64)
        mask_n = k_idx < S
        mask = mask_m[:, None] & mask_n[None, :]
        P = P * mask
        
        dP = tl.dot(dO_tile, V_tile.T)
        
        dS = P * (dP - D_vec[:, None]) * scale
        dS = dS * mask
        
        dS_bf16 = dS.to(tl.bfloat16)
        acc_dK = tl.dot(dS_bf16.T, Q_tile, acc_dK)
        
        P_bf16 = P.to(tl.bfloat16)
        acc_dV = tl.dot(P_bf16.T, dO_tile, acc_dV)
        
    desc_dK.store([b, h, n_start, 0], tl.reshape(acc_dK, [1, 1, 64, 128]))
    desc_dV.store([b, h, n_start, 0], tl.reshape(acc_dV, [1, 1, 64, 128]))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    desc_Q = TensorDescriptor.from_tensor(Q, [1, 1, 64, 128])
    desc_K = TensorDescriptor.from_tensor(K, [1, 1, 64, 128])
    desc_V = TensorDescriptor.from_tensor(V, [1, 1, 64, 128])
    desc_O = TensorDescriptor.from_tensor(O, [1, 1, 64, 128])
    desc_dO = TensorDescriptor.from_tensor(dO, [1, 1, 64, 128])
    desc_dQ = TensorDescriptor.from_tensor(dQ, [1, 1, 64, 128])
    desc_dK = TensorDescriptor.from_tensor(dK, [1, 1, 64, 128])
    desc_dV = TensorDescriptor.from_tensor(dV, [1, 1, 64, 128])
    
    num_blocks = triton.cdiv(S, 64)
    grid = (num_blocks, H, B)
    
    _mha_bwd_dq_kernel[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dQ,
        L,
        H, S, scale,
        num_warps=8, num_stages=3,
    )
    
    _mha_bwd_dk_dv_kernel[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
        L,
        H, S, scale,
        num_warps=8, num_stages=3,
    )