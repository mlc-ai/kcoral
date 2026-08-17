import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr, desc_dQ,
    S_len, d, scale,
    BLOCK: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    
    offset_m = m_idx * BLOCK
    mask_m = (offset_m + tl.arange(0, BLOCK)) < S_len
    
    Q_m_0 = desc_Q.load([b_h_idx, offset_m, 0])
    Q_m_1 = desc_Q.load([b_h_idx, offset_m, 64])
    dO_m_0 = desc_dO.load([b_h_idx, offset_m, 0])
    dO_m_1 = desc_dO.load([b_h_idx, offset_m, 64])
    O_m_0 = desc_O.load([b_h_idx, offset_m, 0])
    O_m_1 = desc_O.load([b_h_idx, offset_m, 64])
    
    Q_m_0 = Q_m_0.to(tl.float32)
    Q_m_1 = Q_m_1.to(tl.float32)
    dO_m_0 = dO_m_0.to(tl.float32)
    dO_m_1 = dO_m_1.to(tl.float32)
    O_m_0 = O_m_0.to(tl.float32)
    O_m_1 = O_m_1.to(tl.float32)
    
    Q_m_0 = tl.where(mask_m[:, None], Q_m_0, 0.0)
    Q_m_1 = tl.where(mask_m[:, None], Q_m_1, 0.0)
    dO_m_0 = tl.where(mask_m[:, None], dO_m_0, 0.0)
    dO_m_1 = tl.where(mask_m[:, None], dO_m_1, 0.0)
    O_m_0 = tl.where(mask_m[:, None], O_m_0, 0.0)
    O_m_1 = tl.where(mask_m[:, None], O_m_1, 0.0)
    
    D_m = (tl.sum(dO_m_0 * O_m_0, axis=1) + tl.sum(dO_m_1 * O_m_1, axis=1))[:, None]
    
    L_base = L_ptr + b_h_idx * S_len + offset_m
    L_m = tl.load(L_base + tl.arange(0, 64), mask=mask_m, other=0.0)[:, None]
    
    acc_dQ_0 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    acc_dQ_1 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    
    for n_idx in range(0, S_len, BLOCK):
        offset_n = n_idx * BLOCK
        mask_n = (offset_n + tl.arange(0, BLOCK)) < S_len
        
        K_n_0 = desc_K.load([b_h_idx, offset_n, 0]).to(tl.float32)
        K_n_1 = desc_K.load([b_h_idx, offset_n, 64]).to(tl.float32)
        V_n_0 = desc_V.load([b_h_idx, offset_n, 0]).to(tl.float32)
        V_n_1 = desc_V.load([b_h_idx, offset_n, 64]).to(tl.float32)
        
        K_n_0 = tl.where(mask_n[:, None], K_n_0, 0.0)
        K_n_1 = tl.where(mask_n[:, None], K_n_1, 0.0)
        V_n_0 = tl.where(mask_n[:, None], V_n_0, 0.0)
        V_n_1 = tl.where(mask_n[:, None], V_n_1, 0.0)
        
        S_acc = tl.dot(Q_m_0, K_n_0.T) + tl.dot(Q_m_1, K_n_1.T)
        S = S_acc * scale
        
        P = tl.exp(S - L_m)
        P = P * mask_m[:, None] * mask_n[None, :]
        
        dP_acc = tl.dot(dO_m_0, V_n_0.T) + tl.dot(dO_m_1, V_n_1.T)
        dS = P * (dP_acc - D_m) * scale
        
        acc_dQ_0 = tl.dot(dS, K_n_0, acc_dQ_0)
        acc_dQ_1 = tl.dot(dS, K_n_1, acc_dQ_1)
    
    desc_dQ.store([b_h_idx, offset_m, 0], acc_dQ_0.to(tl.bfloat16))
    desc_dQ.store([b_h_idx, offset_m, 64], acc_dQ_1.to(tl.bfloat16))


@triton.jit
def _mha_bwd_dkv_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr, desc_dK, desc_dV,
    S_len, d, scale,
    BLOCK: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    n_idx = tl.program_id(1)
    
    offset_n = n_idx * BLOCK
    mask_n = (offset_n + tl.arange(0, BLOCK)) < S_len
    
    K_n_0 = desc_K.load([b_h_idx, offset_n, 0]).to(tl.float32)
    K_n_1 = desc_K.load([b_h_idx, offset_n, 64]).to(tl.float32)
    V_n_0 = desc_V.load([b_h_idx, offset_n, 0]).to(tl.float32)
    V_n_1 = desc_V.load([b_h_idx, offset_n, 64]).to(tl.float32)
    
    K_n_0 = tl.where(mask_n[:, None], K_n_0, 0.0)
    K_n_1 = tl.where(mask_n[:, None], K_n_1, 0.0)
    V_n_0 = tl.where(mask_n[:, None], V_n_0, 0.0)
    V_n_1 = tl.where(mask_n[:, None], V_n_1, 0.0)
    
    acc_dK_0 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    acc_dK_1 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    acc_dV_0 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    acc_dV_1 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    
    for m_idx in range(0, S_len, BLOCK):
        offset_m = m_idx * BLOCK
        mask_m = (offset_m + tl.arange(0, BLOCK)) < S_len
        
        Q_m_0 = desc_Q.load([b_h_idx, offset_m, 0]).to(tl.float32)
        Q_m_1 = desc_Q.load([b_h_idx, offset_m, 64]).to(tl.float32)
        dO_m_0 = desc_dO.load([b_h_idx, offset_m, 0]).to(tl.float32)
        dO_m_1 = desc_dO.load([b_h_idx, offset_m, 64]).to(tl.float32)
        O_m_0 = desc_O.load([b_h_idx, offset_m, 0]).to(tl.float32)
        O_m_1 = desc_O.load([b_h_idx, offset_m, 64]).to(tl.float32)
        
        Q_m_0 = tl.where(mask_m[:, None], Q_m_0, 0.0)
        Q_m_1 = tl.where(mask_m[:, None], Q_m_1, 0.0)
        dO_m_0 = tl.where(mask_m[:, None], dO_m_0, 0.0)
        dO_m_1 = tl.where(mask_m[:, None], dO_m_1, 0.0)
        O_m_0 = tl.where(mask_m[:, None], O_m_0, 0.0)
        O_m_1 = tl.where(mask_m[:, None], O_m_1, 0.0)
        
        D_m = (tl.sum(dO_m_0 * O_m_0, axis=1) + tl.sum(dO_m_1 * O_m_1, axis=1))[:, None]
        
        L_base = L_ptr + b_h_idx * S_len + offset_m
        L_m = tl.load(L_base + tl.arange(0, 64), mask=mask_m, other=0.0)[:, None]
        
        S_acc = tl.dot(Q_m_0, K_n_0.T) + tl.dot(Q_m_1, K_n_1.T)
        S = S_acc * scale
        
        P = tl.exp(S - L_m)
        P = P * mask_m[:, None] * mask_n[None, :]
        
        dP_acc = tl.dot(dO_m_0, V_n_0.T) + tl.dot(dO_m_1, V_n_1.T)
        dS = P * (dP_acc - D_m) * scale
        
        acc_dV_0 = tl.dot(P.T, dO_m_0, acc_dV_0)
        acc_dV_1 = tl.dot(P.T, dO_m_1, acc_dV_1)
        acc_dK_0 = tl.dot(dS.T, Q_m_0, acc_dK_0)
        acc_dK_1 = tl.dot(dS.T, Q_m_1, acc_dK_1)
    
    desc_dK.store([b_h_idx, offset_n, 0], acc_dK_0.to(tl.bfloat16))
    desc_dK.store([b_h_idx, offset_n, 64], acc_dK_1.to(tl.bfloat16))
    desc_dV.store([b_h_idx, offset_n, 0], acc_dV_0.to(tl.bfloat16))
    desc_dV.store([b_h_idx, offset_n, 64], acc_dV_1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    Q_flat = Q.view(B * H, S, 128)
    K_flat = K.view(B * H, S, 128)
    V_flat = V.view(B * H, S, 128)
    O_flat = O.view(B * H, S, 128)
    dO_flat = dO.view(B * H, S, 128)
    dQ_flat = dQ.view(B * H, S, 128)
    dK_flat = dK.view(B * H, S, 128)
    dV_flat = dV.view(B * H, S, 128)
    L_flat = L.view(B * H, S)
    
    desc_Q = TensorDescriptor.from_tensor(Q_flat, [1, 64, 64])
    desc_K = TensorDescriptor.from_tensor(K_flat, [1, 64, 64])
    desc_V = TensorDescriptor.from_tensor(V_flat, [1, 64, 64])
    desc_O = TensorDescriptor.from_tensor(O_flat, [1, 64, 64])
    desc_dO = TensorDescriptor.from_tensor(dO_flat, [1, 64, 64])
    desc_dQ = TensorDescriptor.from_tensor(dQ_flat, [1, 64, 64])
    desc_dK = TensorDescriptor.from_tensor(dK_flat, [1, 64, 64])
    desc_dV = TensorDescriptor.from_tensor(dV_flat, [1, 64, 64])
    
    BLOCK = 64
    grid_dq = (B * H, triton.cdiv(S, BLOCK))
    _mha_bwd_dq_kernel[grid_dq](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, L_flat, desc_dQ,
        S, d, scale,
        BLOCK=BLOCK,
        num_warps=8, num_stages=3,
    )
    
    grid_dkv = (B * H, triton.cdiv(S, BLOCK))
    _mha_bwd_dkv_kernel[grid_dkv](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, L_flat, desc_dK, desc_dV,
        S, d, scale,
        BLOCK=BLOCK,
        num_warps=8, num_stages=3,
    )