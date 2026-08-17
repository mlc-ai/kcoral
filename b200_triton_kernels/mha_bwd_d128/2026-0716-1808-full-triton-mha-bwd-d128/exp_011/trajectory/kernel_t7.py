import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dk_dv_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
    L_ptr, S, d, scale,
):
    """Compute gradients w.r.t keys and values."""
    j = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    seq_idx_j = j * 64
    row_offset_j = bh_idx * S + seq_idx_j
    
    K_j0 = desc_K.load([row_offset_j, 0])
    K_j1 = desc_K.load([row_offset_j, 64])
    V_j0 = desc_V.load([row_offset_j, 0])
    V_j1 = desc_V.load([row_offset_j, 64])
    
    acc_dK0 = tl.zeros((64, 64), dtype=tl.float32)
    acc_dK1 = tl.zeros((64, 64), dtype=tl.float32)
    acc_dV0 = tl.zeros((64, 64), dtype=tl.float32)
    acc_dV1 = tl.zeros((64, 64), dtype=tl.float32)
    
    num_blocks = (S + 63) // 64
    
    for i in range(num_blocks):
        seq_idx_i = i * 64
        row_offset_i = bh_idx * S + seq_idx_i
        
        Q_i0 = desc_Q.load([row_offset_i, 0])
        Q_i1 = desc_Q.load([row_offset_i, 64])
        O_i0 = desc_O.load([row_offset_i, 0])
        O_i1 = desc_O.load([row_offset_i, 64])
        dO_i0 = desc_dO.load([row_offset_i, 0])
        dO_i1 = desc_dO.load([row_offset_i, 64])
        
        D_i = tl.sum(O_i0 * dO_i0 + O_i1 * dO_i1, axis=1)
        
        row_idx = tl.arange(0, 64)
        L_i = tl.load(L_ptr + (bh_idx * S + seq_idx_i) + row_idx, mask=(seq_idx_i + row_idx < S), other=-float('inf'))
        
        acc_S = tl.dot(Q_i0, K_j0.T) + tl.dot(Q_i1, K_j1.T)
        acc_dP = tl.dot(dO_i0, V_j0.T) + tl.dot(dO_i1, V_j1.T)
        
        P_ij = tl.exp(acc_S * scale - L_i[:, None])
        
        row_q = tl.arange(0, 64)[:, None]
        col_k = tl.arange(0, 64)[None, :]
        mask_ij = ((seq_idx_i + row_q < S) & (seq_idx_j + col_k < S))
        P_ij = P_ij * mask_ij
        
        dS_ij = P_ij * (acc_dP - D_i[:, None]) * scale
        
        acc_dV0 = tl.dot(P_ij.T, dO_i0, acc_dV0)
        acc_dV1 = tl.dot(P_ij.T, dO_i1, acc_dV1)
        
        acc_dK0 = tl.dot(dS_ij.T, Q_i0, acc_dK0)
        acc_dK1 = tl.dot(dS_ij.T, Q_i1, acc_dK1)
        
    desc_dK.store([row_offset_j, 0], acc_dK0.to(tl.bfloat16))
    desc_dK.store([row_offset_j, 64], acc_dK1.to(tl.bfloat16))
    desc_dV.store([row_offset_j, 0], acc_dV0.to(tl.bfloat16))
    desc_dV.store([row_offset_j, 64], acc_dV1.to(tl.bfloat16))


@triton.jit
def _bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dQ,
    L_ptr, S, d, scale,
):
    """Compute gradient w.r.t queries."""
    i = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    seq_idx_i = i * 64
    row_offset_i = bh_idx * S + seq_idx_i
    
    Q_i0 = desc_Q.load([row_offset_i, 0])
    Q_i1 = desc_Q.load([row_offset_i, 64])
    dO_i0 = desc_dO.load([row_offset_i, 0])
    dO_i1 = desc_dO.load([row_offset_i, 64])
    O_i0 = desc_O.load([row_offset_i, 0])
    O_i1 = desc_O.load([row_offset_i, 64])
    
    D_i = tl.sum(O_i0 * dO_i0 + O_i1 * dO_i1, axis=1)
    
    row_idx = tl.arange(0, 64)
    L_i = tl.load(L_ptr + (bh_idx * S + seq_idx_i) + row_idx, mask=(seq_idx_i + row_idx < S), other=-float('inf'))
    
    acc_dQ0 = tl.zeros((64, 64), dtype=tl.float32)
    acc_dQ1 = tl.zeros((64, 64), dtype=tl.float32)
    
    num_blocks = (S + 63) // 64
    
    for j in range(num_blocks):
        seq_idx_j = j * 64
        row_offset_j = bh_idx * S + seq_idx_j
        
        K_j0 = desc_K.load([row_offset_j, 0])
        K_j1 = desc_K.load([row_offset_j, 64])
        V_j0 = desc_V.load([row_offset_j, 0])
        V_j1 = desc_V.load([row_offset_j, 64])
        
        acc_S = tl.dot(Q_i0, K_j0.T) + tl.dot(Q_i1, K_j1.T)
        acc_dP = tl.dot(dO_i0, V_j0.T) + tl.dot(dO_i1, V_j1.T)
        
        P_ij = tl.exp(acc_S * scale - L_i[:, None])
        
        row_q = tl.arange(0, 64)[:, None]
        col_k = tl.arange(0, 64)[None, :]
        mask_ij = ((seq_idx_i + row_q < S) & (seq_idx_j + col_k < S))
        P_ij = P_ij * mask_ij
        
        dS_ij = P_ij * (acc_dP - D_i[:, None]) * scale
        
        acc_dQ0 = tl.dot(dS_ij, K_j0, acc_dQ0)
        acc_dQ1 = tl.dot(dS_ij, K_j1, acc_dQ1)
        
    desc_dQ.store([row_offset_i, 0], acc_dQ0.to(tl.bfloat16))
    desc_dQ.store([row_offset_i, 64], acc_dQ1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Backward pass for multi-head attention targeting Hopper architectures.
    Computes exact numerical gradients mapped over destination-passing buffers.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    assert B == 4 and H == 48 and d == 128
    assert Q.dtype == torch.bfloat16
    assert K.dtype == torch.bfloat16
    assert V.dtype == torch.bfloat16
    assert O.dtype == torch.bfloat16
    assert dO.dtype == torch.bfloat16
    assert L.dtype == torch.float32
    
    scale = 1.0 / math.sqrt(d)
    
    Q_ = Q.view(B * H, S, d).reshape(-1, d)
    K_ = K.view(B * H, S, d).reshape(-1, d)
    V_ = V.view(B * H, S, d).reshape(-1, d)
    O_ = O.view(B * H, S, d).reshape(-1, d)
    dO_ = dO.view(B * H, S, d).reshape(-1, d)
    dQ_ = dQ.view(B * H, S, d).reshape(-1, d)
    dK_ = dK.view(B * H, S, d).reshape(-1, d)
    dV_ = dV.view(B * H, S, d).reshape(-1, d)
    
    desc_Q = TensorDescriptor.from_tensor(Q_, [64, 64])
    desc_K = TensorDescriptor.from_tensor(K_, [64, 64])
    desc_V = TensorDescriptor.from_tensor(V_, [64, 64])
    desc_O = TensorDescriptor.from_tensor(O_, [64, 64])
    desc_dO = TensorDescriptor.from_tensor(dO_, [64, 64])
    desc_dQ = TensorDescriptor.from_tensor(dQ_, [64, 64])
    desc_dK = TensorDescriptor.from_tensor(dK_, [64, 64])
    desc_dV = TensorDescriptor.from_tensor(dV_, [64, 64])
    
    L_ptr = L.flatten()
    
    grid_dk_dv = ((S + 63) // 64, B * H)
    _bwd_dk_dv_kernel[grid_dk_dv](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
        L_ptr, S, d, scale,
        num_warps=4, num_stages=2
    )
    
    grid_dq = ((S + 63) // 64, B * H)
    _bwd_dq_kernel[grid_dq](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dQ,
        L_ptr, S, d, scale,
        num_warps=4, num_stages=2
    )