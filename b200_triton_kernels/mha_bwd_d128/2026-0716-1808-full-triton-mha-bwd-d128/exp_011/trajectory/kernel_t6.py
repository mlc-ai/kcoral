import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S, d, scale,
):
    """Compute gradients w.r.t keys and values."""
    j = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    seq_idx_j = j * 64
    
    row = tl.arange(0, 64)[:, None]
    col = tl.arange(0, 64)[None, :]
    
    K_j0 = tl.load(K_ptr + (bh_idx * S + seq_idx_j) * d + col, mask=(seq_idx_j + row < S)[:, None], other=0.0)
    K_j1 = tl.load(K_ptr + (bh_idx * S + seq_idx_j) * d + col + 64, mask=(seq_idx_j + row < S)[:, None], other=0.0)
    V_j0 = tl.load(V_ptr + (bh_idx * S + seq_idx_j) * d + col, mask=(seq_idx_j + row < S)[:, None], other=0.0)
    V_j1 = tl.load(V_ptr + (bh_idx * S + seq_idx_j) * d + col + 64, mask=(seq_idx_j + row < S)[:, None], other=0.0)
    
    acc_dK0 = tl.zeros((64, 64), dtype=tl.float32)
    acc_dK1 = tl.zeros((64, 64), dtype=tl.float32)
    acc_dV0 = tl.zeros((64, 64), dtype=tl.float32)
    acc_dV1 = tl.zeros((64, 64), dtype=tl.float32)
    
    num_blocks = (S + 63) // 64
    
    Q_next0 = tl.zeros((64, 64), dtype=tl.bfloat16)
    Q_next1 = tl.zeros((64, 64), dtype=tl.bfloat16)
    dO_next0 = tl.zeros((64, 64), dtype=tl.bfloat16)
    dO_next1 = tl.zeros((64, 64), dtype=tl.bfloat16)
    O_next0 = tl.zeros((64, 64), dtype=tl.bfloat16)
    O_next1 = tl.zeros((64, 64), dtype=tl.bfloat16)
    
    if 0 < num_blocks:
        seq_idx_next = 0
        Q_next0 = tl.load(Q_ptr + (bh_idx * S + seq_idx_next) * d + col, mask=(seq_idx_next + row < S)[:, None], other=0.0)
        Q_next1 = tl.load(Q_ptr + (bh_idx * S + seq_idx_next) * d + col + 64, mask=(seq_idx_next + row < S)[:, None], other=0.0)
        dO_next0 = tl.load(dO_ptr + (bh_idx * S + seq_idx_next) * d + col, mask=(seq_idx_next + row < S)[:, None], other=0.0)
        dO_next1 = tl.load(dO_ptr + (bh_idx * S + seq_idx_next) * d + col + 64, mask=(seq_idx_next + row < S)[:, None], other=0.0)
        O_next0 = tl.load(O_ptr + (bh_idx * S + seq_idx_next) * d + col, mask=(seq_idx_next + row < S)[:, None], other=0.0)
        O_next1 = tl.load(O_ptr + (bh_idx * S + seq_idx_next) * d + col + 64, mask=(seq_idx_next + row < S)[:, None], other=0.0)
        
    for i in range(num_blocks):
        seq_idx_i = i * 64
        
        if i + 1 < num_blocks:
            seq_idx_next = (i + 1) * 64
            Q_next0 = tl.load(Q_ptr + (bh_idx * S + seq_idx_next) * d + col, mask=(seq_idx_next + row < S)[:, None], other=0.0)
            Q_next1 = tl.load(Q_ptr + (bh_idx * S + seq_idx_next) * d + col + 64, mask=(seq_idx_next + row < S)[:, None], other=0.0)
            dO_next0 = tl.load(dO_ptr + (bh_idx * S + seq_idx_next) * d + col, mask=(seq_idx_next + row < S)[:, None], other=0.0)
            dO_next1 = tl.load(dO_ptr + (bh_idx * S + seq_idx_next) * d + col + 64, mask=(seq_idx_next + row < S)[:, None], other=0.0)
            O_next0 = tl.load(O_ptr + (bh_idx * S + seq_idx_next) * d + col, mask=(seq_idx_next + row < S)[:, None], other=0.0)
            O_next1 = tl.load(O_ptr + (bh_idx * S + seq_idx_next) * d + col + 64, mask=(seq_idx_next + row < S)[:, None], other=0.0)
            
        Q_curr0 = Q_next0
        Q_curr1 = Q_next1
        dO_curr0 = dO_next0
        dO_curr1 = dO_next1
        O_curr0 = O_next0
        O_curr1 = O_next1
        
        D_i = tl.sum(O_curr0 * dO_curr0 + O_curr1 * dO_curr1, axis=1)
        
        row_idx = tl.arange(0, 64)
        L_i = tl.load(L_ptr + (bh_idx * S + seq_idx_i) + row_idx, mask=(seq_idx_i + row_idx < S), other=-float('inf'))
        
        acc_S = tl.dot(Q_curr0, K_j0.T) + tl.dot(Q_curr1, K_j1.T)
        acc_dP = tl.dot(dO_curr0, V_j0.T) + tl.dot(dO_curr1, V_j1.T)
        
        P_ij = tl.exp(acc_S * scale - L_i[:, None])
        
        row_q = tl.arange(0, 64)[:, None]
        col_k = tl.arange(0, 64)[None, :]
        mask_ij = ((seq_idx_i + row_q < S) & (seq_idx_j + col_k < S))
        P_ij = P_ij * mask_ij
        
        dS_ij = P_ij * (acc_dP - D_i[:, None]) * scale
        
        acc_dV0 = tl.dot(P_ij.T, dO_curr0, acc_dV0)
        acc_dV1 = tl.dot(P_ij.T, dO_curr1, acc_dV1)
        
        acc_dK0 = tl.dot(dS_ij.T, Q_curr0, acc_dK0)
        acc_dK1 = tl.dot(dS_ij.T, Q_curr1, acc_dK1)
        
    ptrs_dK0 = dK_ptr + (bh_idx * S + seq_idx_j) * d + col
    ptrs_dK1 = dK_ptr + (bh_idx * S + seq_idx_j) * d + col + 64
    ptrs_dV0 = dV_ptr + (bh_idx * S + seq_idx_j) * d + col
    ptrs_dV1 = dV_ptr + (bh_idx * S + seq_idx_j) * d + col + 64
    
    mask_j = (seq_idx_j + row < S)[:, None]
    tl.store(ptrs_dK0, acc_dK0.to(tl.bfloat16), mask=mask_j)
    tl.store(ptrs_dK1, acc_dK1.to(tl.bfloat16), mask=mask_j)
    tl.store(ptrs_dV0, acc_dV0.to(tl.bfloat16), mask=mask_j)
    tl.store(ptrs_dV1, acc_dV1.to(tl.bfloat16), mask=mask_j)


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, d, scale,
):
    """Compute gradient w.r.t queries."""
    i = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    seq_idx_i = i * 64
    
    row = tl.arange(0, 64)[:, None]
    col = tl.arange(0, 64)[None, :]
    
    Q_i0 = tl.load(Q_ptr + (bh_idx * S + seq_idx_i) * d + col, mask=(seq_idx_i + row < S)[:, None], other=0.0)
    Q_i1 = tl.load(Q_ptr + (bh_idx * S + seq_idx_i) * d + col + 64, mask=(seq_idx_i + row < S)[:, None], other=0.0)
    dO_i0 = tl.load(dO_ptr + (bh_idx * S + seq_idx_i) * d + col, mask=(seq_idx_i + row < S)[:, None], other=0.0)
    dO_i1 = tl.load(dO_ptr + (bh_idx * S + seq_idx_i) * d + col + 64, mask=(seq_idx_i + row < S)[:, None], other=0.0)
    O_i0 = tl.load(O_ptr + (bh_idx * S + seq_idx_i) * d + col, mask=(seq_idx_i + row < S)[:, None], other=0.0)
    O_i1 = tl.load(O_ptr + (bh_idx * S + seq_idx_i) * d + col + 64, mask=(seq_idx_i + row < S)[:, None], other=0.0)
    
    D_i = tl.sum(O_i0 * dO_i0 + O_i1 * dO_i1, axis=1)
    
    row_idx = tl.arange(0, 64)
    L_i = tl.load(L_ptr + (bh_idx * S + seq_idx_i) + row_idx, mask=(seq_idx_i + row_idx < S), other=-float('inf'))
    
    acc_dQ0 = tl.zeros((64, 64), dtype=tl.float32)
    acc_dQ1 = tl.zeros((64, 64), dtype=tl.float32)
    
    num_blocks = (S + 63) // 64
    
    for j in range(num_blocks):
        seq_idx_j = j * 64
        
        K_j0 = tl.load(K_ptr + (bh_idx * S + seq_idx_j) * d + col, mask=(seq_idx_j + row < S)[:, None], other=0.0)
        K_j1 = tl.load(K_ptr + (bh_idx * S + seq_idx_j) * d + col + 64, mask=(seq_idx_j + row < S)[:, None], other=0.0)
        V_j0 = tl.load(V_ptr + (bh_idx * S + seq_idx_j) * d + col, mask=(seq_idx_j + row < S)[:, None], other=0.0)
        V_j1 = tl.load(V_ptr + (bh_idx * S + seq_idx_j) * d + col + 64, mask=(seq_idx_j + row < S)[:, None], other=0.0)
        
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
        
    ptrs_dQ0 = dQ_ptr + (bh_idx * S + seq_idx_i) * d + col
    ptrs_dQ1 = dQ_ptr + (bh_idx * S + seq_idx_i) * d + col + 64
    
    mask_i = (seq_idx_i + row < S)[:, None]
    tl.store(ptrs_dQ0, acc_dQ0.to(tl.bfloat16), mask=mask_i)
    tl.store(ptrs_dQ1, acc_dQ1.to(tl.bfloat16), mask=mask_i)


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
    
    Q_ptr = Q.flatten().contiguous()
    K_ptr = K.flatten().contiguous()
    V_ptr = V.flatten().contiguous()
    O_ptr = O.flatten().contiguous()
    dO_ptr = dO.flatten().contiguous()
    
    dQ_ptr = dQ.flatten().contiguous()
    dK_ptr = dK.flatten().contiguous()
    dV_ptr = dV.flatten().contiguous()
    
    L_ptr = L.flatten()
    
    grid_dk_dv = ((S + 63) // 64, B * H)
    _bwd_dk_dv_kernel[grid_dk_dv](
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dK_ptr, dV_ptr, S, d, scale,
        num_warps=4, num_stages=2
    )
    
    grid_dq = ((S + 63) // 64, B * H)
    _bwd_dq_kernel[grid_dq](
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, S, d, scale,
        num_warps=4, num_stages=2
    )