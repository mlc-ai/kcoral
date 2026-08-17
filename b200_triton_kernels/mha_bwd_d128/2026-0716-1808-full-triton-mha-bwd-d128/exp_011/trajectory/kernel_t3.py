import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S, scale,
):
    """Compute gradients w.r.t keys and values."""
    j = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    seq_idx_j = j * 32
    
    row = tl.arange(0, 32)[:, None]
    col = tl.arange(0, 128)[None, :]
    
    tl.multiple_of(K_ptr, 128)
    tl.multiple_of(V_ptr, 128)
    
    row_start_j = bh_idx * S + seq_idx_j
    K_j0 = tl.load(K_ptr + (row_start_j + row) * 128 + col, mask=(seq_idx_j + row < S)[:, None], other=0.0)
    K_j1 = tl.load(K_ptr + (row_start_j + row) * 128 + col + 64, mask=(seq_idx_j + row < S)[:, None], other=0.0)
    V_j0 = tl.load(V_ptr + (row_start_j + row) * 128 + col, mask=(seq_idx_j + row < S)[:, None], other=0.0)
    V_j1 = tl.load(V_ptr + (row_start_j + row) * 128 + col + 64, mask=(seq_idx_j + row < S)[:, None], other=0.0)
    
    acc_dK0 = tl.zeros((32, 64), dtype=tl.float32)
    acc_dK1 = tl.zeros((32, 64), dtype=tl.float32)
    acc_dV0 = tl.zeros((32, 64), dtype=tl.float32)
    acc_dV1 = tl.zeros((32, 64), dtype=tl.float32)
    
    num_blocks = (S + 31) // 32
    
    for i in range(num_blocks):
        seq_idx_i = i * 32
        row_start_i = bh_idx * S + seq_idx_i
        
        tl.multiple_of(Q_ptr, 128)
        tl.multiple_of(O_ptr, 128)
        tl.multiple_of(dO_ptr, 128)
        
        Q_i0 = tl.load(Q_ptr + (row_start_i + row) * 128 + col, mask=(seq_idx_i + row < S)[:, None], other=0.0)
        Q_i1 = tl.load(Q_ptr + (row_start_i + row) * 128 + col + 64, mask=(seq_idx_i + row < S)[:, None], other=0.0)
        dO_i0 = tl.load(dO_ptr + (row_start_i + row) * 128 + col, mask=(seq_idx_i + row < S)[:, None], other=0.0)
        dO_i1 = tl.load(dO_ptr + (row_start_i + row) * 128 + col + 64, mask=(seq_idx_i + row < S)[:, None], other=0.0)
        O_i0 = tl.load(O_ptr + (row_start_i + row) * 128 + col, mask=(seq_idx_i + row < S)[:, None], other=0.0)
        O_i1 = tl.load(O_ptr + (row_start_i + row) * 128 + col + 64, mask=(seq_idx_i + row < S)[:, None], other=0.0)
        
        D_i = tl.sum(O_i0 * dO_i0 + O_i1 * dO_i1, axis=1)
        
        L_i = tl.load(L_ptr + (bh_idx * S + seq_idx_i) + tl.arange(0, 32), mask=(seq_idx_i + tl.arange(0, 32) < S), other=-float('inf'))
        
        acc_S = tl.zeros((32, 32), dtype=tl.float32)
        acc_dP = tl.zeros((32, 32), dtype=tl.float32)
        
        acc_S = tl.dot(Q_i0, K_j0.T, acc_S)
        acc_S = tl.dot(Q_i1, K_j1.T, acc_S)
        
        acc_dP = tl.dot(dO_i0, V_j0.T, acc_dP)
        acc_dP = tl.dot(dO_i1, V_j1.T, acc_dP)
        
        P_ij = tl.exp(acc_S * scale - L_i[:, None])
        
        row_q = tl.arange(0, 32)
        col_k = tl.arange(0, 32)[None, :]
        mask_i = (seq_idx_i + row_q < S)[:, None]
        P_ij = P_ij * mask_i
        
        dS_ij = P_ij * (acc_dP - D_i[:, None]) * scale
        
        acc_dV0 = tl.dot(P_ij.T, dO_i0, acc_dV0)
        acc_dV1 = tl.dot(P_ij.T, dO_i1, acc_dV1)
        
        acc_dK0 = tl.dot(dS_ij.T, Q_i0, acc_dK0)
        acc_dK1 = tl.dot(dS_ij.T, Q_i1, acc_dK1)
        
    ptrs_dK0 = dK_ptr + (row_start_j + row) * 128 + col
    ptrs_dK1 = dK_ptr + (row_start_j + row) * 128 + col + 64
    ptrs_dV0 = dV_ptr + (row_start_j + row) * 128 + col
    ptrs_dV1 = dV_ptr + (row_start_j + row) * 128 + col + 64
    
    mask_j = (seq_idx_j + row < S)[:, None]
    tl.store(ptrs_dK0, acc_dK0.to(tl.bfloat16), mask=mask_j)
    tl.store(ptrs_dK1, acc_dK1.to(tl.bfloat16), mask=mask_j)
    tl.store(ptrs_dV0, acc_dV0.to(tl.bfloat16), mask=mask_j)
    tl.store(ptrs_dV1, acc_dV1.to(tl.bfloat16), mask=mask_j)


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, scale,
):
    """Compute gradient w.r.t queries."""
    i = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    seq_idx_i = i * 32
    
    row = tl.arange(0, 32)[:, None]
    col = tl.arange(0, 128)[None, :]
    
    row_start_i = bh_idx * S + seq_idx_i
    
    tl.multiple_of(Q_ptr, 128)
    tl.multiple_of(O_ptr, 128)
    tl.multiple_of(dO_ptr, 128)
    
    Q_i0 = tl.load(Q_ptr + (row_start_i + row) * 128 + col, mask=(seq_idx_i + row < S)[:, None], other=0.0)
    Q_i1 = tl.load(Q_ptr + (row_start_i + row) * 128 + col + 64, mask=(seq_idx_i + row < S)[:, None], other=0.0)
    dO_i0 = tl.load(dO_ptr + (row_start_i + row) * 128 + col, mask=(seq_idx_i + row < S)[:, None], other=0.0)
    dO_i1 = tl.load(dO_ptr + (row_start_i + row) * 128 + col + 64, mask=(seq_idx_i + row < S)[:, None], other=0.0)
    O_i0 = tl.load(O_ptr + (row_start_i + row) * 128 + col, mask=(seq_idx_i + row < S)[:, None], other=0.0)
    O_i1 = tl.load(O_ptr + (row_start_i + row) * 128 + col + 64, mask=(seq_idx_i + row < S)[:, None], other=0.0)
    
    D_i = tl.sum(O_i0 * dO_i0 + O_i1 * dO_i1, axis=1)
    L_i = tl.load(L_ptr + (bh_idx * S + seq_idx_i) + tl.arange(0, 32), mask=(seq_idx_i + tl.arange(0, 32) < S), other=-float('inf'))
    
    acc_dQ0 = tl.zeros((32, 64), dtype=tl.float32)
    acc_dQ1 = tl.zeros((32, 64), dtype=tl.float32)
    
    num_blocks = (S + 31) // 32
    
    for j in range(num_blocks):
        seq_idx_j = j * 32
        row_start_j = bh_idx * S + seq_idx_j
        
        tl.multiple_of(K_ptr, 128)
        tl.multiple_of(V_ptr, 128)
        
        K_j0 = tl.load(K_ptr + (row_start_j + row) * 128 + col, mask=(seq_idx_j + row < S)[:, None], other=0.0)
        K_j1 = tl.load(K_ptr + (row_start_j + row) * 128 + col + 64, mask=(seq_idx_j + row < S)[:, None], other=0.0)
        V_j0 = tl.load(V_ptr + (row_start_j + row) * 128 + col, mask=(seq_idx_j + row < S)[:, None], other=0.0)
        V_j1 = tl.load(V_ptr + (row_start_j + row) * 128 + col + 64, mask=(seq_idx_j + row < S)[:, None], other=0.0)
        
        acc_S = tl.zeros((32, 32), dtype=tl.float32)
        acc_dP = tl.zeros((32, 32), dtype=tl.float32)
        
        acc_S = tl.dot(Q_i0, K_j0.T, acc_S)
        acc_S = tl.dot(Q_i1, K_j1.T, acc_S)
        
        acc_dP = tl.dot(dO_i0, V_j0.T, acc_dP)
        acc_dP = tl.dot(dO_i1, V_j1.T, acc_dP)
        
        P_ij = tl.exp(acc_S * scale - L_i[:, None])
        
        row_q = tl.arange(0, 32)
        col_k = tl.arange(0, 32)[None, :]
        mask_j = (seq_idx_j + col_k < S)[None, :]
        P_ij = P_ij * mask_j
        
        dS_ij = P_ij * (acc_dP - D_i[:, None]) * scale
        
        acc_dQ0 = tl.dot(dS_ij, K_j0, acc_dQ0)
        acc_dQ1 = tl.dot(dS_ij, K_j1, acc_dQ1)
        
    ptrs_dQ0 = dQ_ptr + (row_start_i + row) * 128 + col
    ptrs_dQ1 = dQ_ptr + (row_start_i + row) * 128 + col + 64
    
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
    
    Q_ptr = Q.reshape(B * H, S, d)
    K_ptr = K.reshape(B * H, S, d)
    V_ptr = V.reshape(B * H, S, d)
    O_ptr = O.reshape(B * H, S, d)
    dO_ptr = dO.reshape(B * H, S, d)
    
    dQ_ptr = dQ.reshape(B * H, S, d)
    dK_ptr = dK.reshape(B * H, S, d)
    dV_ptr = dV.reshape(B * H, S, d)
    
    L_ptr = L
    
    grid_dk_dv = ((S + 31) // 32, B * H)
    _bwd_dk_dv_kernel[grid_dk_dv](
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dK_ptr, dV_ptr, S, scale,
        num_warps=4, num_stages=3
    )
    
    grid_dq = ((S + 31) // 32, B * H)
    _bwd_dq_kernel[grid_dq](
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
        dQ_ptr, S, scale,
        num_warps=4, num_stages=3
    )