import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, dO_ptr, dQ_ptr,
    S, H, B,
    sdpa_scale,
    BLOCK_M: tl.constexpr,
):
    col_idx = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    start_row_seq = col_idx * BLOCK_M
    batch_head_offset = (b * H + h) * S * 128
    
    rows = tl.arange(0, BLOCK_M)
    seq_q = start_row_seq + rows
    mask_q = (seq_q < S)[:, None]
    
    q_off0 = batch_head_offset + seq_q[:, None] * 128 + tl.arange(0, 64)[None, :]
    q_off1 = batch_head_offset + seq_q[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    
    Q0 = tl.load(Q_ptr + q_off0, mask=mask_q, other=0.0).to(tl.float32)
    Q1 = tl.load(Q_ptr + q_off1, mask=mask_q, other=0.0).to(tl.float32)
    
    dO0 = tl.load(dO_ptr + q_off0, mask=mask_q, other=0.0).to(tl.float32)
    dO1 = tl.load(dO_ptr + q_off1, mask=mask_q, other=0.0).to(tl.float32)
    
    O0 = tl.load(O_ptr + q_off0, mask=mask_q, other=0.0).to(tl.float32)
    O1 = tl.load(O_ptr + q_off1, mask=mask_q, other=0.0).to(tl.float32)
    
    D = tl.sum(O0 * dO0 + O1 * dO1, axis=1)
    
    lse_off_L = (b * H + h) * S + seq_q
    lse = tl.load(L_ptr + lse_off_L, mask=seq_q < S, other=0.0)
    
    dQ_acc0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    dQ_acc1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    num_k_blocks = tl.cdiv(S, 64)
    for j in range(num_k_blocks):
        seq_k = j * 64 + rows
        mask_k = (seq_k < S)[:, None]
        
        k_off0 = batch_head_offset + seq_k[:, None] * 128 + tl.arange(0, 64)[None, :]
        k_off1 = batch_head_offset + seq_k[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
        
        K0 = tl.load(K_ptr + k_off0, mask=mask_k, other=0.0).to(tl.float32)
        K1 = tl.load(K_ptr + k_off1, mask=mask_k, other=0.0).to(tl.float32)
        
        V0 = tl.load(V_ptr + k_off0, mask=mask_k, other=0.0).to(tl.float32)
        V1 = tl.load(V_ptr + k_off1, mask=mask_k, other=0.0).to(tl.float32)

        S_val = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)

        P = tl.exp(S_val * sdpa_scale - lse[:, None])
        
        mask_q_2d = (start_row_seq + rows < S)[:, None]
        mask_k_2d = (j * 64 + rows < S)[None, :]
        P = P * mask_q_2d * mask_k_2d
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)

        dS = P * (dP - D[:, None]) * sdpa_scale
        
        dQ_acc0 = tl.dot(dS, K0, acc=dQ_acc0)
        dQ_acc1 = tl.dot(dS, K1, acc=dQ_acc1)

    dq_off0 = batch_head_offset + seq_q[:, None] * 128 + tl.arange(0, 64)[None, :]
    dq_off1 = batch_head_offset + seq_q[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    
    store_mask = (seq_q < S)[:, None]
    tl.store(dQ_ptr + dq_off0, dQ_acc0.to(Q_ptr.dtype.element_ty), mask=store_mask)
    tl.store(dQ_ptr + dq_off1, dQ_acc1.to(Q_ptr.dtype.element_ty), mask=store_mask)


@triton.jit
def _bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, dO_ptr, dK_ptr, dV_ptr,
    S, H, B,
    sdpa_scale,
    BLOCK_N: tl.constexpr,
):
    col_idx = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    start_row_seq = col_idx * BLOCK_N
    batch_head_offset = (b * H + h) * S * 128
    
    rows = tl.arange(0, BLOCK_N)
    seq_k = start_row_seq + rows
    mask_k = (seq_k < S)[:, None]
    
    k_off0 = batch_head_offset + seq_k[:, None] * 128 + tl.arange(0, 64)[None, :]
    k_off1 = batch_head_offset + seq_k[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    
    K0 = tl.load(K_ptr + k_off0, mask=mask_k, other=0.0).to(tl.float32)
    K1 = tl.load(K_ptr + k_off1, mask=mask_k, other=0.0).to(tl.float32)
    
    V0 = tl.load(V_ptr + k_off0, mask=mask_k, other=0.0).to(tl.float32)
    V1 = tl.load(V_ptr + k_off1, mask=mask_k, other=0.0).to(tl.float32)

    dK_acc0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    dK_acc1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    dV_acc0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    dV_acc1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)

    num_q_blocks = tl.cdiv(S, 64)
    for i in range(num_q_blocks):
        seq_q = i * 64 + rows
        mask_q = (seq_q < S)[:, None]
        
        q_off0 = batch_head_offset + seq_q[:, None] * 128 + tl.arange(0, 64)[None, :]
        q_off1 = batch_head_offset + seq_q[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
        
        Q0 = tl.load(Q_ptr + q_off0, mask=mask_q, other=0.0).to(tl.float32)
        Q1 = tl.load(Q_ptr + q_off1, mask=mask_q, other=0.0).to(tl.float32)
        
        dO0 = tl.load(dO_ptr + q_off0, mask=mask_q, other=0.0).to(tl.float32)
        dO1 = tl.load(dO_ptr + q_off1, mask=mask_q, other=0.0).to(tl.float32)
        
        O0 = tl.load(O_ptr + q_off0, mask=mask_q, other=0.0).to(tl.float32)
        O1 = tl.load(O_ptr + q_off1, mask=mask_q, other=0.0).to(tl.float32)

        D = tl.sum(O0 * dO0 + O1 * dO1, axis=1)
        
        lse_off_L = (b * H + h) * S + seq_q
        lse = tl.load(L_ptr + lse_off_L, mask=seq_q < S, other=0.0)

        S_val = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)

        P = tl.exp(S_val * sdpa_scale - lse[:, None])
        
        mask_q_2d = (i * 64 + rows < S)[:, None]
        mask_k_2d = (start_row_seq + rows < S)[None, :]
        P = P * mask_q_2d * mask_k_2d
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)

        dS = P * (dP - D[:, None]) * sdpa_scale
        
        dV_acc0 = tl.dot(P.T, dO0, acc=dV_acc0)
        dV_acc1 = tl.dot(P.T, dO1, acc=dV_acc1)
        
        dK_acc0 = tl.dot(dS.T, Q0, acc=dK_acc0)
        dK_acc1 = tl.dot(dS.T, Q1, acc=dK_acc1)

    dk_off0 = batch_head_offset + seq_k[:, None] * 128 + tl.arange(0, 64)[None, :]
    dk_off1 = batch_head_offset + seq_k[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    
    store_mask_k = (seq_k < S)[:, None]
    tl.store(dK_ptr + dk_off0, dK_acc0.to(Q_ptr.dtype.element_ty), mask=store_mask_k)
    tl.store(dK_ptr + dk_off1, dK_acc1.to(Q_ptr.dtype.element_ty), mask=store_mask_k)
    
    tl.store(dV_ptr + dk_off0, dV_acc0.to(Q_ptr.dtype.element_ty), mask=store_mask_k)
    tl.store(dV_ptr + dk_off1, dV_acc1.to(Q_ptr.dtype.element_ty), mask=store_mask_k)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute the backward pass for multi-head attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sdpa_scale = 0.08838834764831843 
    
    grid_dq = (triton.cdiv(S, 64), H, B)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, L, dO, dQ, S, H, B, sdpa_scale,
        BLOCK_M=64, num_warps=8, num_stages=2
    )

    grid_dk_dv = (triton.cdiv(S, 64), H, B)
    _bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, L, dO, dK, dV, S, H, B, sdpa_scale,
        BLOCK_N=64, num_warps=8, num_stages=2
    )