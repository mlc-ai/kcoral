import math
import torch
import triton
import triton.language as tl


@triton.jit
def _precompute_D_kernel(O_ptr, dO_ptr, D_ptr, S, H, B, BLOCK: tl.constexpr):
    b = tl.program_id(2)
    h = tl.program_id(1)
    row_idx_2d = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch_head_offset = (b * H + h) * S * 128
    
    cols_128 = tl.arange(0, 128)
    o_off = batch_head_offset + row_idx_2d[:, None] * 128 + cols_128[None, :]
    
    mask_row = row_idx_2d < S
    O_tile = tl.load(O_ptr + o_off, mask=mask_row[:, None], other=0.0)
    dO_tile = tl.load(dO_ptr + o_off, mask=mask_row[:, None], other=0.0)
    
    o_row_fp32 = O_tile.to(tl.float32)
    do_row_fp32 = dO_tile.to(tl.float32)
    d_val = tl.sum(o_row_fp32 * do_row_fp32, axis=1)
    
    d_off = (b * H + h) * S + row_idx_2d
    tl.store(D_ptr + d_off, d_val, mask=mask_row)


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, D_ptr, dO_ptr, dQ_ptr,
    S, H, B,
    sdpa_scale,
    BLOCK_M: tl.constexpr,
):
    col_idx = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    start_row_seq = col_idx * BLOCK_M
    batch_head_offset = (b * H + h) * S * 128
    
    rows_2d = tl.arange(0, 128)
    cols_d0 = tl.arange(0, 64)
    cols_d1 = tl.arange(0, 64)
    
    q_off0 = batch_head_offset + rows_2d[:, None] * 128 + cols_d0[None, :]
    q_off1 = batch_head_offset + rows_2d[:, None] * 128 + cols_d1[None, :]
    
    mask_q = (start_row_seq + rows_2d < S)[:, None] & (cols_d0 < 128)[None, :]
    Q0 = tl.load(Q_ptr + q_off0, mask=mask_q, other=0.0)
    Q1 = tl.load(Q_ptr + q_off1, mask=mask_q, other=0.0)
    
    mask_q_do = (start_row_seq + rows_2d < S)[:, None]
    dO0 = tl.load(dO_ptr + q_off0, mask=mask_q_do, other=0.0)
    dO1 = tl.load(dO_ptr + q_off1, mask=mask_q_do, other=0.0)
    O0 = tl.load(O_ptr + q_off0, mask=mask_q_do, other=0.0)
    O1 = tl.load(O_ptr + q_off1, mask=mask_q_do, other=0.0)
    
    D = tl.sum(O0 * dO0 + O1 * dO1, axis=1)
    
    seq_q = start_row_seq + rows_2d
    lse_off_L = (b * H + h) * S + seq_q
    lse = tl.load(L_ptr + lse_off_L, mask=seq_q < S, other=0.0)
    
    dQ_acc0 = tl.zeros((128, 64), dtype=tl.float32)
    dQ_acc1 = tl.zeros((128, 64), dtype=tl.float32)
    
    for k_start_seq in range(0, S, 128):
        mask_k = (k_start_seq + rows_2d < S)[:, None]
        
        k_off0 = batch_head_offset + rows_2d[:, None] * 128 + cols_d0[None, :]
        k_off1 = batch_head_offset + rows_2d[:, None] * 128 + cols_d1[None, :]
        
        K0 = tl.load(K_ptr + k_off0, mask=mask_k, other=0.0)
        K1 = tl.load(K_ptr + k_off1, mask=mask_k, other=0.0)
        
        V0 = tl.load(V_ptr + k_off0, mask=mask_k, other=0.0)
        V1 = tl.load(V_ptr + k_off1, mask=mask_k, other=0.0)

        S_val = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)

        P = tl.exp(S_val * sdpa_scale - lse[:, None])
        mask_q_2d = (start_row_seq + rows_2d < S)[:, None]
        mask_k_2d = (k_start_seq + rows_2d < S)[None, :]
        P = P * mask_q_2d * mask_k_2d
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)

        dS = P * (dP - D[:, None]) * sdpa_scale
        
        dQ_acc0 = tl.dot(dS, K0, acc=dQ_acc0)
        dQ_acc1 = tl.dot(dS, K1, acc=dQ_acc1)

    dq_off0 = batch_head_offset + rows_2d[:, None] * 128 + cols_d0[None, :]
    dq_off1 = batch_head_offset + rows_2d[:, None] * 128 + cols_d1[None, :]
    
    dq_out0 = dQ_acc0.to(Q_ptr.dtype.element_ty)
    dq_out1 = dQ_acc1.to(Q_ptr.dtype.element_ty)
    
    store_mask = (start_row_seq + rows_2d < S)[:, None]
    tl.store(dQ_ptr + dq_off0, dq_out0, mask=store_mask)
    tl.store(dQ_ptr + dq_off1, dq_out1, mask=store_mask)


@triton.jit
def _bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, D_ptr, dO_ptr, dK_ptr, dV_ptr,
    S, H, B,
    sdpa_scale,
    BLOCK_N: tl.constexpr,
):
    col_idx = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    start_row_seq = col_idx * BLOCK_N
    batch_head_offset = (b * H + h) * S * 128
    
    rows_2d = tl.arange(0, 128)
    cols_d0 = tl.arange(0, 64)
    cols_d1 = tl.arange(0, 64)
    
    mask_k = (start_row_seq + rows_2d < S)[:, None] & (cols_d0 < 128)[None, :]
    
    k_off0 = batch_head_offset + rows_2d[:, None] * 128 + cols_d0[None, :]
    k_off1 = batch_head_offset + rows_2d[:, None] * 128 + cols_d1[None, :]
    
    K0 = tl.load(K_ptr + k_off0, mask=mask_k, other=0.0)
    K1 = tl.load(K_ptr + k_off1, mask=mask_k, other=0.0)
    
    V0 = tl.load(V_ptr + k_off0, mask=mask_k, other=0.0)
    V1 = tl.load(V_ptr + k_off1, mask=mask_k, other=0.0)

    dK_acc0 = tl.zeros((128, 64), dtype=tl.float32)
    dK_acc1 = tl.zeros((128, 64), dtype=tl.float32)
    dV_acc0 = tl.zeros((128, 64), dtype=tl.float32)
    dV_acc1 = tl.zeros((128, 64), dtype=tl.float32)

    for q_start_seq in range(0, S, 128):
        seq_q = q_start_seq + rows_2d
        q_off0 = batch_head_offset + rows_2d[:, None] * 128 + cols_d0[None, :]
        q_off1 = batch_head_offset + rows_2d[:, None] * 128 + cols_d1[None, :]
        
        mask_q_do = (seq_q < S)[:, None]
        Q0 = tl.load(Q_ptr + q_off0, mask=mask_q_do, other=0.0)
        Q1 = tl.load(Q_ptr + q_off1, mask=mask_q_do, other=0.0)
        
        dO0 = tl.load(dO_ptr + q_off0, mask=mask_q_do, other=0.0)
        dO1 = tl.load(dO_ptr + q_off1, mask=mask_q_do, other=0.0)
        
        O0 = tl.load(O_ptr + q_off0, mask=mask_q_do, other=0.0)
        O1 = tl.load(O_ptr + q_off1, mask=mask_q_do, other=0.0)

        D = tl.sum(O0 * dO0 + O1 * dO1, axis=1)
        
        lse_off_L = (b * H + h) * S + seq_q
        lse = tl.load(L_ptr + lse_off_L, mask=seq_q < S, other=0.0)

        S_val = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)

        P = tl.exp(S_val * sdpa_scale - lse[:, None])
        
        mask_q_2d = (q_start_seq + rows_2d < S)[:, None]
        mask_k_2d = (start_row_seq + rows_2d < S)[None, :]
        P = P * mask_q_2d * mask_k_2d
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)

        dS = P * (dP - D[:, None]) * sdpa_scale
        
        dV_acc0 = tl.dot(P.T, dO0, acc=dV_acc0)
        dV_acc1 = tl.dot(P.T, dO1, acc=dV_acc1)
        
        dK_acc0 = tl.dot(dS.T, Q0, acc=dK_acc0)
        dK_acc1 = tl.dot(dS.T, Q1, acc=dK_acc1)

    dk_off0 = batch_head_offset + rows_2d[:, None] * 128 + cols_d0[None, :]
    dk_off1 = batch_head_offset + rows_2d[:, None] * 128 + cols_d1[None, :]
    
    tl.store(dK_ptr + dk_off0, dK_acc0.to(Q_ptr.dtype.element_ty), mask=mask_k)
    tl.store(dK_ptr + dk_off1, dK_acc1.to(Q_ptr.dtype.element_ty), mask=mask_k)
    
    tl.store(dV_ptr + dk_off0, dV_acc0.to(Q_ptr.dtype.element_ty), mask=mask_k)
    tl.store(dV_ptr + dk_off1, dV_acc1.to(Q_ptr.dtype.element_ty), mask=mask_k)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute the backward pass for multi-head attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sdpa_scale = 0.08838834764831843 
    
    device = Q.device
    D_ptr = torch.empty((B, H, S), dtype=torch.float32, device=device)
    grid_D = ((S + 255) // 256, H, B)
    _precompute_D_kernel[grid_D](O, dO, D_ptr, S, H, B, BLOCK=256)

    grid_dq = (triton.cdiv(S, 128), H, B)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, L, D_ptr, dO, dQ, S, H, B, sdpa_scale,
        BLOCK_M=128, num_warps=8, num_stages=2
    )

    grid_dk_dv = (triton.cdiv(S, 128), H, B)
    _bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, L, D_ptr, dO, dK, dV, S, H, B, sdpa_scale,
        BLOCK_N=128, num_warps=8, num_stages=2
    )