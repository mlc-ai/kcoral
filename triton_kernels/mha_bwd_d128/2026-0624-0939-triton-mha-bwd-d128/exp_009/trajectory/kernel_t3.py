import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, O_ptr, dO_ptr, dQ_ptr,
    S, H, B,
    sdpa_scale,
    BLOCK_M: tl.constexpr,
):
    col_idx = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    bha = (b * H + h) * S * 128
    
    rows = tl.arange(0, BLOCK_M)
    seq_q = col_idx * BLOCK_M + rows
    mask_q = seq_q < S
    
    Q0 = tl.load(Q_ptr + bha + seq_q[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    Q1 = tl.load(Q_ptr + bha + seq_q[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    
    dO0 = tl.load(dO_ptr + bha + seq_q[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    dO1 = tl.load(dO_ptr + bha + seq_q[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    
    O0 = tl.load(O_ptr + bha + seq_q[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    O1 = tl.load(O_ptr + bha + seq_q[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    
    D = tl.sum(O0 * dO0 + O1 * dO1, axis=1)
    
    lse_off_L = (b * H + h) * S + seq_q
    lse = tl.load(L_ptr + lse_off_L, mask=seq_q < S, other=0.0)
    
    dQ0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    dQ1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    num_k_blocks = tl.cdiv(S, 64)
    for j in range(num_k_blocks):
        seq_k = j * 64 + rows
        mask_k = seq_k < S
        
        K0 = tl.load(K_ptr + bha + seq_k[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
        K1 = tl.load(K_ptr + bha + seq_k[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
        
        V0 = tl.load(V_ptr + bha + seq_k[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
        V1 = tl.load(V_ptr + bha + seq_k[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)

        S_val = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)

        P = tl.exp(S_val * sdpa_scale - lse[:, None])
        P = P * mask_q[:, None] * mask_k[None, :]
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)

        dS = P * (dP - D[:, None]) * sdpa_scale
        
        dQ0 = tl.dot(dS, K0, acc=dQ0)
        dQ1 = tl.dot(dS, K1, acc=dQ1)

    dq_off0 = bha + seq_q[:, None] * 128 + tl.arange(0, 64)[None, :]
    dq_off1 = bha + seq_q[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    
    tl.store(dQ_ptr + dq_off0, dQ0.to(Q_ptr.dtype.element_ty), mask=mask_q[:, None])
    tl.store(dQ_ptr + dq_off1, dQ1.to(Q_ptr.dtype.element_ty), mask=mask_q[:, None])


@triton.jit
def _bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, O_ptr, dO_ptr, dK_ptr, dV_ptr,
    S, H, B,
    sdpa_scale,
    BLOCK_N: tl.constexpr,
):
    col_idx = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    bha = (b * H + h) * S * 128
    
    rows = tl.arange(0, BLOCK_N)
    seq_k = col_idx * BLOCK_N + rows
    mask_k = seq_k < S
    
    K0 = tl.load(K_ptr + bha + seq_k[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
    K1 = tl.load(K_ptr + bha + seq_k[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
    
    V0 = tl.load(V_ptr + bha + seq_k[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
    V1 = tl.load(V_ptr + bha + seq_k[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)

    dK0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    dK1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    dV0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    dV1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)

    num_q_blocks = tl.cdiv(S, 64)
    for i in range(num_q_blocks):
        seq_q = i * 64 + rows
        mask_q = seq_q < S
        
        Q0 = tl.load(Q_ptr + bha + seq_q[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        Q1 = tl.load(Q_ptr + bha + seq_q[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        
        dO0 = tl.load(dO_ptr + bha + seq_q[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        dO1 = tl.load(dO_ptr + bha + seq_q[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        
        O0 = tl.load(O_ptr + bha + seq_q[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        O1 = tl.load(O_ptr + bha + seq_q[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)

        D = tl.sum(O0 * dO0 + O1 * dO1, axis=1)
        
        lse_off_L = (b * H + h) * S + seq_q
        lse = tl.load(L_ptr + lse_off_L, mask=seq_q < S, other=0.0)

        S_val = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)

        P = tl.exp(S_val * sdpa_scale - lse[:, None])
        P = P * mask_q[:, None] * mask_k[None, :]
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)

        dS = P * (dP - D[:, None]) * sdpa_scale
        
        dV0 = tl.dot(P.T, dO0, acc=dV0)
        dV1 = tl.dot(P.T, dO1, acc=dV1)
        
        dK0 = tl.dot(dS.T, Q0, acc=dK0)
        dK1 = tl.dot(dS.T, Q1, acc=dK1)

    dk_off0 = bha + seq_k[:, None] * 128 + tl.arange(0, 64)[None, :]
    dk_off1 = bha + seq_k[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    
    tl.store(dK_ptr + dk_off0, dK0.to(Q_ptr.dtype.element_ty), mask=mask_k[:, None])
    tl.store(dK_ptr + dk_off1, dK1.to(Q_ptr.dtype.element_ty), mask=mask_k[:, None])
    
    tl.store(dV_ptr + dk_off0, dV0.to(Q_ptr.dtype.element_ty), mask=mask_k[:, None])
    tl.store(dV_ptr + dk_off1, dV1.to(Q_ptr.dtype.element_ty), mask=mask_k[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute the backward pass for multi-head attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sdpa_scale = 0.08838834764831843 
    
    grid_dq = (triton.cdiv(S, 64), H, B)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, L, O, dO, dQ, S, H, B, sdpa_scale,
        BLOCK_M=64, num_warps=4, num_stages=2
    )

    grid_dk_dv = (triton.cdiv(S, 64), H, B)
    _bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, L, O, dO, dK, dV, S, H, B, sdpa_scale,
        BLOCK_N=64, num_warps=4, num_stages=2
    )