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
    
    q_off0 = bha + seq_q[:, None] * 128 + tl.arange(0, 64)[None, :]
    q_off1 = bha + seq_q[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    
    Q0 = tl.load(Q_ptr + q_off0, zero_padding=True).to(tl.float32)
    Q1 = tl.load(Q_ptr + q_off1, zero_padding=True).to(tl.float32)
    
    dO0 = tl.load(dO_ptr + q_off0, zero_padding=True).to(tl.float32)
    dO1 = tl.load(dO_ptr + q_off1, zero_padding=True).to(tl.float32)
    
    O0 = tl.load(O_ptr + q_off0, zero_padding=True).to(tl.float32)
    O1 = tl.load(O_ptr + q_off1, zero_padding=True).to(tl.float32)
    
    D = tl.sum(O0 * dO0 + O1 * dO1, axis=1)
    
    lse_off_L = (b * H + h) * S + seq_q
    lse = tl.load(L_ptr + lse_off_L, zero_padding=True)
    
    dQ0_acc = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    dQ1_acc = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    mask_q = (seq_q < S)[:, None]
    
    num_k_blocks = tl.cdiv(S, BLOCK_M)
    for j in range(num_k_blocks):
        seq_k = j * BLOCK_M + rows
        
        k_off0 = bha + seq_k[:, None] * 128 + tl.arange(0, 64)[None, :]
        k_off1 = bha + seq_k[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
        
        K0 = tl.load(K_ptr + k_off0, zero_padding=True).to(tl.float32)
        K1 = tl.load(K_ptr + k_off1, zero_padding=True).to(tl.float32)
        
        V0 = tl.load(V_ptr + k_off0, zero_padding=True).to(tl.float32)
        V1 = tl.load(V_ptr + k_off1, zero_padding=True).to(tl.float32)

        S_val = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)

        P = tl.exp(S_val * sdpa_scale - lse[:, None])
        
        mask_k = (seq_k < S)[None, :]
        P = P * mask_q * mask_k
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)

        dS = P * (dP - D[:, None]) * sdpa_scale
        
        dQ0_acc = tl.dot(dS, K0, acc=dQ0_acc)
        dQ1_acc = tl.dot(dS, K1, acc=dQ1_acc)

    tl.store(dQ_ptr + q_off0, dQ0_acc.to(Q_ptr.dtype.element_ty), mask=mask_q)
    tl.store(dQ_ptr + q_off1, dQ1_acc.to(Q_ptr.dtype.element_ty), mask=mask_q)


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
    
    k_off0 = bha + seq_k[:, None] * 128 + tl.arange(0, 64)[None, :]
    k_off1 = bha + seq_k[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    
    K0 = tl.load(K_ptr + k_off0, zero_padding=True).to(tl.float32)
    K1 = tl.load(K_ptr + k_off1, zero_padding=True).to(tl.float32)
    
    V0 = tl.load(V_ptr + k_off0, zero_padding=True).to(tl.float32)
    V1 = tl.load(V_ptr + k_off1, zero_padding=True).to(tl.float32)

    dK0_acc = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    dK1_acc = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    dV0_acc = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    dV1_acc = tl.zeros((BLOCK_N, 64), dtype=tl.float32)

    mask_k = (seq_k < S)[None, :]

    num_q_blocks = tl.cdiv(S, BLOCK_N)
    for i in range(num_q_blocks):
        seq_q = i * BLOCK_N + rows
        
        q_off0 = bha + seq_q[:, None] * 128 + tl.arange(0, 64)[None, :]
        q_off1 = bha + seq_q[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
        
        Q0 = tl.load(Q_ptr + q_off0, zero_padding=True).to(tl.float32)
        Q1 = tl.load(Q_ptr + q_off1, zero_padding=True).to(tl.float32)
        
        dO0 = tl.load(dO_ptr + q_off0, zero_padding=True).to(tl.float32)
        dO1 = tl.load(dO_ptr + q_off1, zero_padding=True).to(tl.float32)
        
        O0 = tl.load(O_ptr + q_off0, zero_padding=True).to(tl.float32)
        O1 = tl.load(O_ptr + q_off1, zero_padding=True).to(tl.float32)

        D = tl.sum(O0 * dO0 + O1 * dO1, axis=1)
        
        lse_off_L = (b * H + h) * S + seq_q
        lse = tl.load(L_ptr + lse_off_L, zero_padding=True)

        S_val = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)

        P = tl.exp(S_val * sdpa_scale - lse[:, None])
        
        mask_q = (seq_q < S)[:, None]
        P = P * mask_q * mask_k
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)

        dS = P * (dP - D[:, None]) * sdpa_scale
        
        dV0_acc = tl.dot(P.T, dO0, acc=dV0_acc)
        dV1_acc = tl.dot(P.T, dO1, acc=dV1_acc)
        
        dK0_acc = tl.dot(dS.T, Q0, acc=dK0_acc)
        dK1_acc = tl.dot(dS.T, Q1, acc=dK1_acc)

    tl.store(dK_ptr + k_off0, dK0_acc.to(Q_ptr.dtype.element_ty), mask=mask_k)
    tl.store(dK_ptr + k_off1, dK1_acc.to(Q_ptr.dtype.element_ty), mask=mask_k)
    
    tl.store(dV_ptr + k_off0, dV0_acc.to(Q_ptr.dtype.element_ty), mask=mask_k)
    tl.store(dV_ptr + k_off1, dV1_acc.to(Q_ptr.dtype.element_ty), mask=mask_k)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute the backward pass for multi-head attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sdpa_scale = 0.08838834764831843 
    
    grid_dq = (triton.cdiv(S, 64), H, B)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, L, O, dO, dQ, S, H, B, sdpa_scale,
        BLOCK_M=64, num_warps=8, num_stages=2
    )

    grid_dk_dv = (triton.cdiv(S, 64), H, B)
    _bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, L, O, dO, dK, dV, S, H, B, sdpa_scale,
        BLOCK_N=64, num_warps=8, num_stages=2
    )