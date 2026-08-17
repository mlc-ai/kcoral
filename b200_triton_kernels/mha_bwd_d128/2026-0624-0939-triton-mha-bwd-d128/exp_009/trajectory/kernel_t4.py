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
    
    q_off = bha + seq_q[:, None] * 128 + tl.arange(0, 128)[None, :]
    
    Q_tile = tl.load(Q_ptr + q_off, mask=mask_q[:, None], other=0.0)
    dO_tile = tl.load(dO_ptr + q_off, mask=mask_q[:, None], other=0.0)
    O_tile = tl.load(O_ptr + q_off, mask=mask_q[:, None], other=0.0)
    
    D = tl.sum(O_tile * dO_tile, axis=1)
    
    lse_off_L = (b * H + h) * S + seq_q
    lse = tl.load(L_ptr + lse_off_L, mask=seq_q < S, other=0.0)
    
    dQ_acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    
    num_k_blocks = tl.cdiv(S, 64)
    for j in range(num_k_blocks):
        seq_k = j * 64 + rows
        mask_k = seq_k < S
        
        k_off = bha + seq_k[:, None] * 128 + tl.arange(0, 128)[None, :]
        
        K_tile = tl.load(K_ptr + k_off, mask=mask_k[:, None], other=0.0)
        V_tile = tl.load(V_ptr + k_off, mask=mask_k[:, None], other=0.0)

        S_val = tl.dot(Q_tile, K_tile.T)

        P = tl.exp(S_val * sdpa_scale - lse[:, None])
        P = P * mask_q[:, None] * mask_k[None, :]
        
        dP = tl.dot(dO_tile, V_tile.T)

        dS = P * (dP - D[:, None]) * sdpa_scale
        
        dQ_acc = tl.dot(dS, K_tile, acc=dQ_acc)

    dq_off = bha + seq_q[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(dQ_ptr + dq_off, dQ_acc.to(Q_ptr.dtype.element_ty), mask=mask_q[:, None])


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
    
    k_off = bha + seq_k[:, None] * 128 + tl.arange(0, 128)[None, :]
    
    K_tile = tl.load(K_ptr + k_off, mask=mask_k[:, None], other=0.0)
    V_tile = tl.load(V_ptr + k_off, mask=mask_k[:, None], other=0.0)

    dK_acc = tl.zeros((BLOCK_N, 128), dtype=tl.float32)
    dV_acc = tl.zeros((BLOCK_N, 128), dtype=tl.float32)

    num_q_blocks = tl.cdiv(S, 64)
    for i in range(num_q_blocks):
        seq_q = i * 64 + rows
        mask_q = seq_q < S
        
        q_off = bha + seq_q[:, None] * 128 + tl.arange(0, 128)[None, :]
        
        Q_tile = tl.load(Q_ptr + q_off, mask=mask_q[:, None], other=0.0)
        dO_tile = tl.load(dO_ptr + q_off, mask=mask_q[:, None], other=0.0)
        O_tile = tl.load(O_ptr + q_off, mask=mask_q[:, None], other=0.0)

        D = tl.sum(O_tile * dO_tile, axis=1)
        
        lse_off_L = (b * H + h) * S + seq_q
        lse = tl.load(L_ptr + lse_off_L, mask=seq_q < S, other=0.0)

        S_val = tl.dot(Q_tile, K_tile.T)

        P = tl.exp(S_val * sdpa_scale - lse[:, None])
        P = P * mask_q[:, None] * mask_k[None, :]
        
        dP = tl.dot(dO_tile, V_tile.T)

        dS = P * (dP - D[:, None]) * sdpa_scale
        
        dV_acc = tl.dot(P.T, dO_tile, acc=dV_acc)
        dK_acc = tl.dot(dS.T, Q_tile, acc=dK_acc)

    dk_off = bha + seq_k[:, None] * 128 + tl.arange(0, 128)[None, :]
    tl.store(dK_ptr + dk_off, dK_acc.to(Q_ptr.dtype.element_ty), mask=mask_k[:, None])
    tl.store(dV_ptr + dk_off, dV_acc.to(Q_ptr.dtype.element_ty), mask=mask_k[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute the backward pass for multi-head attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sdpa_scale = 0.08838834764831843 
    
    grid_dq = (triton.cdiv(S, 128), H, B)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, L, O, dO, dQ, S, H, B, sdpa_scale,
        BLOCK_M=128, num_warps=4, num_stages=2
    )

    grid_dk_dv = (triton.cdiv(S, 128), H, B)
    _bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, L, O, dO, dK, dV, S, H, B, sdpa_scale,
        BLOCK_N=128, num_warps=4, num_stages=2
    )