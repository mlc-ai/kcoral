import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, dO_ptr, dQ_ptr,
    S, H, B,
    sdpa_scale,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    col_idx = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    str_S = 128
    str_d = 1
    
    bha = b * H * S * 128 + h * S * 128
    
    rows = tl.arange(0, BLOCK_N)
    seq_q = col_idx * BLOCK_N + rows
    mask_q = seq_q < S
    
    Q_tile = []
    dO_tile = []
    for c in range(NUM_CHUNKS):
        k_offs = c * 64 + tl.arange(0, 64)
        off = bha + seq_q[:, None] * str_S + k_offs[None, :] * str_d
        Q_tile.append(tl.load(Q_ptr + off, mask=mask_q[:, None], other=0.0))
        do_offs = bha + seq_q[:, None] * str_S + k_offs[None, :] * str_d
        dO_tile.append(tl.load(dO_ptr + do_offs, mask=mask_q[:, None], other=0.0))

    dQ_acc = [tl.zeros((BLOCK_N, 64), dtype=tl.float32) for _ in range(NUM_CHUNKS)]
    
    lse_off_L = (b * H + h) * S + seq_q
    lse = tl.load(L_ptr + lse_off_L, mask=seq_q < S, other=0.0)

    num_k_blocks = tl.cdiv(S, BLOCK_K)
    for j in range(num_k_blocks):
        rows_k = tl.arange(0, BLOCK_K)
        seq_k = j * BLOCK_K + rows_k
        mask_k = seq_k < S
        
        K_tile = []
        V_tile = []
        for c in range(NUM_CHUNKS):
            k_offs = c * 64 + tl.arange(0, 64)
            off = bha + seq_k[:, None] * str_S + k_offs[None, :] * str_d
            K_tile.append(tl.load(K_ptr + off, mask=mask_k[:, None], other=0.0))
            V_tile.append(tl.load(V_ptr + off, mask=mask_k[:, None], other=0.0))

        S_val = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for c_idx in range(NUM_CHUNKS):
            S_val = tl.dot(Q_tile[c_idx], K_tile[c_idx].T, acc=S_val)

        P = tl.exp(S_val * sdpa_scale - lse[:, None])
        P = P * mask_q[:, None] * mask_k[None, :]
        
        dP = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for c_idx in range(NUM_CHUNKS):
            dP = tl.dot(dO_tile[c_idx], V_tile[c_idx].T, acc=dP)

        dS = P * (dP - 0.0) * sdpa_scale
        
        for c_idx in range(NUM_CHUNKS):
            dQ_acc[c_idx] = tl.dot(dS, K_tile[c_idx], acc=dQ_acc[c_idx])

    for c in range(NUM_CHUNKS):
        k_offs = c * 64 + tl.arange(0, 64)
        out_off = bha + seq_q[:, None] * str_S + k_offs[None, :] * str_d
        tl.store(dQ_ptr + out_off, dQ_acc[c].to(Q_ptr.dtype.element_ty), mask=mask_q[:, None])


@triton.jit
def _bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, dO_ptr, dK_ptr, dV_ptr,
    S, H, B,
    sdpa_scale,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    col_idx = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    str_S = 128
    str_d = 1
    
    bha = b * H * S * 128 + h * S * 128
    
    rows = tl.arange(0, BLOCK_K)
    seq_k = col_idx * BLOCK_K + rows
    mask_k = seq_k < S
    
    K_tile = []
    V_tile = []
    for c in range(NUM_CHUNKS):
        k_offs = c * 64 + tl.arange(0, 64)
        off = bha + seq_k[:, None] * str_S + k_offs[None, :] * str_d
        K_tile.append(tl.load(K_ptr + off, mask=mask_k[:, None], other=0.0))
        V_tile.append(tl.load(V_ptr + off, mask=mask_k[:, None], other=0.0))

    dK_acc = [tl.zeros((BLOCK_K, 64), dtype=tl.float32) for _ in range(NUM_CHUNKS)]
    dV_acc = [tl.zeros((BLOCK_K, 64), dtype=tl.float32) for _ in range(NUM_CHUNKS)]

    num_q_blocks = tl.cdiv(S, BLOCK_N)
    for i in range(num_q_blocks):
        rows_q = tl.arange(0, BLOCK_N)
        seq_q = i * BLOCK_N + rows_q
        mask_q = seq_q < S
        
        Q_tile = []
        dO_tile = []
        for c in range(NUM_CHUNKS):
            k_offs = c * 64 + tl.arange(0, 64)
            off = bha + seq_q[:, None] * str_S + k_offs[None, :] * str_d
            Q_tile.append(tl.load(Q_ptr + off, mask=mask_q[:, None], other=0.0))
            do_offs = bha + seq_q[:, None] * str_S + k_offs[None, :] * str_d
            dO_tile.append(tl.load(dO_ptr + do_offs, mask=mask_q[:, None], other=0.0))

        S_val = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for c_idx in range(NUM_CHUNKS):
            S_val = tl.dot(Q_tile[c_idx], K_tile[c_idx].T, acc=S_val)

        lse_off_L = (b * H + h) * S + seq_q
        lse = tl.load(L_ptr + lse_off_L, mask=seq_q < S, other=0.0)
        
        P = tl.exp(S_val * sdpa_scale - lse[:, None])
        P = P * mask_q[:, None] * mask_k[None, :]
        
        dP = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for c_idx in range(NUM_CHUNKS):
            dP = tl.dot(dO_tile[c_idx], V_tile[c_idx].T, acc=dP)

        dS = P * (dP - 0.0) * sdpa_scale
        
        for c_idx in range(NUM_CHUNKS):
            dV_acc[c_idx] = tl.dot(P.T, dO_tile[c_idx], acc=dV_acc[c_idx])
            dK_acc[c_idx] = tl.dot(dS.T, Q_tile[c_idx], acc=dK_acc[c_idx])

    for c in range(NUM_CHUNKS):
        k_offs = c * 64 + tl.arange(0, 64)
        out_off = bha + seq_k[:, None] * str_S + k_offs[None, :] * str_d
        tl.store(dK_ptr + out_off, dK_acc[c].to(Q_ptr.dtype.element_ty), mask=mask_k[:, None])
        tl.store(dV_ptr + out_off, dV_acc[c].to(Q_ptr.dtype.element_ty), mask=mask_k[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute the backward pass for multi-head attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    sdpa_scale = 0.08838834764831843 
    
    grid_dq = (triton.cdiv(S, 64), H, B)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, L, dO, dQ, S, H, B, sdpa_scale,
        NUM_CHUNKS=2, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2
    )

    grid_dk_dv = (triton.cdiv(S, 64), H, B)
    _bwd_dk_dv_kernel[grid_dk_dv](
        Q, K, V, L, dO, dK, dV, S, H, B, sdpa_scale,
        NUM_CHUNKS=2, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2
    )