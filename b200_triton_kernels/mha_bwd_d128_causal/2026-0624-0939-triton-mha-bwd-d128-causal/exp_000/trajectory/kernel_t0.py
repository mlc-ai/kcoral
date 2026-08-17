import torch
import triton
import triton.language as tl
import math


@triton.jit
def compute_dQ(
    Q, K, V, O, dO, L, dQ,
    S_len,
    scale,
    attn_scale,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(1) * tl.program_id(2)
    j_start = 0
    i = tl.program_id(0)
    
    if i >= tl.cdiv(S_len, BLOCK):
        return
    
    b_h_offset = off_b * S_len * 128
    b_h_offset_s = off_b * S_len
    
    stride_s = 128
    
    idx_query = i * BLOCK + tl.arange(0, BLOCK)
    mask_q = idx_query < S_len
    
    Q_chunk = [
        tl.load(Q + b_h_offset + idx_query[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
        tl.load(Q + b_h_offset + idx_query[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
    ]
    dO_chunk = [
        tl.load(dO + b_h_offset + idx_query[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
        tl.load(dO + b_h_offset + idx_query[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
    ]
    
    L_vals = tl.load(L + b_h_offset_s + idx_query, mask=mask_q, other=0.0)
    
    dQ_acc = [tl.zeros((64, 64), tl.float32), tl.zeros((64, 64), tl.float32)]
    
    num_blocks = tl.cdiv(S_len, BLOCK)
    
    for j in range(j_start, i + 1):
        idx_key = j * BLOCK + tl.arange(0, BLOCK)
        mask_k = idx_key < S_len
        
        K_chunk = [
            tl.load(K + b_h_offset + idx_key[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
            tl.load(K + b_h_offset + idx_key[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
        ]
        V_chunk = [
            tl.load(V + b_h_offset + idx_key[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
            tl.load(V + b_h_offset + idx_key[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
        ]
        
        mask = (idx_query[:, None] >= idx_key[None, :]) & (mask_q[:, None]) & (mask_k[None, :])
        
        S_t = Q_chunk[0] @ K_chunk[0].T + Q_chunk[1] @ K_chunk[1].T
        
        S_t = tl.exp(S_t * scale - L_vals[:, None])
        S_t = S_t * mask
        
        dQ_acc[0] += S_t @ V_chunk[0]
        dQ_acc[1] += S_t @ V_chunk[1]
        
        O_chunk = [
            tl.load(O + b_h_offset + idx_query[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
            tl.load(O + b_h_offset + idx_query[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
        ]
        D_i = tl.sum(O_chunk[0] * dO_chunk[0], axis=1) + tl.sum(O_chunk[1] * dO_chunk[1], axis=1)
        
        S_t = S_t * (dO_chunk[0] @ V_chunk[0].T + dO_chunk[1] @ V_chunk[1].T - D_i[:, None]) * scale
        
        tl.store_shared("S_t", S_t)
        S_t = tl.load_shared("S_t", (64, 64), tl.float32)
        
        dQ_acc[0] -= S_t @ K_chunk[0]
        dQ_acc[1] -= S_t @ K_chunk[1]
    
    out = dQ + b_h_offset + idx_query[:, None] * stride_s + tl.arange(0, 64)[None, :]
    tl.store(out, dQ_acc[0], mask=mask_q[:, None])
    
    out = dQ + b_h_offset + idx_query[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :]
    tl.store(out, dQ_acc[1], mask=mask_q[:, None])


@triton.jit
def compute_dK(
    Q, K, V, O, dO, L,
    dK, dV, 
    S_len,
    scale,
    attn_scale,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(1) * tl.program_id(2)
    j = tl.program_id(0)
    
    if j >= tl.cdiv(S_len, BLOCK):
        return
    
    b_h_offset = off_b * S_len * 128
    b_h_offset_s = off_b * S_len
    
    stride_s = 128
    
    idx_key = j * BLOCK + tl.arange(0, BLOCK)
    mask_k = idx_key < S_len
    
    K_chunk = [
        tl.load(K + b_h_offset + idx_key[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
        tl.load(K + b_h_offset + idx_key[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
    ]
    V_chunk = [
        tl.load(V + b_h_offset + idx_key[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
        tl.load(V + b_h_offset + idx_key[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
    ]
    
    dK_acc = [tl.zeros((64, 64), tl.float32), tl.zeros((64, 64), tl.float32)]
    dV_acc = [tl.zeros((64, 64), tl.float32), tl.zeros((64, 64), tl.float32)]
    
    num_blocks = tl.cdiv(S_len, BLOCK)
    
    for i in range(j, num_blocks):
        idx_query = i * BLOCK + tl.arange(0, BLOCK)
        mask_q = idx_query < S_len
        
        Q_chunk = [
            tl.load(Q + b_h_offset + idx_query[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
            tl.load(Q + b_h_offset + idx_query[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
        ]
        dO_chunk = [
            tl.load(dO + b_h_offset + idx_query[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
            tl.load(dO + b_h_offset + idx_query[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
        ]
        
        L_vals = tl.load(L + b_h_offset_s + idx_query, mask=mask_q, other=0.0)
        
        mask = (idx_query[:, None] >= idx_key[None, :]) & (mask_q[:, None]) & (mask_k[None, :])
        
        S_t = Q_chunk[0] @ K_chunk[0].T + Q_chunk[1] @ K_chunk[1].T
        
        S_t = tl.exp(S_t * attn_scale - L_vals[:, None])
        S_t = S_t * mask
        
        dK_acc[0] += S_t @ Q_chunk[0]
        dK_acc[1] += S_t @ Q_chunk[1]
        
        dV_acc[0] += S_t @ dO_chunk[0]
        dV_acc[1] += S_t @ dO_chunk[1]
        
        O_chunk = [
            tl.load(O + b_h_offset + idx_key[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
            tl.load(O + b_h_offset + idx_key[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
        ]
        D_i = tl.sum(O_chunk[0] * dO_chunk[0], axis=1) + tl.sum(O_chunk[1] * dO_chunk[1], axis=1)
        
        S_t = S_t * (dO_chunk[0] @ V_chunk[0].T + dO_chunk[1] @ V_chunk[1].T - D_i[:, None]) * attn_scale
        
        tl.store_shared("S_t", S_t)
        S_t = tl.load_shared("S_t", (64, 64), tl.float32)
        
        dK_acc[0] -= S_t @ Q_chunk[0]
        dK_acc[1] -= S_t @ Q_chunk[1]
    
    out = dK + b_h_offset + idx_key[:, None] * stride_s + tl.arange(0, 64)[None, :]
    tl.store(out, dK_acc[0], mask=mask_k[:, None])
    
    out = dK + b_h_offset + idx_key[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :]
    tl.store(out, dK_acc[1], mask=mask_k[:, None])

    out = dV + b_h_offset + idx_key[:, None] * stride_s + tl.arange(0, 64)[None, :]
    tl.store(out, dV_acc[0], mask=mask_k[:, None])
    
    out = dV + b_h_offset + idx_key[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :]
    tl.store(out, dV_acc[1], mask=mask_k[:, None])


@triton.jit
def compute_dV(
    Q, K, V, O, dO, L,
    dV, 
    S_len,
    scale,
    attn_scale,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(1) * tl.program_id(2)
    j = tl.program_id(0)
    
    if j >= tl.cdiv(S_len, BLOCK):
        return
    
    b_h_offset = off_b * S_len * 128
    b_h_offset_s = off_b * S_len
    
    stride_s = 128
    
    idx_key = j * BLOCK + tl.arange(0, BLOCK)
    mask_k = idx_key < S_len
    
    V_chunk = [
        tl.load(V + b_h_offset + idx_key[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
        tl.load(V + b_h_offset + idx_key[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
    ]
    
    dV_acc = [tl.zeros((64, 64), tl.float32), tl.zeros((64, 64), tl.float32)]
    
    num_blocks = tl.cdiv(S_len, BLOCK)
    
    for i in range(j, num_blocks):
        idx_query = i * BLOCK + tl.arange(0, BLOCK)
        mask_q = idx_query < S_len
        
        Q_chunk = [
            tl.load(Q + b_h_offset + idx_query[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
            tl.load(Q + b_h_offset + idx_query[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
        ]
        dO_chunk = [
            tl.load(dO + b_h_offset + idx_query[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
            tl.load(dO + b_h_offset + idx_query[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0),
        ]
        
        K_chunk = [
            tl.load(K + b_h_offset + idx_key[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
            tl.load(K + b_h_offset + idx_key[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0),
        ]
        
        L_vals = tl.load(L + b_h_offset_s + idx_query, mask=mask_q, other=0.0)
        
        mask = (idx_query[:, None] >= idx_key[None, :]) & (mask_q[:, None]) & (mask_k[None, :])
        
        S_t = Q_chunk[0] @ K_chunk[0].T + Q_chunk[1] @ K_chunk[1].T
        
        S_t = tl.exp(S_t * scale - L_vals[:, None])
        S_t = S_t * mask
        
        tl.store_shared("S_t", S_t)
        S_t = tl.load_shared("S_t", (64, 64), tl.float32)
        
        dV_acc[0] += S_t.T @ dO_chunk[0]
        dV_acc[1] += S_t.T @ dO_chunk[1]
    
    out = dV + b_h_offset + idx_key[:, None] * stride_s + tl.arange(0, 64)[None, :]
    tl.store(out, dV_acc[0], mask=mask_k[:, None])
    
    out = dV + b_h_offset + idx_key[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :]
    tl.store(out, dV_acc[1], mask=mask_k[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, d_head = Q.shape
    
    if S_len == 0:
        return
        
    num_blocks = triton.cdiv(S_len, 64)
    grid = (num_blocks, B, H)
    
    scale = 1.0 / math.sqrt(d_head)
    attn_scale = 1.0 / math.sqrt(d_head)
    
    compute_dQ[grid](Q, K, V, O, dO, L, dQ, S_len, scale, attn_scale, BLOCK=64, num_warps=4, num_stages=3, epilogue_num_stages=1)
    compute_dK[grid](Q, K, V, O, dO, L, dK, dV, S_len, scale, attn_scale, BLOCK=64, num_warps=4, num_stages=3, epilogue_num_stages=1)
    compute_dV[grid](Q, K, V, O, dO, L, dV, S_len, scale, attn_scale, BLOCK=64, num_warps=4, num_stages=3, epilogue_num_stages=1)