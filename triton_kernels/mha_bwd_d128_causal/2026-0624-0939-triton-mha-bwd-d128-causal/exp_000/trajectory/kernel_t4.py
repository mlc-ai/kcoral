import torch
import triton
import triton.language as tl
import math


@triton.jit
def _epilogue_q(
    Q, K, V, O, dO, L, dQ,
    S_len, scale, B, H,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    i = tl.program_id(0)
    
    if i >= tl.cdiv(S_len, BLOCK):
        return
    
    b_offset = b * H * S_len * 128
    h_offset = h * S_len * 128
    b_offset_L = b * H * S_len
    h_offset_L = h * S_len
    
    row_start = i * BLOCK
    idx_query = row_start + tl.arange(0, BLOCK)
    mask_q = idx_query < S_len
    
    base_q = Q + b_offset + h_offset + row_start * 128
    Q_tile = tl.load(base_q + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_q[:, None], other=0.0)
    
    base_o = O + b_offset + h_offset + row_start * 128
    O_tile = tl.load(base_o + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_q[:, None], other=0.0)
    
    base_do = dO + b_offset + h_offset + row_start * 128
    dO_tile = tl.load(base_do + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_q[:, None], other=0.0)
    
    L_vals = tl.load(L + b_offset_L + h_offset_L + idx_query, mask=mask_q, other=0.0)
    D_vals = tl.sum(O_tile * dO_tile, axis=1)
    
    dQ_acc = tl.zeros((BLOCK, 128), tl.float32)
    
    for j in range(0, i + 1):
        key_start = j * BLOCK
        idx_key = key_start + tl.arange(0, BLOCK)
        mask_k = idx_key < S_len
        
        base_k = K + b_offset + h_offset + key_start * 128
        K_tile = tl.load(base_k + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_k[:, None], other=0.0)
        
        base_v = V + b_offset + h_offset + key_start * 128
        V_tile = tl.load(base_v + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_k[:, None], other=0.0)
        
        causal_mask = (idx_query[:, None] >= idx_key[None, :]) & mask_q[:, None] & mask_k[None, :]
        
        S = tl.dot(Q_tile, K_tile.T, out_dtype=tl.float32) * scale
        P = tl.exp(S - L_vals[:, None])
        P = P * causal_mask
        
        dOV = tl.dot(dO_tile, V_tile.T, out_dtype=tl.float32)
        dP = P * (dOV - D_vals[:, None]) * scale
        
        dQ_acc += tl.dot(dP, K_tile, acc=dQ_acc, out_dtype=tl.float32)
    
    out_base = dQ + b_offset + h_offset + row_start * 128
    tl.store(out_base + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 128)[None, :], dQ_acc.to(tl.bfloat16), mask=mask_q[:, None])


@triton.jit
def _epilogue_kv(
    Q, K, V, O, dO, L, dK, dV,
    S_len, scale, B, H,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    j = tl.program_id(0)
    
    if j >= tl.cdiv(S_len, BLOCK):
        return
    
    b_offset = b * H * S_len * 128
    h_offset = h * S_len * 128
    b_offset_L = b * H * S_len
    h_offset_L = h * S_len
    
    key_start = j * BLOCK
    idx_key = key_start + tl.arange(0, BLOCK)
    mask_k = idx_key < S_len
    
    base_k = K + b_offset + h_offset + key_start * 128
    K_tile = tl.load(base_k + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_k[:, None], other=0.0)
    
    base_v = V + b_offset + h_offset + key_start * 128
    V_tile = tl.load(base_v + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_k[:, None], other=0.0)
    
    dK_acc = tl.zeros((BLOCK, 128), tl.float32)
    dV_acc = tl.zeros((BLOCK, 128), tl.float32)
    
    num_blocks = tl.cdiv(S_len, BLOCK)
    
    for i in range(j, num_blocks):
        row_start = i * BLOCK
        idx_query = row_start + tl.arange(0, BLOCK)
        mask_q = idx_query < S_len
        
        base_q = Q + b_offset + h_offset + row_start * 128
        Q_tile = tl.load(base_q + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_q[:, None], other=0.0)
        
        base_o = O + b_offset + h_offset + row_start * 128
        O_tile = tl.load(base_o + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_q[:, None], other=0.0)
        
        base_do = dO + b_offset + h_offset + row_start * 128
        dO_tile = tl.load(base_do + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_q[:, None], other=0.0)
        
        L_vals = tl.load(L + b_offset_L + h_offset_L + idx_query, mask=mask_q, other=0.0)
        D_vals = tl.sum(O_tile * dO_tile, axis=1)
        
        causal_mask = (idx_query[:, None] >= idx_key[None, :]) & mask_q[:, None] & mask_k[None, :]
        
        S = tl.dot(Q_tile, K_tile.T, out_dtype=tl.float32) * scale
        P = tl.exp(S - L_vals[:, None])
        P = P * causal_mask
        
        dOV = tl.dot(dO_tile, V_tile.T, out_dtype=tl.float32)
        dP = P * (dOV - D_vals[:, None]) * scale
        
        dK_acc += tl.dot(dP.T, Q_tile, acc=dK_acc, out_dtype=tl.float32)
        dV_acc += tl.dot(P.T, dO_tile, acc=dV_acc, out_dtype=tl.float32)
    
    out_base_k = dK + b_offset + h_offset + key_start * 128
    tl.store(out_base_k + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 128)[None, :], dK_acc.to(tl.bfloat16), mask=mask_k[:, None])
    
    out_base_v = dV + b_offset + h_offset + key_start * 128
    tl.store(out_base_v + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 128)[None, :], dV_acc.to(tl.bfloat16), mask=mask_k[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, d_head = Q.shape
    
    if S_len == 0:
        return
        
    num_blocks = triton.cdiv(S_len, 128)
    grid = (num_blocks, H, B)
    
    scale = 1.0 / math.sqrt(d_head)
    
    _epilogue_kv[grid](
        Q, K, V, O, dO, L, dK, dV, S_len, scale, B, H,
        BLOCK=128, num_warps=8, num_stages=2
    )
    _epilogue_q[grid](
        Q, K, V, O, dO, L, dQ, S_len, scale, B, H,
        BLOCK=128, num_warps=4, num_stages=3
    )