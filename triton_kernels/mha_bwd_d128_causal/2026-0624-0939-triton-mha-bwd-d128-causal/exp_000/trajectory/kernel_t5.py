import torch
import triton
import triton.language as tl
import math


@triton.jit
def _kernel_q(
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
    Q0 = tl.load(base_q + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0).to(tl.float32)
    Q1 = tl.load(base_q + tl.arange(0, BLOCK)[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0).to(tl.float32)
    
    base_o = O + b_offset + h_offset + row_start * 128
    O0 = tl.load(base_o + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0).to(tl.float32)
    O1 = tl.load(base_o + tl.arange(0, BLOCK)[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0).to(tl.float32)
    
    base_do = dO + b_offset + h_offset + row_start * 128
    dO0 = tl.load(base_do + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0).to(tl.float32)
    dO1 = tl.load(base_do + tl.arange(0, BLOCK)[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0).to(tl.float32)
    
    L_vals = tl.load(L + b_offset_L + h_offset_L + idx_query, mask=mask_q, other=0.0)
    D_vals = tl.sum(O0 * dO0 + O1 * dO1, axis=1)
    
    dQ0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    for j in range(0, i + 1):
        key_start = j * BLOCK
        idx_key = key_start + tl.arange(0, BLOCK)
        mask_k = idx_key < S_len
        
        base_k = K + b_offset + h_offset + key_start * 128
        K0 = tl.load(base_k + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0).to(tl.float32)
        K1 = tl.load(base_k + tl.arange(0, BLOCK)[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0).to(tl.float32)
        
        base_v = V + b_offset + h_offset + key_start * 128
        V0 = tl.load(base_v + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0).to(tl.float32)
        V1 = tl.load(base_v + tl.arange(0, BLOCK)[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0).to(tl.float32)
        
        causal_mask = (idx_query[:, None] >= idx_key[None, :]) & mask_q[:, None] & mask_k[None, :]
        
        S = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        P = tl.exp(S * scale - L_vals[:, None])
        P = P * causal_mask
        
        dOV = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        dP = P * (dOV - D_vals[:, None]) * scale
        
        dQ0_acc += tl.dot(dP, K0)
        dQ1_acc += tl.dot(dP, K1)
    
    out_base = dQ + b_offset + h_offset + row_start * 128
    tl.store(out_base + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 64)[None, :], dQ0_acc.to(tl.bfloat16), mask=mask_q[:, None])
    tl.store(out_base + tl.arange(0, BLOCK)[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], dQ1_acc.to(tl.bfloat16), mask=mask_q[:, None])


@triton.jit
def _kernel_kv(
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
    K0 = tl.load(base_k + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0).to(tl.float32)
    K1 = tl.load(base_k + tl.arange(0, BLOCK)[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0).to(tl.float32)
    
    base_v = V + b_offset + h_offset + key_start * 128
    V0 = tl.load(base_v + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0).to(tl.float32)
    V1 = tl.load(base_v + tl.arange(0, BLOCK)[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0).to(tl.float32)
    
    dK0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    num_blocks = tl.cdiv(S_len, BLOCK)
    
    for i in range(j, num_blocks):
        row_start = i * BLOCK
        idx_query = row_start + tl.arange(0, BLOCK)
        mask_q = idx_query < S_len
        
        base_q = Q + b_offset + h_offset + row_start * 128
        Q0 = tl.load(base_q + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0).to(tl.float32)
        Q1 = tl.load(base_q + tl.arange(0, BLOCK)[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0).to(tl.float32)
        
        base_o = O + b_offset + h_offset + row_start * 128
        O0 = tl.load(base_o + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0).to(tl.float32)
        O1 = tl.load(base_o + tl.arange(0, BLOCK)[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0).to(tl.float32)
        
        base_do = dO + b_offset + h_offset + row_start * 128
        dO0 = tl.load(base_do + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0).to(tl.float32)
        dO1 = tl.load(base_do + tl.arange(0, BLOCK)[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0).to(tl.float32)
        
        L_vals = tl.load(L + b_offset_L + h_offset_L + idx_query, mask=mask_q, other=0.0)
        D_vals = tl.sum(O0 * dO0 + O1 * dO1, axis=1)
        
        causal_mask = (idx_query[:, None] >= idx_key[None, :]) & mask_q[:, None] & mask_k[None, :]
        
        S = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        P = tl.exp(S * scale - L_vals[:, None])
        P = P * causal_mask
        
        dOV = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        dP = P * (dOV - D_vals[:, None]) * scale
        
        dK0_acc += tl.dot(dP.T, Q0)
        dK1_acc += tl.dot(dP.T, Q1)
        
        dV0_acc += tl.dot(P.T, dO0)
        dV1_acc += tl.dot(P.T, dO1)
    
    out_base_k = dK + b_offset + h_offset + key_start * 128
    tl.store(out_base_k + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 64)[None, :], dK0_acc.to(tl.bfloat16), mask=mask_k[:, None])
    tl.store(out_base_k + tl.arange(0, BLOCK)[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], dK1_acc.to(tl.bfloat16), mask=mask_k[:, None])
    
    out_base_v = dV + b_offset + h_offset + key_start * 128
    tl.store(out_base_v + tl.arange(0, BLOCK)[:, None] * 128 + tl.arange(0, 64)[None, :], dV0_acc.to(tl.bfloat16), mask=mask_k[:, None])
    tl.store(out_base_v + tl.arange(0, BLOCK)[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], dV1_acc.to(tl.bfloat16), mask=mask_k[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, d_head = Q.shape
    
    if S_len == 0:
        return
        
    num_blocks = triton.cdiv(S_len, 64)
    grid = (num_blocks, H, B)
    
    scale = 1.0 / math.sqrt(d_head)
    
    _kernel_kv[grid](
        Q, K, V, O, dO, L, dK, dV, S_len, scale, B, H,
        BLOCK=64, num_warps=4, num_stages=3
    )
    _kernel_q[grid](
        Q, K, V, O, dO, L, dQ, S_len, scale, B, H,
        BLOCK=64, num_warps=4, num_stages=3
    )