import torch
import triton
import triton.language as tl
import math


@triton.jit
def _compute_D(
    O, dO, D,
    S_len,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(1)
    i = tl.program_id(0)
    
    if i >= tl.cdiv(S_len, BLOCK):
        return
    
    b_h_offset = off_b * S_len * 128
    
    idx = i * BLOCK + tl.arange(0, BLOCK)
    mask = idx < S_len
    
    O0 = tl.load(O + b_h_offset + idx[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask[:, None], other=0.0)
    O1 = tl.load(O + b_h_offset + idx[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask[:, None], other=0.0)
    
    dO0 = tl.load(dO + b_h_offset + idx[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask[:, None], other=0.0)
    dO1 = tl.load(dO + b_h_offset + idx[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask[:, None], other=0.0)
    
    D_val = tl.sum(O0 * dO0 + O1 * dO1, axis=1)
    
    tl.store(D + off_b * S_len + idx, D_val, mask=mask)


@triton.jit
def _compute_dQ(
    Q, K, V, O, dO, L, D, dQ,
    S_len,
    scale,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(1)
    i = tl.program_id(0)
    
    if i >= tl.cdiv(S_len, BLOCK):
        return
    
    b_h_offset = off_b * S_len * 128
    b_h_offset_s = off_b * S_len
    
    idx_query = i * BLOCK + tl.arange(0, BLOCK)
    mask_q = idx_query < S_len
    
    Q0 = tl.load(Q + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    Q1 = tl.load(Q + b_h_offset + idx_query[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    
    dO0 = tl.load(dO + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    dO1 = tl.load(dO + b_h_offset + idx_query[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    
    L_vals = tl.load(L + b_h_offset_s + idx_query, mask=mask_q, other=0.0)
    D_vals = tl.load(D + b_h_offset_s + idx_query, mask=mask_q, other=0.0)
    
    dQ0_acc = tl.zeros((64, 64), tl.float32)
    dQ1_acc = tl.zeros((64, 64), tl.float32)
    
    for j in range(0, i + 1):
        idx_key = j * BLOCK + tl.arange(0, BLOCK)
        mask_k = idx_key < S_len
        
        K0 = tl.load(K + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
        K1 = tl.load(K + b_h_offset + idx_key[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
        
        V0 = tl.load(V + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
        V1 = tl.load(V + b_h_offset + idx_key[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
        
        causal_mask = (idx_query[:, None] >= idx_key[None, :]) & mask_q[:, None] & mask_k[None, :]
        
        S = tl.dot(Q0, K0.T, input_precision="tf32", out_dtype=tl.float32) + \
            tl.dot(Q1, K1.T, input_precision="tf32", out_dtype=tl.float32)
        
        P = tl.exp(S * scale - L_vals[:, None])
        P = P * causal_mask
        
        dOV = tl.dot(dO0, V0.T, input_precision="tf32", out_dtype=tl.float32) + \
              tl.dot(dO1, V1.T, input_precision="tf32", out_dtype=tl.float32)
        
        dP = P * (dOV - D_vals[:, None]) * scale
        
        dQ0_acc += tl.dot(dP, K0, acc=dQ0_acc, input_precision="tf32", out_dtype=tl.float32)
        dQ1_acc += tl.dot(dP, K1, acc=dQ1_acc, input_precision="tf32", out_dtype=tl.float32)
    
    dtype = Q.dtype.element_ty
    out0 = dQ + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :]
    tl.store(out0, dQ0_acc.to(dtype), mask=mask_q[:, None])
    
    out1 = dQ + b_h_offset + idx_query[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    tl.store(out1, dQ1_acc.to(dtype), mask=mask_q[:, None])


@triton.jit
def _compute_dK_dV(
    Q, K, V, O, dO, L, D, dK, dV,
    S_len,
    scale,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(1)
    j = tl.program_id(0)
    
    if j >= tl.cdiv(S_len, BLOCK):
        return
    
    b_h_offset = off_b * S_len * 128
    b_h_offset_s = off_b * S_len
    
    idx_key = j * BLOCK + tl.arange(0, BLOCK)
    mask_k = idx_key < S_len
    
    K0 = tl.load(K + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
    K1 = tl.load(K + b_h_offset + idx_key[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
    
    V0 = tl.load(V + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
    V1 = tl.load(V + b_h_offset + idx_key[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
    
    dK0_acc = tl.zeros((64, 64), tl.float32)
    dK1_acc = tl.zeros((64, 64), tl.float32)
    dV0_acc = tl.zeros((64, 64), tl.float32)
    dV1_acc = tl.zeros((64, 64), tl.float32)
    
    num_blocks = tl.cdiv(S_len, BLOCK)
    
    for i in range(j, num_blocks):
        idx_query = i * BLOCK + tl.arange(0, BLOCK)
        mask_q = idx_query < S_len
        
        Q0 = tl.load(Q + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        Q1 = tl.load(Q + b_h_offset + idx_query[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        
        dO0 = tl.load(dO + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        dO1 = tl.load(dO + b_h_offset + idx_query[:, None] * 128 + 64 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        
        L_vals = tl.load(L + b_h_offset_s + idx_query, mask=mask_q, other=0.0)
        D_vals = tl.load(D + b_h_offset_s + idx_query, mask=mask_q, other=0.0)
        
        causal_mask = (idx_query[:, None] >= idx_key[None, :]) & mask_q[:, None] & mask_k[None, :]
        
        S = tl.dot(Q0, K0.T, input_precision="tf32", out_dtype=tl.float32) + \
            tl.dot(Q1, K1.T, input_precision="tf32", out_dtype=tl.float32)
        
        P = tl.exp(S * scale - L_vals[:, None])
        P = P * causal_mask
        
        dOV = tl.dot(dO0, V0.T, input_precision="tf32", out_dtype=tl.float32) + \
              tl.dot(dO1, V1.T, input_precision="tf32", out_dtype=tl.float32)
        
        dP = P * (dOV - D_vals[:, None]) * scale
        
        dK0_acc += tl.dot(dP.T, Q0, acc=dK0_acc, input_precision="tf32", out_dtype=tl.float32)
        dK1_acc += tl.dot(dP.T, Q1, acc=dK1_acc, input_precision="tf32", out_dtype=tl.float32)
        
        dV0_acc += tl.dot(P.T, dO0, acc=dV0_acc, input_precision="tf32", out_dtype=tl.float32)
        dV1_acc += tl.dot(P.T, dO1, acc=dV1_acc, input_precision="tf32", out_dtype=tl.float32)
    
    dtype = K.dtype.element_ty
    out_k0 = dK + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :]
    tl.store(out_k0, dK0_acc.to(dtype), mask=mask_k[:, None])
    
    out_k1 = dK + b_h_offset + idx_key[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    tl.store(out_k1, dK1_acc.to(dtype), mask=mask_k[:, None])

    out_v0 = dV + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :]
    tl.store(out_v0, dV0_acc.to(dtype), mask=mask_k[:, None])
    
    out_v1 = dV + b_h_offset + idx_key[:, None] * 128 + 64 + tl.arange(0, 64)[None, :]
    tl.store(out_v1, dV1_acc.to(dtype), mask=mask_k[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, d_head = Q.shape
    
    if S_len == 0:
        return
        
    num_blocks = triton.cdiv(S_len, 64)
    grid = (num_blocks, B * H)
    
    scale = 1.0 / math.sqrt(d_head)
    
    D = torch.zeros((B, H, S_len), device=Q.device, dtype=torch.float32)
    
    _compute_D[grid](O, dO, D, S_len, BLOCK=64, num_warps=4, num_stages=3)
    _compute_dQ[grid](Q, K, V, O, dO, L, D, dQ, S_len, scale, BLOCK=64, num_warps=4, num_stages=3)
    _compute_dK_dV[grid](Q, K, V, O, dO, L, D, dK, dV, S_len, scale, BLOCK=64, num_warps=4, num_stages=3)