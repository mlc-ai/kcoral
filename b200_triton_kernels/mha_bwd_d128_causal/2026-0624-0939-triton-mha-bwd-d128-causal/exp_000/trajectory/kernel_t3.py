import torch
import triton
import triton.language as tl
import math


@triton.jit
def _kernel_q(
    Q0, Q1, K0, K1, V0, V1, O0, O1, dO0, dO1, L, dQ0, dQ1,
    S_len, scale, B, H,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    i = tl.program_id(0)
    
    if i >= tl.cdiv(S_len, BLOCK):
        return
    
    b_h_offset = (b * H + h) * S_len * 128
    b_offset_3d = (b * H + h) * S_len
    
    idx_query = i * BLOCK + tl.arange(0, BLOCK)
    mask_q = idx_query < S_len
    
    Q0_tile = tl.load(Q0 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    Q1_tile = tl.load(Q1 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    
    O0_tile = tl.load(O0 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    O1_tile = tl.load(O1 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    
    dO0_tile = tl.load(dO0 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    dO1_tile = tl.load(dO1 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
    
    L_vals = tl.load(L + b_offset_3d + idx_query, mask=mask_q, other=0.0)
    D_vals = tl.sum(O0_tile * dO0_tile, axis=1) + tl.sum(O1_tile * dO1_tile, axis=1)
    
    dQ0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    for j in range(0, i + 1):
        idx_key = j * BLOCK + tl.arange(0, BLOCK)
        mask_k = idx_key < S_len
        
        K0_tile = tl.load(K0 + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
        K1_tile = tl.load(K1 + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
        
        V0_tile = tl.load(V0 + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
        V1_tile = tl.load(V1 + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
        
        causal_mask = (idx_query[:, None] >= idx_key[None, :]) & mask_q[:, None] & mask_k[None, :]
        
        S = tl.dot(Q0_tile, K0_tile.T) + tl.dot(Q1_tile, K1_tile.T)
        P = tl.exp(S * scale - L_vals[:, None])
        P = P * causal_mask
        
        dOV = tl.dot(dO0_tile, V0_tile.T) + tl.dot(dO1_tile, V1_tile.T)
        dP = P * (dOV - D_vals[:, None]) * scale
        
        dQ0_acc += tl.dot(dP, K0_tile)
        dQ1_acc += tl.dot(dP, K1_tile)
    
    dtype = Q0.dtype.element_ty
    out0 = dQ0 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :]
    tl.store(out0, dQ0_acc.to(dtype), mask=mask_q[:, None])
    
    out1 = dQ1 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :]
    tl.store(out1, dQ1_acc.to(dtype), mask=mask_q[:, None])


@triton.jit
def _kernel_kv(
    Q0, Q1, K0, K1, V0, V1, O0, O1, dO0, dO1, L, dK0, dK1, dV0, dV1,
    S_len, scale, B, H,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    j = tl.program_id(0)
    
    if j >= tl.cdiv(S_len, BLOCK):
        return
    
    b_h_offset = (b * H + h) * S_len * 128
    b_offset_3d = (b * H + h) * S_len
    
    idx_key = j * BLOCK + tl.arange(0, BLOCK)
    mask_k = idx_key < S_len
    
    K0_tile = tl.load(K0 + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
    K1_tile = tl.load(K1 + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
    
    V0_tile = tl.load(V0 + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
    V1_tile = tl.load(V1 + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_k[:, None], other=0.0)
    
    dK0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    num_blocks = tl.cdiv(S_len, BLOCK)
    
    for i in range(j, num_blocks):
        idx_query = i * BLOCK + tl.arange(0, BLOCK)
        mask_q = idx_query < S_len
        
        Q0_tile = tl.load(Q0 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        Q1_tile = tl.load(Q1 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        
        O0_tile = tl.load(O0 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        O1_tile = tl.load(O1 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        
        dO0_tile = tl.load(dO0 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        dO1_tile = tl.load(dO1 + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 64)[None, :], mask=mask_q[:, None], other=0.0)
        
        L_vals = tl.load(L + b_offset_3d + idx_query, mask=mask_q, other=0.0)
        D_vals = tl.sum(O0_tile * dO0_tile, axis=1) + tl.sum(O1_tile * dO1_tile, axis=1)
        
        causal_mask = (idx_query[:, None] >= idx_key[None, :]) & mask_q[:, None] & mask_k[None, :]
        
        S = tl.dot(Q0_tile, K0_tile.T) + tl.dot(Q1_tile, K1_tile.T)
        P = tl.exp(S * scale - L_vals[:, None])
        P = P * causal_mask
        
        dOV = tl.dot(dO0_tile, V0_tile.T) + tl.dot(dO1_tile, V1_tile.T)
        dP = P * (dOV - D_vals[:, None]) * scale
        
        dK0_acc += tl.dot(dP.T, Q0_tile)
        dK1_acc += tl.dot(dP.T, Q1_tile)
        
        dV0_acc += tl.dot(P.T, dO0_tile)
        dV1_acc += tl.dot(P.T, dO1_tile)
    
    dtype = K0.dtype.element_ty
    out_k0 = dK0 + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :]
    tl.store(out_k0, dK0_acc.to(dtype), mask=mask_k[:, None])
    
    out_k1 = dK1 + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :]
    tl.store(out_k1, dK1_acc.to(dtype), mask=mask_k[:, None])

    out_v0 = dV0 + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :]
    tl.store(out_v0, dV0_acc.to(dtype), mask=mask_k[:, None])
    
    out_v1 = dV1 + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 64)[None, :]
    tl.store(out_v1, dV1_acc.to(dtype), mask=mask_k[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, d_head = Q.shape
    
    if S_len == 0:
        return
        
    Q0, Q1 = Q, Q[:, :, :, 64:]
    K0, K1 = K, K[:, :, :, 64:]
    V0, V1 = V, V[:, :, :, 64:]
    O0, O1 = O, O[:, :, :, 64:]
    dO0, dO1 = dO, dO[:, :, :, 64:]
    dQ0, dQ1 = dQ, dQ[:, :, :, 64:]
    dK0, dK1 = dK, dK[:, :, :, 64:]
    dV0, dV1 = dV, dV[:, :, :, 64:]
    
    num_blocks = triton.cdiv(S_len, 64)
    grid = (num_blocks, H, B)
    
    scale = 1.0 / math.sqrt(d_head)
    
    _KERNEL_KV[grid](
        Q0, Q1, K0, K1, V0, V1, O0, O1, dO0, dO1, L, dK0, dK1, dV0, dV1, S_len, scale, B, H,
        BLOCK=64, num_warps=4, num_stages=3
    )
    _KERNEL_Q[grid](
        Q0, Q1, K0, K1, V0, V1, O0, O1, dO0, dO1, L, dQ0, dQ1, S_len, scale, B, H,
        BLOCK=64, num_warps=4, num_stages=3
    )