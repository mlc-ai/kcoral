import torch
import triton
import triton.language as tl
import math


@triton.jit
def _kernel_kv(
    Q, K, V, O, dO, L, dK, dV,
    S_len, scale,
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
    
    K0 = tl.load(K + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_k[:, None], other=0.0)
    V0 = tl.load(V + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_k[:, None], other=0.0)
    
    dK0_acc = tl.zeros((BLOCK, 128), tl.float32)
    dV0_acc = tl.zeros((BLOCK, 128), tl.float32)
    
    num_blocks = tl.cdiv(S_len, BLOCK)
    
    for i in range(j, num_blocks):
        idx_query = i * BLOCK + tl.arange(0, BLOCK)
        mask_q = idx_query < S_len
        
        Q0 = tl.load(Q + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_q[:, None], other=0.0)
        dO0 = tl.load(dO + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_q[:, None], other=0.0)
        O0 = tl.load(O + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_q[:, None], other=0.0)
        
        L_vals = tl.load(L + b_h_offset_s + idx_query, mask=mask_q, other=0.0)
        D_vals = tl.sum(O0 * dO0, axis=1)
        
        causal_mask = (idx_query[:, None] >= idx_key[None, :]) & mask_q[:, None] & mask_k[None, :]
        
        S = tl.dot(Q0, K0.T) * scale
        P = tl.exp(S - L_vals[:, None])
        P = P * causal_mask
        
        dOV = tl.dot(dO0, V0.T)
        dP = P * (dOV - D_vals[:, None]) * scale
        
        dK0_acc += tl.dot(dP.T, Q0)
        dV0_acc += tl.dot(P.T, dO0)
    
    dtype = K.dtype.element_ty
    tl.store(dK + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 128)[None, :], dK0_acc.to(dtype), mask=mask_k[:, None])
    tl.store(dV + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 128)[None, :], dV0_acc.to(dtype), mask=mask_k[:, None])


@triton.jit
def _kernel_q(
    Q, K, V, O, dO, L, dQ,
    S_len, scale,
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
    
    Q0 = tl.load(Q + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_q[:, None], other=0.0)
    dO0 = tl.load(dO + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_q[:, None], other=0.0)
    O0 = tl.load(O + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_q[:, None], other=0.0)
    
    L_vals = tl.load(L + b_h_offset_s + idx_query, mask=mask_q, other=0.0)
    D_vals = tl.sum(O0 * dO0, axis=1)
    
    dQ0_acc = tl.zeros((BLOCK, 128), tl.float32)
    
    for j in range(0, i + 1):
        idx_key = j * BLOCK + tl.arange(0, BLOCK)
        mask_k = idx_key < S_len
        
        K0 = tl.load(K + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_k[:, None], other=0.0)
        V0 = tl.load(V + b_h_offset + idx_key[:, None] * 128 + tl.arange(0, 128)[None, :], mask=mask_k[:, None], other=0.0)
        
        causal_mask = (idx_query[:, None] >= idx_key[None, :]) & mask_q[:, None] & mask_k[None, :]
        
        S = tl.dot(Q0, K0.T) * scale
        P = tl.exp(S - L_vals[:, None])
        P = P * causal_mask
        
        dOV = tl.dot(dO0, V0.T)
        dP = P * (dOV - D_vals[:, None]) * scale
        
        dQ0_acc += tl.dot(dP, K0)
    
    dtype = Q.dtype.element_ty
    tl.store(dQ + b_h_offset + idx_query[:, None] * 128 + tl.arange(0, 128)[None, :], dQ0_acc.to(dtype), mask=mask_q[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, d_head = Q.shape
    
    if S_len == 0:
        return
        
    num_blocks = triton.cdiv(S_len, 128)
    grid = (num_blocks, B * H)
    
    scale = 1.0 / math.sqrt(d_head)
    
    _kernel_kv[grid](
        Q, K, V, O, dO, L, dK, dV, S_len, scale,
        BLOCK=128, num_warps=8, num_stages=3
    )
    _kernel_q[grid](
        Q, K, V, O, dO, L, dQ, S_len, scale,
        BLOCK=128, num_warps=8, num_stages=3
    )