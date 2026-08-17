import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _precompute_D(
    O0_desc, O1_desc, dO0_desc, dO1_desc, D,
    S_len, H,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(1)
    i = tl.program_id(0)
    
    if i >= tl.cdiv(S_len, BLOCK):
        return
        
    b = off_b // H
    h = off_b % H
    
    idx = i * BLOCK + tl.arange(0, BLOCK)
    mask = idx < S_len
    
    O0 = O0_desc.load([b, h, i*BLOCK, 0])
    O1 = O1_desc.load([b, h, i*BLOCK, 0])
    dO0 = dO0_desc.load([b, h, i*BLOCK, 0])
    dO1 = dO1_desc.load([b, h, i*BLOCK, 0])
    
    D_val = tl.sum(O0 * dO0 + O1 * dO1, axis=1)
    
    tl.store(D + off_b * S_len + idx, D_val, mask=mask)


@triton.jit
def _kernel_q(
    Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc, L, dQ0_desc, dQ1_desc, D,
    S_len, scale, H,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    i = tl.program_id(0)
    
    if i >= tl.cdiv(S_len, BLOCK):
        return
    
    idx_query = i * BLOCK + tl.arange(0, BLOCK)
    mask_q = idx_query < S_len
    
    Q0 = Q0_desc.load([b, h, i*BLOCK, 0])
    Q1 = Q1_desc.load([b, h, i*BLOCK, 0])
    
    dO0 = dO0_desc.load([b, h, i*BLOCK, 0])
    dO1 = dO1_desc.load([b, h, i*BLOCK, 0])
    
    L_vals = tl.load(L + (b*H+h)*S_len + idx_query, mask=mask_q, other=0.0)
    D_vals = tl.load(D + (b*H+h)*S_len + idx_query, mask=mask_q, other=0.0)
    
    dQ0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    for j in range(0, i + 1):
        K0 = K0_desc.load([b, h, j*BLOCK, 0])
        K1 = K1_desc.load([b, h, j*BLOCK, 0])
        V0 = V0_desc.load([b, h, j*BLOCK, 0])
        V1 = V1_desc.load([b, h, j*BLOCK, 0])
        
        idx_key = j * BLOCK + tl.arange(0, BLOCK)
        causal_mask = (idx_query[:, None] >= idx_key[None, :]) & mask_q[:, None] & (idx_key[None, :] < S_len)
        
        S = tl.dot(Q0, K0.T, out_dtype=tl.float32) + tl.dot(Q1, K1.T, out_dtype=tl.float32)
        P = tl.exp(S * scale - L_vals[:, None])
        P = P * causal_mask
        
        dOV = tl.dot(dO0, V0.T, out_dtype=tl.float32) + tl.dot(dO1, V1.T, out_dtype=tl.float32)
        dS = (P * (dOV - D_vals[:, None]) * scale).to(tl.bfloat16)
        
        dQ0_acc = tl.dot(dS, K0, acc=dQ0_acc, out_dtype=tl.float32)
        dQ1_acc = tl.dot(dS, K1, acc=dQ1_acc, out_dtype=tl.float32)
    
    dQ0_desc.store([b, h, i*BLOCK, 0], dQ0_acc)
    dQ1_desc.store([b, h, i*BLOCK, 0], dQ1_acc)


@triton.jit
def _kernel_kv(
    Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc, L, dO0_desc, dO1_desc, D, dK0_desc, dK1_desc, dV0_desc, dV1_desc,
    S_len, scale, H,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    j = tl.program_id(0)
    
    if j >= tl.cdiv(S_len, BLOCK):
        return
    
    idx_key = j * BLOCK + tl.arange(0, BLOCK)
    
    K0 = K0_desc.load([b, h, j*BLOCK, 0])
    K1 = K1_desc.load([b, h, j*BLOCK, 0])
    V0 = V0_desc.load([b, h, j*BLOCK, 0])
    V1 = V1_desc.load([b, h, j*BLOCK, 0])
    
    dK0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    num_blocks = tl.cdiv(S_len, BLOCK)
    
    for i in range(j, num_blocks):
        Q0 = Q0_desc.load([b, h, i*BLOCK, 0])
        Q1 = Q1_desc.load([b, h, i*BLOCK, 0])
        
        dO0 = dO0_desc.load([b, h, i*BLOCK, 0])
        dO1 = dO1_desc.load([b, h, i*BLOCK, 0])
        
        idx_query = i * BLOCK + tl.arange(0, BLOCK)
        mask_q = idx_query < S_len
        
        L_vals = tl.load(L + (b*H+h)*S_len + idx_query, mask=mask_q, other=0.0)
        D_vals = tl.load(D + (b*H+h)*S_len + idx_query, mask=mask_q, other=0.0)
        
        causal_mask = (idx_query[:, None] >= idx_key[None, :]) & mask_q[:, None] & (idx_key[None, :] < S_len)
        
        S = tl.dot(Q0, K0.T, out_dtype=tl.float32) + tl.dot(Q1, K1.T, out_dtype=tl.float32)
        P = tl.exp(S * scale - L_vals[:, None])
        P = P * causal_mask
        
        dOV = tl.dot(dO0, V0.T, out_dtype=tl.float32) + tl.dot(dO1, V1.T, out_dtype=tl.float32)
        dS = P * (dOV - D_vals[:, None]) * scale
        dS_cast = dS.to(tl.bfloat16)
        
        dK0_acc = tl.dot(dS_cast.T, Q0, acc=dK0_acc, out_dtype=tl.float32)
        dK1_acc = tl.dot(dS_cast.T, Q1, acc=dK1_acc, out_dtype=tl.float32)
        
        P_cast = (P * causal_mask).to(tl.bfloat16)
        dV0_acc = tl.dot(P_cast.T, dO0, acc=dV0_acc, out_dtype=tl.float32)
        dV1_acc = tl.dot(P_cast.T, dO1, acc=dV1_acc, out_dtype=tl.float32)
    
    dK0_desc.store([b, h, j*BLOCK, 0], dK0_acc)
    dK1_desc.store([b, h, j*BLOCK, 0], dK1_acc)
    dV0_desc.store([b, h, j*BLOCK, 0], dV0_acc)
    dV1_desc.store([b, h, j*BLOCK, 0], dV1_acc)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, d_head = Q.shape
    
    if S_len == 0:
        return
        
    Q0 = Q[..., :64]
    Q1 = Q[..., 64:]
    K0 = K[..., :64]
    K1 = K[..., 64:]
    V0 = V[..., :64]
    V1 = V[..., 64:]
    O0 = O[..., :64]
    O1 = O[..., 64:]
    dO0 = dO[..., :64]
    dO1 = dO[..., 64:]
    dQ0 = dQ[..., :64]
    dQ1 = dQ[..., 64:]
    dK0 = dK[..., :64]
    dK1 = dK[..., 64:]
    dV0 = dV[..., :64]
    dV1 = dV[..., 64:]
    
    Q0_desc = TensorDescriptor.from_tensor(Q0, [64, 64])
    Q1_desc = TensorDescriptor.from_tensor(Q1, [64, 64])
    K0_desc = TensorDescriptor.from_tensor(K0, [64, 64])
    K1_desc = TensorDescriptor.from_tensor(K1, [64, 64])
    V0_desc = TensorDescriptor.from_tensor(V0, [64, 64])
    V1_desc = TensorDescriptor.from_tensor(V1, [64, 64])
    O0_desc = TensorDescriptor.from_tensor(O0, [64, 64])
    O1_desc = TensorDescriptor.from_tensor(O1, [64, 64])
    dO0_desc = TensorDescriptor.from_tensor(dO0, [64, 64])
    dO1_desc = TensorDescriptor.from_tensor(dO1, [64, 64])
    dQ0_desc = TensorDescriptor.from_tensor(dQ0, [64, 64])
    dQ1_desc = TensorDescriptor.from_tensor(dQ1, [64, 64])
    dK0_desc = TensorDescriptor.from_tensor(dK0, [64, 64])
    dK1_desc = TensorDescriptor.from_tensor(dK1, [64, 64])
    dV0_desc = TensorDescriptor.from_tensor(dV0, [64, 64])
    dV1_desc = TensorDescriptor.from_tensor(dV1, [64, 64])
    
    num_blocks = triton.cdiv(S_len, 64)
    grid_d = (num_blocks, B * H)
    grid = (num_blocks, H, B)
    
    scale = 1.0 / math.sqrt(d_head)
    
    D = torch.zeros((B, H, S_len), device=Q.device, dtype=torch.float32)
    
    _precompute_D[grid_d](
        O0_desc, O1_desc, dO0_desc, dO1_desc, D, S_len, H,
        BLOCK=64, num_warps=4, num_stages=3
    )
    _kernel_kv[grid](
        Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc, L, dO0_desc, dO1_desc, D, dK0_desc, dK1_desc, dV0_desc, dV1_desc,
        S_len, scale, H,
        BLOCK=64, num_warps=4, num_stages=3
    )
    _kernel_q[grid](
        Q0_desc, Q1_desc, K0_desc, K1_desc, V0_desc, V1_desc, L, dQ0_desc, dQ1_desc, D,
        S_len, scale, H,
        BLOCK=64, num_warps=4, num_stages=3
    )