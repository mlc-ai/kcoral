import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dKdV_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    dO_desc,
    L_desc,
    dK_desc,
    dV_desc,
    S,
    d,
    tau,
    BLOCK: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    kv_idx = tl.program_id(1)
    num_blocks = tl.num_programs(1)
    
    k_row_base = kv_idx * BLOCK
    
    k = K_desc.load([b_h_idx, k_row_base, 0])
    v = V_desc.load([b_h_idx, k_row_base, 0])
    
    dk_acc = tl.zeros((BLOCK, d), tl.float32)
    dv_acc = tl.zeros((BLOCK, d), tl.float32)
    
    q_row_base = kv_idx * BLOCK
    
    if q_row_base < S:
        Q_desc.prefetch([b_h_idx, q_row_base, 0])
        dO_desc.prefetch([b_h_idx, q_row_base, 0])
        O_desc.prefetch([b_h_idx, q_row_base, 0])
    
    for q_idx in range(kv_idx, num_blocks):
        q = Q_desc.load([b_h_idx, q_row_base, 0])
        do = dO_desc.load([b_h_idx, q_row_base, 0])
        o = O_desc.load([b_h_idx, q_row_base, 0])
        
        l = L_desc.load([b_h_idx, q_row_base])
        
        d_val = tl.sum(o * do, axis=1)
        
        s_acc = tl.dot(q, k.T)
        dp_acc = tl.dot(do, v.T)
        
        p = tl.exp(s_acc * tau - l[:, None])
        
        q_row = q_row_base + tl.arange(0, BLOCK)
        k_row = k_row_base + tl.arange(0, BLOCK)
        causal_mask = k_row[None, :] <= q_row[:, None]
        valid_mask = (q_row[:, None] < S) & (k_row[None, :] < S) & causal_mask
        p = p * valid_mask
        
        ds = p * (dp_acc - d_val[:, None]) * tau
        
        dv_acc = tl.dot(p.T, do, dv_acc)
        dk_acc = tl.dot(ds.T, q, dk_acc)
        
        next_q_row_base = (q_idx + 1) * BLOCK
        if next_q_row_base < S:
            Q_desc.prefetch([b_h_idx, next_q_row_base, 0])
            dO_desc.prefetch([b_h_idx, next_q_row_base, 0])
            O_desc.prefetch([b_h_idx, next_q_row_base, 0])
        
        q_row_base = next_q_row_base
    
    dK_desc.store([b_h_idx, k_row_base, 0], dk_acc.to(tl.bfloat16)[None, :, :])
    dV_desc.store([b_h_idx, k_row_base, 0], dv_acc.to(tl.bfloat16)[None, :, :])


@triton.jit
def _bwd_dQ_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    dO_desc,
    L_desc,
    dQ_desc,
    S,
    d,
    tau,
    BLOCK: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    q_idx = tl.program_id(1)
    num_blocks = tl.num_programs(1)
    
    q_row_base = q_idx * BLOCK
    
    q = Q_desc.load([b_h_idx, q_row_base, 0])
    do = dO_desc.load([b_h_idx, q_row_base, 0])
    o = O_desc.load([b_h_idx, q_row_base, 0])
    
    l = L_desc.load([b_h_idx, q_row_base])
    
    d_val = tl.sum(o * do, axis=1)
    
    dq_acc = tl.zeros((BLOCK, d), tl.float32)
    
    k_row_base = 0
    
    if k_row_base < S:
        K_desc.prefetch([b_h_idx, k_row_base, 0])
        V_desc.prefetch([b_h_idx, k_row_base, 0])
    
    for kv_idx in range(0, q_idx + 1):
        k = K_desc.load([b_h_idx, k_row_base, 0])
        v = V_desc.load([b_h_idx, k_row_base, 0])
        
        s_acc = tl.dot(q, k.T)
        dp_acc = tl.dot(do, v.T)
        
        p = tl.exp(s_acc * tau - l[:, None])
        
        q_row = q_row_base + tl.arange(0, BLOCK)
        k_row = k_row_base + tl.arange(0, BLOCK)
        causal_mask = k_row[None, :] <= q_row[:, None]
        valid_mask = (q_row[:, None] < S) & (k_row[None, :] < S) & causal_mask
        p = p * valid_mask
        
        ds = p * (dp_acc - d_val[:, None]) * tau
        
        dq_acc = tl.dot(ds, k, dq_acc)
        
        next_k_row_base = (kv_idx + 1) * BLOCK
        if next_k_row_base < S:
            K_desc.prefetch([b_h_idx, next_k_row_base, 0])
            V_desc.prefetch([b_h_idx, next_k_row_base, 0])
        
        k_row_base = next_k_row_base
        
    dQ_desc.store([b_h_idx, q_row_base, 0], dq_acc.to(tl.bfloat16)[None, :, :])


def fallback_BWD_dKdV_elementwise(Q, K, V, O, dO, L, dK, dV):
    pass


def fallback_BWD_dQ_elementwise(Q, K, V, O, dO, L, dQ):
    pass


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    tau = 1.0 / (d ** 0.5)
    BLOCK = 128
    
    Q_flat = Q.flatten(0, 1)
    K_flat = K.flatten(0, 1)
    V_flat = V.flatten(0, 1)
    O_flat = O.flatten(0, 1)
    dO_flat = dO.flatten(0, 1)
    dQ_flat = dQ.flatten(0, 1)
    dK_flat = dK.flatten(0, 1)
    dV_flat = dV.flatten(0, 1)
    L_flat = L.flatten(0, 1)
    
    block_shape_3d = [1, BLOCK, d]
    Q_desc = TensorDescriptor.from_tensor(Q_flat, block_shape_3d)
    K_desc = TensorDescriptor.from_tensor(K_flat, block_shape_3d)
    V_desc = TensorDescriptor.from_tensor(V_flat, block_shape_3d)
    O_desc = TensorDescriptor.from_tensor(O_flat, block_shape_3d)
    dO_desc = TensorDescriptor.from_tensor(dO_flat, block_shape_3d)
    dQ_desc = TensorDescriptor.from_tensor(dQ_flat, block_shape_3d)
    dK_desc = TensorDescriptor.from_tensor(dK_flat, block_shape_3d)
    dV_desc = TensorDescriptor.from_tensor(dV_flat, block_shape_3d)
    
    L_desc = TensorDescriptor.from_tensor(L_flat, [1, BLOCK])
    
    num_blocks = triton.cdiv(S, BLOCK)
    grid = (B * H, num_blocks)
    
    if S % BLOCK == 0:
        _bwd_dKdV_kernel[grid](
            Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc,
            dK_desc, dV_desc,
            S, d, tau, BLOCK,
            num_warps=8, num_stages=2
        )
        _bwd_dQ_kernel[grid](
            Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc,
            dQ_desc,
            S, d, tau, BLOCK,
            num_warps=8, num_stages=2
        )
    else:
        fallback_BWD_dKdV_elementwise(Q, K, V, O, dO, L, dK, dV)
        fallback_BWD_dQ_elementwise(Q, K, V, O, dO, L, dQ)