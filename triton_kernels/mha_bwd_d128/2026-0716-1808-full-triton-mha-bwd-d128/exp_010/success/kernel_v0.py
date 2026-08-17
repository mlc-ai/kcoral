import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def dKdV_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr,
    desc_dK, desc_dV,
    S, scale, BLOCK_M: tl.constexpr,
):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    off_j = j * BLOCK_M
    
    k0 = desc_K.load([bh * S + off_j, 0])
    k1 = desc_K.load([bh * S + off_j, 64])
    
    v0 = desc_V.load([bh * S + off_j, 0])
    v1 = desc_V.load([bh * S + off_j, 64])
    
    acc_dK0 = tl.zeros((BLOCK_M, 64), tl.float32)
    acc_dK1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    acc_dV0 = tl.zeros((BLOCK_M, 64), tl.float32)
    acc_dV1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    num_blocks = S // BLOCK_M
    
    for i in range(num_blocks):
        off_i = i * BLOCK_M
        
        q0 = desc_Q.load([bh * S + off_i, 0])
        q1 = desc_Q.load([bh * S + off_i, 64])
        
        do0 = desc_dO.load([bh * S + off_i, 0])
        do1 = desc_dO.load([bh * S + off_i, 64])
        
        o0 = desc_O.load([bh * S + off_i, 0])
        o1 = desc_O.load([bh * S + off_i, 64])
        
        d_i = tl.sum(do0 * o0, axis=1) + tl.sum(do1 * o1, axis=1)
        
        l_i = tl.load(L_ptr + bh * S + off_i + tl.arange(0, BLOCK_M), 
                      mask=(off_i + tl.arange(0, BLOCK_M)) < S, other=-float('inf'))
        
        acc_s = tl.dot(q0, k0.T)
        acc_s = tl.dot(q1, k1.T, acc_s)
        
        acc_dp = tl.dot(do0, v0.T)
        acc_dp = tl.dot(do1, v1.T, acc_dp)
        
        s_tmp = acc_s * scale
        
        p_tmp = tl.exp(s_tmp - l_i[:, None])
        
        row_offs_i = off_i + tl.arange(0, BLOCK_M)
        col_offs_j = off_j + tl.arange(0, BLOCK_M)
        pos_mask = (row_offs_i[:, None] < S) & (col_offs_j[None, :] < S)
        
        p_tmp = tl.where(pos_mask, p_tmp, 0.0)
        
        d_i_expanded = d_i[:, None]
        
        ds_tmp = p_tmp * (acc_dp - d_i_expanded) * scale
        
        ds_tmp = tl.where(pos_mask, ds_tmp, 0.0)
        
        p_tmp = p_tmp.to(tl.bfloat16)
        ds_tmp = ds_tmp.to(tl.bfloat16)
        
        acc_dV0 = tl.dot(p_tmp.T, do0, acc_dV0)
        acc_dV1 = tl.dot(p_tmp.T, do1, acc_dV1)
        
        acc_dK0 = tl.dot(ds_tmp.T, q0, acc_dK0)
        acc_dK1 = tl.dot(ds_tmp.T, q1, acc_dK1)
        
    desc_dK.store([bh * S + off_j, 0], acc_dK0.to(tl.bfloat16))
    desc_dK.store([bh * S + off_j, 64], acc_dK1.to(tl.bfloat16))
    
    desc_dV.store([bh * S + off_j, 0], acc_dV0.to(tl.bfloat16))
    desc_dV.store([bh * S + off_j, 64], acc_dV1.to(tl.bfloat16))


@triton.jit
def dQ_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr, desc_dQ,
    S, scale, BLOCK_M: tl.constexpr,
):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    off_i = i * BLOCK_M
    
    q0 = desc_Q.load([bh * S + off_i, 0])
    q1 = desc_Q.load([bh * S + off_i, 64])
    
    do0 = desc_dO.load([bh * S + off_i, 0])
    do1 = desc_dO.load([bh * S + off_i, 64])
    
    o0 = desc_O.load([bh * S + off_i, 0])
    o1 = desc_O.load([bh * S + off_i, 64])
    
    d_i = tl.sum(do0 * o0, axis=1) + tl.sum(do1 * o1, axis=1)
    
    l_i = tl.load(L_ptr + bh * S + off_i + tl.arange(0, BLOCK_M), mask=(off_i + tl.arange(0, BLOCK_M)) < S, other=-float('inf'))
    
    acc_dq0 = tl.zeros((BLOCK_M, 64), tl.float32)
    acc_dq1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    num_blocks = S // BLOCK_M
    
    for j in range(num_blocks):
        off_j = j * BLOCK_M
        
        k0 = desc_K.load([bh * S + off_j, 0])
        k1 = desc_K.load([bh * S + off_j, 64])
        
        v0 = desc_V.load([bh * S + off_j, 0])
        v1 = desc_V.load([bh * S + off_j, 64])
        
        acc_s = tl.dot(q0, k0.T)
        acc_s = tl.dot(q1, k1.T, acc_s)
        
        acc_dp = tl.dot(do0, v0.T)
        acc_dp = tl.dot(do1, v1.T, acc_dp)
        
        s_tmp = acc_s * scale
        
        p_tmp = tl.exp(s_tmp - l_i[:, None])
        
        row_offs_i = off_i + tl.arange(0, BLOCK_M)
        col_offs_j = off_j + tl.arange(0, BLOCK_M)
        pos_mask = (row_offs_i[:, None] < S) & (col_offs_j[None, :] < S)
        
        p_tmp = tl.where(pos_mask, p_tmp, 0.0)
        
        d_i_expanded = d_i[:, None]
        
        ds_tmp = p_tmp * (acc_dp - d_i_expanded) * scale
        
        ds_tmp = tl.where(pos_mask, ds_tmp, 0.0)
        
        ds_tmp = ds_tmp.to(tl.bfloat16)
        
        acc_dq0 = tl.dot(ds_tmp, k0, acc_dq0)
        acc_dq1 = tl.dot(ds_tmp, k1, acc_dq1)
        
    desc_dQ.store([bh * S + off_i, 0], acc_dq0.to(tl.bfloat16))
    desc_dQ.store([bh * S + off_i, 64], acc_dq1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Computes the backward pass of multi-head attention natively on CUDA."""
    device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    
    B, H, S, d = Q.shape
    
    Q = Q.view(-1, Q.shape[-1])
    K = K.view(-1, K.shape[-1])
    V = V.view(-1, V.shape[-1])
    O = O.view(-1, O.shape[-1])
    dO = dO.view(-1, dO.shape[-1])
    dQ = dQ.view(-1, dQ.shape[-1])
    dK = dK.view(-1, dK.shape[-1])
    dV = dV.view(-1, dV.shape[-1])
    
    BLOCK_M = 64
    
    desc_Q = TensorDescriptor.from_tensor(Q, [BLOCK_M, 64])
    desc_K = TensorDescriptor.from_tensor(K, [BLOCK_M, 64])
    desc_V = TensorDescriptor.from_tensor(V, [BLOCK_M, 64])
    desc_O = TensorDescriptor.from_tensor(O, [BLOCK_M, 64])
    desc_dO = TensorDescriptor.from_tensor(dO, [BLOCK_M, 64])
    
    desc_dQ = TensorDescriptor.from_tensor(dQ, [BLOCK_M, 64])
    desc_dK = TensorDescriptor.from_tensor(dK, [BLOCK_M, 64])
    desc_dV = TensorDescriptor.from_tensor(dV, [BLOCK_M, 64])
    
    scale = 1.0 / math.sqrt(d)
    
    grid = (S // BLOCK_M, B * H)
    
    dQ_kernel[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, L, desc_dQ,
        S, scale, BLOCK_M, num_warps=8, num_stages=3
    )
    dKdV_kernel[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, L,
        desc_dK, desc_dV,
        S, scale, BLOCK_M, num_warps=8, num_stages=3
    )