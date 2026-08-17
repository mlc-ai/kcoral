import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

TILE = 64

@triton.jit
def _bwd_dq_kernel(Q_desc, K_desc, V_desc, dO_desc, O_desc, L, dQ_desc, S, scale, B: tl.constexpr, H: tl.constexpr):
    b_h = tl.program_id(0)
    s_blk = tl.program_id(1)
    s_start = s_blk * TILE
    
    b = b_h // H
    h = b_h % H
    
    Q_0 = Q_desc.load([b, h, s_start, 0])
    Q_1 = Q_desc.load([b, h, s_start, 64])
    
    dO_0 = dO_desc.load([b, h, s_start, 0])
    dO_1 = dO_desc.load([b, h, s_start, 64])
    
    O_0 = O_desc.load([b, h, s_start, 0])
    O_1 = O_desc.load([b, h, s_start, 64])
    
    l_s = tl.load(L + (b_h * S + s_start) + tl.arange(0, TILE), mask=((s_start + tl.arange(0, TILE)) < S), other=0.0)
    
    d_val_0 = O_0 * dO_0
    d_val_1 = O_1 * dO_1
    d_sum = tl.sum(d_val_0.to(tl.float32) + d_val_1.to(tl.float32), axis=1)
    
    acc_Q_0 = tl.zeros((TILE, TILE), tl.float32)
    acc_Q_1 = tl.zeros((TILE, TILE), tl.float32)
    
    num_key_blks = tl.cdiv(S, TILE)
    for j_blk in range(num_key_blks):
        j_start = j_blk * TILE
        
        K_0 = K_desc.load([b, h, j_start, 0])
        K_1 = K_desc.load([b, h, j_start, 64])
        
        V_0 = V_desc.load([b, h, j_start, 0])
        V_1 = V_desc.load([b, h, j_start, 64])
        
        s = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        
        dp = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        mask = ((s_start + tl.arange(0, TILE))[:, None] < S) & ((j_start + tl.arange(0, TILE))[None, :] < S)
        s = s * mask
        
        p = tl.exp(s * scale - l_s[:, None])
        ds = p * (dp - d_sum[:, None]) * scale
        ds = ds * mask
        
        acc_Q_0 = tl.dot(ds, K_0, acc_Q_0)
        acc_Q_1 = tl.dot(ds, K_1, acc_Q_1)
    
    dQ_desc.store([b, h, s_start, 0], acc_Q_0.to(tl.bfloat16))
    dQ_desc.store([b, h, s_start, 64], acc_Q_1.to(tl.bfloat16))

@triton.jit
def _bwd_dkv_kernel(Q_desc, K_desc, V_desc, dO_desc, O_desc, L, dK_desc, dV_desc, S, scale, B: tl.constexpr, H: tl.constexpr):
    b_h = tl.program_id(0)
    s_blk = tl.program_id(1)
    s_start = s_blk * TILE
    
    b = b_h // H
    h = b_h % H
    
    K_0 = K_desc.load([b, h, s_start, 0])
    K_1 = K_desc.load([b, h, s_start, 64])
    
    V_0 = V_desc.load([b, h, s_start, 0])
    V_1 = V_desc.load([b, h, s_start, 64])
    
    acc_K_0 = tl.zeros((TILE, TILE), tl.float32)
    acc_K_1 = tl.zeros((TILE, TILE), tl.float32)
    
    acc_V_0 = tl.zeros((TILE, TILE), tl.float32)
    acc_V_1 = tl.zeros((TILE, TILE), tl.float32)
    
    num_query_blks = tl.cdiv(S, TILE)
    for i_blk in range(num_query_blks):
        i_start = i_blk * TILE
        
        Q_0 = Q_desc.load([b, h, i_start, 0])
        Q_1 = Q_desc.load([b, h, i_start, 64])
        
        dO_0 = dO_desc.load([b, h, i_start, 0])
        dO_1 = dO_desc.load([b, h, i_start, 64])
        
        O_0 = O_desc.load([b, h, i_start, 0])
        O_1 = O_desc.load([b, h, i_start, 64])
        
        l_i = tl.load(L + (b_h * S + i_start) + tl.arange(0, TILE), mask=((i_start + tl.arange(0, TILE)) < S), other=0.0)
        
        d_val_0 = O_0 * dO_0
        d_val_1 = O_1 * dO_1
        d_sum = tl.sum(d_val_0.to(tl.float32) + d_val_1.to(tl.float32), axis=1)
        
        s = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        
        dp = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        mask = ((i_start + tl.arange(0, TILE))[:, None] < S) & ((s_start + tl.arange(0, TILE))[None, :] < S)
        s = s * mask
        
        p = tl.exp(s * scale - l_i[:, None])
        ds = p * (dp - d_sum[:, None]) * scale
        ds = ds * mask
        
        p_T = p.T * mask.T
        ds_T = ds.T * mask.T
        
        acc_K_0 = tl.dot(ds_T, Q_0, acc_K_0)
        acc_K_1 = tl.dot(ds_T, Q_1, acc_K_1)
        
        acc_V_0 = tl.dot(p_T, dO_0, acc_V_0)
        acc_V_1 = tl.dot(p_T, dO_1, acc_V_1)
    
    dK_desc.store([b, h, s_start, 0], acc_K_0.to(tl.bfloat16))
    dK_desc.store([b, h, s_start, 64], acc_K_1.to(tl.bfloat16))
    
    dV_desc.store([b, h, s_start, 0], acc_V_0.to(tl.bfloat16))
    dV_desc.store([b, h, s_start, 64], acc_V_1.to(tl.bfloat16))

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward pass dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(
        Q, shape=[B, H, S, d], strides=[H*S*d, S*d, d, 1]
    )
    K_desc = TensorDescriptor.from_tensor(
        K, shape=[B, H, S, d], strides=[H*S*d, S*d, d, 1]
    )
    V_desc = TensorDescriptor.from_tensor(
        V, shape=[B, H, S, d], strides=[H*S*d, S*d, d, 1]
    )
    dO_desc = TensorDescriptor.from_tensor(
        dO, shape=[B, H, S, d], strides=[H*S*d, S*d, d, 1]
    )
    O_desc = TensorDescriptor.from_tensor(
        O, shape=[B, H, S, d], strides=[H*S*d, S*d, d, 1]
    )
    dQ_desc = TensorDescriptor.from_tensor(
        dQ, shape=[B, H, S, d], strides=[H*S*d, S*d, d, 1]
    )
    dK_desc = TensorDescriptor.from_tensor(
        dK, shape=[B, H, S, d], strides=[H*S*d, S*d, d, 1]
    )
    dV_desc = TensorDescriptor.from_tensor(
        dV, shape=[B, H, S, d], strides=[H*S*d, S*d, d, 1]
    )
    
    grid = (B * H, triton.cdiv(S, TILE))
    
    _bwd_dq_kernel[grid](Q_desc, K_desc, V_desc, dO_desc, O_desc, L, dQ_desc, S, 1.0 / math.sqrt(128), B=B, H=H, num_warps=4, num_stages=2)
    _bwd_dkv_kernel[grid](Q_desc, K_desc, V_desc, dO_desc, O_desc, L, dK_desc, dV_desc, S, 1.0 / math.sqrt(128), B=B, H=H, num_warps=4, num_stages=2)