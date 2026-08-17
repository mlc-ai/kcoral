import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

TILE = 32

@triton.jit
def _compute_D_kernel(O, dO, D, B, H, S, BLOCK: tl.constexpr):
    row_idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    elem_idx = tl.arange(0, 128)
    off_O = row_idx[:, None] * 128 + elem_idx[None, :]
    off_dO = row_idx[:, None] * 128 + elem_idx[None, :]
    mask = row_idx[:, None] < B * H * S
    o = tl.load(O + off_O, mask=mask, other=0.0)
    do = tl.load(dO + off_dO, mask=mask, other=0.0)
    d = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    if row_idx < B * H * S:
        tl.store(D + row_idx, d)

@triton.jit
def _bwd_dq_kernel(Q_desc, K_desc, V_desc, dO_desc, L, D, dQ_desc, S, scale):
    b_h = tl.program_id(0)
    s_blk = tl.program_id(1)
    s_start = s_blk * TILE
    
    Q_0 = Q_desc.load([(b_h * S + s_start), 0])
    Q_1 = Q_desc.load([(b_h * S + s_start), 32])
    Q_2 = Q_desc.load([(b_h * S + s_start), 64])
    Q_3 = Q_desc.load([(b_h * S + s_start), 96])
    
    dO_0 = dO_desc.load([(b_h * S + s_start), 0])
    dO_1 = dO_desc.load([(b_h * S + s_start), 32])
    dO_2 = dO_desc.load([(b_h * S + s_start), 64])
    dO_3 = dO_desc.load([(b_h * S + s_start), 96])
    
    l_s = tl.load(L + (b_h * S + s_start) + tl.arange(0, TILE), mask=((s_start + tl.arange(0, TILE)) < S), other=0.0)
    d_s = tl.load(D + (b_h * S + s_start) + tl.arange(0, TILE), mask=((s_start + tl.arange(0, TILE)) < S), other=0.0)
    
    acc_Q_0 = tl.zeros((TILE, TILE), tl.float32)
    acc_Q_1 = tl.zeros((TILE, TILE), tl.float32)
    acc_Q_2 = tl.zeros((TILE, TILE), tl.float32)
    acc_Q_3 = tl.zeros((TILE, TILE), tl.float32)
    
    num_key_blks = tl.cdiv(S, TILE)
    for j_blk in range(num_key_blks):
        j_start = j_blk * TILE
        
        K_0 = K_desc.load([(b_h * S + j_start), 0])
        K_1 = K_desc.load([(b_h * S + j_start), 32])
        K_2 = K_desc.load([(b_h * S + j_start), 64])
        K_3 = K_desc.load([(b_h * S + j_start), 96])
        
        V_0 = V_desc.load([(b_h * S + j_start), 0])
        V_1 = V_desc.load([(b_h * S + j_start), 32])
        V_2 = V_desc.load([(b_h * S + j_start), 64])
        V_3 = V_desc.load([(b_h * S + j_start), 96])
        
        s = tl.zeros((TILE, TILE), tl.float32)
        s = tl.dot(Q_0, K_0.T, s)
        s = tl.dot(Q_1, K_1.T, s)
        s = tl.dot(Q_2, K_2.T, s)
        s = tl.dot(Q_3, K_3.T, s)
        
        dp = tl.zeros((TILE, TILE), tl.float32)
        dp = tl.dot(dO_0, V_0.T, dp)
        dp = tl.dot(dO_1, V_1.T, dp)
        dp = tl.dot(dO_2, V_2.T, dp)
        dp = tl.dot(dO_3, V_3.T, dp)
        
        mask = ((s_start + tl.arange(0, TILE))[:, None] < S) & ((j_start + tl.arange(0, TILE))[None, :] < S)
        s = s * mask
        dp = dp * mask
        
        p = tl.exp(s * scale - l_s[:, None])
        ds = p * (dp - d_s[:, None]) * scale
        
        acc_Q_0 = tl.dot(ds, K_0, acc_Q_0)
        acc_Q_1 = tl.dot(ds, K_1, acc_Q_1)
        acc_Q_2 = tl.dot(ds, K_2, acc_Q_2)
        acc_Q_3 = tl.dot(ds, K_3, acc_Q_3)
    
    dQ_desc.store([(b_h * S + s_start), 0], acc_Q_0.to(tl.bfloat16))
    dQ_desc.store([(b_h * S + s_start), 32], acc_Q_1.to(tl.bfloat16))
    dQ_desc.store([(b_h * S + s_start), 64], acc_Q_2.to(tl.bfloat16))
    dQ_desc.store([(b_h * S + s_start), 96], acc_Q_3.to(tl.bfloat16))

@triton.jit
def _bwd_dkv_kernel(Q_desc, K_desc, V_desc, dO_desc, L, D, dK_desc, dV_desc, S, scale):
    b_h = tl.program_id(0)
    s_blk = tl.program_id(1)
    s_start = s_blk * TILE
    
    K_0 = K_desc.load([(b_h * S + s_start), 0])
    K_1 = K_desc.load([(b_h * S + s_start), 32])
    K_2 = K_desc.load([(b_h * S + s_start), 64])
    K_3 = K_desc.load([(b_h * S + s_start), 96])
    
    V_0 = V_desc.load([(b_h * S + s_start), 0])
    V_1 = V_desc.load([(b_h * S + s_start), 32])
    V_2 = V_desc.load([(b_h * S + s_start), 64])
    V_3 = V_desc.load([(b_h * S + s_start), 96])
    
    acc_K_0 = tl.zeros((TILE, TILE), tl.float32)
    acc_K_1 = tl.zeros((TILE, TILE), tl.float32)
    acc_K_2 = tl.zeros((TILE, TILE), tl.float32)
    acc_K_3 = tl.zeros((TILE, TILE), tl.float32)
    
    acc_V_0 = tl.zeros((TILE, TILE), tl.float32)
    acc_V_1 = tl.zeros((TILE, TILE), tl.float32)
    acc_V_2 = tl.zeros((TILE, TILE), tl.float32)
    acc_V_3 = tl.zeros((TILE, TILE), tl.float32)
    
    num_query_blks = tl.cdiv(S, TILE)
    for i_blk in range(num_query_blks):
        i_start = i_blk * TILE
        
        Q_0 = Q_desc.load([(b_h * S + i_start), 0])
        Q_1 = Q_desc.load([(b_h * S + i_start), 32])
        Q_2 = Q_desc.load([(b_h * S + i_start), 64])
        Q_3 = Q_desc.load([(b_h * S + i_start), 96])
        
        dO_0 = dO_desc.load([(b_h * S + i_start), 0])
        dO_1 = dO_desc.load([(b_h * S + i_start), 32])
        dO_2 = dO_desc.load([(b_h * S + i_start), 64])
        dO_3 = dO_desc.load([(b_h * S + i_start), 96])
        
        l_i = tl.load(L + (b_h * S + i_start) + tl.arange(0, TILE), mask=((i_start + tl.arange(0, TILE)) < S), other=0.0)
        d_i = tl.load(D + (b_h * S + i_start) + tl.arange(0, TILE), mask=((i_start + tl.arange(0, TILE)) < S), other=0.0)
        
        s = tl.zeros((TILE, TILE), tl.float32)
        s = tl.dot(Q_0, K_0.T, s)
        s = tl.dot(Q_1, K_1.T, s)
        s = tl.dot(Q_2, K_2.T, s)
        s = tl.dot(Q_3, K_3.T, s)
        
        dp = tl.zeros((TILE, TILE), tl.float32)
        dp = tl.dot(dO_0, V_0.T, dp)
        dp = tl.dot(dO_1, V_1.T, dp)
        dp = tl.dot(dO_2, V_2.T, dp)
        dp = tl.dot(dO_3, V_3.T, dp)
        
        mask = ((i_start + tl.arange(0, TILE))[:, None] < S) & ((s_start + tl.arange(0, TILE))[None, :] < S)
        s = s * mask
        dp = dp * mask
        
        p = tl.exp(s * scale - l_i[:, None])
        ds = p * (dp - d_i[:, None]) * scale
        
        acc_K_0 = tl.dot(ds.T, Q_0, acc_K_0)
        acc_K_1 = tl.dot(ds.T, Q_1, acc_K_1)
        acc_K_2 = tl.dot(ds.T, Q_2, acc_K_2)
        acc_K_3 = tl.dot(ds.T, Q_3, acc_K_3)
        
        acc_V_0 = tl.dot(p.T, dO_0, acc_V_0)
        acc_V_1 = tl.dot(p.T, dO_1, acc_V_1)
        acc_V_2 = tl.dot(p.T, dO_2, acc_V_2)
        acc_V_3 = tl.dot(p.T, dO_3, acc_V_3)
    
    dK_desc.store([(b_h * S + s_start), 0], acc_K_0.to(tl.bfloat16))
    dK_desc.store([(b_h * S + s_start), 32], acc_K_1.to(tl.bfloat16))
    dK_desc.store([(b_h * S + s_start), 64], acc_K_2.to(tl.bfloat16))
    dK_desc.store([(b_h * S + s_start), 96], acc_K_3.to(tl.bfloat16))
    
    dV_desc.store([(b_h * S + s_start), 0], acc_V_0.to(tl.bfloat16))
    dV_desc.store([(b_h * S + s_start), 32], acc_V_1.to(tl.bfloat16))
    dV_desc.store([(b_h * S + s_start), 64], acc_V_2.to(tl.bfloat16))
    dV_desc.store([(b_h * S + s_start), 96], acc_V_3.to(tl.bfloat16))

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward pass dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    
    D = torch.empty(B * H * S, device=Q.device, dtype=torch.float32)
    grid_D = (triton.cdiv(B * H * S, 32),)
    _compute_D_kernel[grid_D](O, dO, D, B, H, S, BLOCK=32)
    
    block_shape = [TILE, TILE]
    Q_desc = TensorDescriptor.from_tensor(Q, block_shape)
    K_desc = TensorDescriptor.from_tensor(K, block_shape)
    V_desc = TensorDescriptor.from_tensor(V, block_shape)
    dO_desc = TensorDescriptor.from_tensor(dO, block_shape)
    dQ_desc = TensorDescriptor.from_tensor(dQ, block_shape)
    dK_desc = TensorDescriptor.from_tensor(dK, block_shape)
    dV_desc = TensorDescriptor.from_tensor(dV, block_shape)
    
    grid = (B * H, triton.cdiv(S, TILE))
    
    _bwd_dq_kernel[grid](Q_desc, K_desc, V_desc, dO_desc, L, D, dQ_desc, S, 1.0 / math.sqrt(128))
    _bwd_dkv_kernel[grid](Q_desc, K_desc, V_desc, dO_desc, L, D, dK_desc, dV_desc, S, 1.0 / math.sqrt(128))