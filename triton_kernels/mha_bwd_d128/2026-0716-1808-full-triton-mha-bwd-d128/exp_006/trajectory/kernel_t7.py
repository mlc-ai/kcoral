import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

TILE = 128

@triton.jit
def _compute_D_kernel(O_ptr, dO_ptr, D_ptr, B, H, S, BLOCK: tl.constexpr):
    row_idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    elem_idx = tl.arange(0, 64)
    
    off_O_0 = row_idx[:, None] * 128 + elem_idx[None, :]
    off_dO_0 = row_idx[:, None] * 128 + elem_idx[None, :]
    
    off_O_1 = row_idx[:, None] * 128 + 64 + elem_idx[None, :]
    off_dO_1 = row_idx[:, None] * 128 + 64 + elem_idx[None, :]
    
    mask = row_idx[:, None] < B * H * S
    
    o_0 = tl.load(O_ptr + off_O_0, mask=mask, other=0.0)
    o_1 = tl.load(O_ptr + off_O_1, mask=mask, other=0.0)
    
    do_0 = tl.load(dO_ptr + off_dO_0, mask=mask, other=0.0)
    do_1 = tl.load(dO_ptr + off_dO_1, mask=mask, other=0.0)
    
    d_val_0 = o_0 * do_0
    d_val_1 = o_1 * do_1
    
    d = tl.sum(d_val_0.to(tl.float32) + d_val_1.to(tl.float32), axis=1)
    
    valid = row_idx < B * H * S
    tl.store(D_ptr + row_idx, d, mask=valid)

@triton.jit
def _bwd_dq_kernel(Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr, D_ptr, dQ_ptr, S, scale):
    b_h = tl.program_id(0)
    s_blk = tl.program_id(1)
    s_start = s_blk * TILE
    
    row_idx = b_h * S + s_start
    Q_0 = Q_desc.load([row_idx, 0])
    Q_1 = Q_desc.load([row_idx, 64])
    
    dO_0 = dO_desc.load([row_idx, 0])
    dO_1 = dO_desc.load([row_idx, 64])
    
    O_0 = O_desc.load([row_idx, 0])
    O_1 = O_desc.load([row_idx, 64])
    
    d_val_0 = O_0 * dO_0
    d_val_1 = O_1 * dO_1
    d_sum = tl.sum(d_val_0.to(tl.float32) + d_val_1.to(tl.float32), axis=1)
    
    l_s = tl.load(L_ptr + (b_h * S + s_start) + tl.arange(0, TILE), mask=((s_start + tl.arange(0, TILE)) < S), other=0.0)
    
    dQ_0 = tl.zeros((TILE, 64), dtype=tl.float32)
    dQ_1 = tl.zeros((TILE, 64), dtype=tl.float32)
    
    num_key_blks = (S + TILE - 1) // TILE
    for j_blk in range(num_key_blks):
        j_start = j_blk * TILE
        k_row_idx = b_h * S + j_start
        
        K_0 = K_desc.load([k_row_idx, 0]).to(tl.float32)
        K_1 = K_desc.load([k_row_idx, 64]).to(tl.float32)
        
        V_0 = V_desc.load([k_row_idx, 0]).to(tl.float32)
        V_1 = V_desc.load([k_row_idx, 64]).to(tl.float32)
        
        s = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        dp = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        q_idx = tl.arange(0, TILE)[:, None]
        k_idx = tl.arange(0, TILE)[None, :]
        mask_s = ((s_start + q_idx) < S) & ((j_start + k_idx) < S)
        s = s * mask_s
        dp = dp * mask_s
        
        p = tl.exp(s * scale - l_s[:, None])
        ds = p * (dp - d_sum[:, None]) * scale
        ds = ds * mask_s
        
        dQ_0 = tl.dot(ds, K_0, dQ_0)
        dQ_1 = tl.dot(ds, K_1, dQ_1)
        
    dQ_desc = TensorDescriptor.from_tensor(dQ, [TILE, 64])
    dQ_desc.store([row_idx, 0], dQ_0.to(tl.bfloat16))
    dQ_desc.store([row_idx, 64], dQ_1.to(tl.bfloat16))

@triton.jit
def _bwd_dkv_kernel(Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr, D_ptr, dK_ptr, dV_ptr, S, scale):
    b_h = tl.program_id(0)
    s_blk = tl.program_id(1)
    s_start = s_blk * TILE
    
    row_idx = b_h * S + s_start
    K_0 = K_desc.load([row_idx, 0]).to(tl.float32)
    K_1 = K_desc.load([row_idx, 64]).to(tl.float32)
    
    V_0 = V_desc.load([row_idx, 0]).to(tl.float32)
    V_1 = V_desc.load([row_idx, 64]).to(tl.float32)
    
    dK_0 = tl.zeros((TILE, 64), dtype=tl.float32)
    dK_1 = tl.zeros((TILE, 64), dtype=tl.float32)
    dV_0 = tl.zeros((TILE, 64), dtype=tl.float32)
    dV_1 = tl.zeros((TILE, 64), dtype=tl.float32)
    
    num_query_blks = (S + TILE - 1) // TILE
    for i_blk in range(num_query_blks):
        i_start = i_blk * TILE
        q_row_idx = b_h * S + i_start
        
        Q_0 = Q_desc.load([q_row_idx, 0]).to(tl.float32)
        Q_1 = Q_desc.load([q_row_idx, 64]).to(tl.float32)
        
        dO_0 = dO_desc.load([q_row_idx, 0]).to(tl.float32)
        dO_1 = dO_desc.load([q_row_idx, 64]).to(tl.float32)
        
        O_0 = O_desc.load([q_row_idx, 0]).to(tl.float32)
        O_1 = O_desc.load([q_row_idx, 64]).to(tl.float32)
        
        d_val_0 = O_0 * dO_0
        d_val_1 = O_1 * dO_1
        d_i = tl.sum(d_val_0 + d_val_1, axis=1)
        
        l_i = tl.load(L_ptr + (b_h * S + i_start) + tl.arange(0, TILE), mask=((i_start + tl.arange(0, TILE)) < S), other=0.0)
        
        s = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        dp = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        q_idx = tl.arange(0, TILE)[:, None]
        k_idx = tl.arange(0, TILE)[None, :]
        mask_s = ((i_start + q_idx) < S) & ((s_start + k_idx) < S)
        s = s * mask_s
        dp = dp * mask_s
        
        p = tl.exp(s * scale - l_i[:, None])
        ds = p * (dp - d_i[:, None]) * scale
        
        p = p * mask_s
        ds = ds * mask_s
        
        p_T = p.T
        ds_T = ds.T
        
        dK_0 = tl.dot(ds_T, Q_0, dK_0)
        dK_1 = tl.dot(ds_T, Q_1, dK_1)
        
        dV_0 = tl.dot(p_T, dO_0, dV_0)
        dV_1 = tl.dot(p_T, dO_1, dV_1)
    
    dK_desc = TensorDescriptor.from_tensor(dK, [TILE, 64])
    dV_desc = TensorDescriptor.from_tensor(dV, [TILE, 64])
    
    dK_desc.store([row_idx, 0], dK_0.to(tl.bfloat16))
    dK_desc.store([row_idx, 64], dK_1.to(tl.bfloat16))
    
    dV_desc.store([row_idx, 0], dV_0.to(tl.bfloat16))
    dV_desc.store([row_idx, 64], dV_1.to(tl.bfloat16))

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward pass dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    
    D = torch.empty(B * H * S, device=Q.device, dtype=torch.float32)
    grid_D = (triton.cdiv(B * H * S, 32),)
    _compute_D_kernel[grid_D](O, dO, D, B, H, S, BLOCK=32)
    
    Q_f = Q.reshape(B * H * S, d)
    K_f = K.reshape(B * H * S, d)
    V_f = V.reshape(B * H * S, d)
    dO_f = dO.reshape(B * H * S, d)
    O_f = O.reshape(B * H * S, d)
    
    Q_desc = TensorDescriptor.from_tensor(Q_f, [TILE, 64])
    K_desc = TensorDescriptor.from_tensor(K_f, [TILE, 64])
    V_desc = TensorDescriptor.from_tensor(V_f, [TILE, 64])
    dO_desc = TensorDescriptor.from_tensor(dO_f, [TILE, 64])
    O_desc = TensorDescriptor.from_tensor(O_f, [TILE, 64])
    
    dQ_ptr = dQ.data_ptr()
    dK_ptr = dK.data_ptr()
    dV_ptr = dV.data_ptr()
    
    num_blocks = (S + TILE - 1) // TILE
    grid = (B * H, num_blocks)
    
    _bwd_dq_kernel[grid](Q_desc, K_desc, V_desc, dO_desc, O_desc, L, D, dQ_ptr, S, 1.0 / math.sqrt(128), BLOCK_M=128, BLOCK_N=64)
    _bwd_dkv_kernel[grid](Q_desc, K_desc, V_desc, dO_desc, O_desc, L, D, dK_ptr, dV_ptr, S, 1.0 / math.sqrt(128), BLOCK_M=128, BLOCK_N=64)