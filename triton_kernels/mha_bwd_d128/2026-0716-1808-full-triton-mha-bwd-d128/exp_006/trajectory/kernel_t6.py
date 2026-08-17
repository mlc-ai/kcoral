import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

@triton.jit
def _bwd_dq_kernel(Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr, dQ_ptr, S, scale):
    b_h = tl.program_id(0)
    s_blk = tl.program_id(1)
    s_start = s_blk * 64
    
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
    
    l_s = tl.load(L_ptr + (b_h * S + s_start) + tl.arange(0, 64), mask=((s_start + tl.arange(0, 64)) < S), other=0.0)
    
    dQ_0 = tl.zeros((64, 64), dtype=tl.float32)
    dQ_1 = tl.zeros((64, 64), dtype=tl.float32)
    
    num_key_blks = (S + 63) // 64
    for j_blk in range(num_key_blks):
        j_start = j_blk * 64
        k_row_idx = b_h * S + j_start
        
        K_0 = K_desc.load([k_row_idx, 0])
        K_1 = K_desc.load([k_row_idx, 64])
        
        V_0 = V_desc.load([k_row_idx, 0])
        V_1 = V_desc.load([k_row_idx, 64])
        
        s = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        dp = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        q_idx = tl.arange(0, 64)[:, None]
        k_idx = tl.arange(0, 64)[None, :]
        mask_s = ((s_start + q_idx) < S) & ((j_start + k_idx) < S)
        s = s * mask_s
        dp = dp * mask_s
        
        p = tl.exp(s * scale - l_s[:, None])
        ds = p * (dp - d_sum[:, None]) * scale
        ds = ds * mask_s
        
        dQ_0 = tl.dot(ds, K_0, dQ_0)
        dQ_1 = tl.dot(ds, K_1, dQ_1)
        
    base_ptr = dQ_ptr + (b_h * S + s_start) * 128
    row_idx_2d = tl.arange(0, 64)[:, None]
    col_idx_0 = tl.arange(0, 64)[None, :]
    col_idx_1 = tl.arange(0, 64)[None, :] + 64
    
    mask_store = ((s_start + tl.arange(0, 64)) < S)
    
    ptr_0 = base_ptr + row_idx_2d * 128 + col_idx_0
    tl.store(ptr_0, dQ_0.to(tl.bfloat16), mask=mask_store[:, None])
    
    ptr_1 = base_ptr + row_idx_2d * 128 + col_idx_1
    tl.store(ptr_1, dQ_1.to(tl.bfloat16), mask=mask_store[:, None])

@triton.jit
def _bwd_dkv_kernel(Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr, dK_ptr, dV_ptr, S, scale):
    b_h = tl.program_id(0)
    s_blk = tl.program_id(1)
    s_start = s_blk * 64
    
    row_idx = b_h * S + s_start
    K_0 = K_desc.load([row_idx, 0])
    K_1 = K_desc.load([row_idx, 64])
    
    V_0 = V_desc.load([row_idx, 0])
    V_1 = V_desc.load([row_idx, 64])
    
    dK_0 = tl.zeros((64, 64), dtype=tl.float32)
    dK_1 = tl.zeros((64, 64), dtype=tl.float32)
    dV_0 = tl.zeros((64, 64), dtype=tl.float32)
    dV_1 = tl.zeros((64, 64), dtype=tl.float32)
    
    num_query_blks = (S + 63) // 64
    for i_blk in range(num_query_blks):
        i_start = i_blk * 64
        q_row_idx = b_h * S + i_start
        
        Q_0 = Q_desc.load([q_row_idx, 0])
        Q_1 = Q_desc.load([q_row_idx, 64])
        
        dO_0 = dO_desc.load([q_row_idx, 0])
        dO_1 = dO_desc.load([q_row_idx, 64])
        
        O_0 = O_desc.load([q_row_idx, 0])
        O_1 = O_desc.load([q_row_idx, 64])
        
        d_val_0 = O_0 * dO_0
        d_val_1 = O_1 * dO_1
        d_i = tl.sum(d_val_0.to(tl.float32) + d_val_1.to(tl.float32), axis=1)
        
        l_i = tl.load(L_ptr + (b_h * S + i_start) + tl.arange(0, 64), mask=((i_start + tl.arange(0, 64)) < S), other=0.0)
        
        s = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        dp = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        q_idx = tl.arange(0, 64)[:, None]
        k_idx = tl.arange(0, 64)[None, :]
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
    
    base_ptr_K = dK_ptr + (b_h * S + s_start) * 128
    base_ptr_V = dV_ptr + (b_h * S + s_start) * 128
    
    row_idx_2d = tl.arange(0, 64)[:, None]
    col_idx_0 = tl.arange(0, 64)[None, :]
    col_idx_1 = tl.arange(0, 64)[None, :] + 64
    
    mask_store = ((s_start + tl.arange(0, 64)) < S)
    
    ptr_K_0 = base_ptr_K + row_idx_2d * 128 + col_idx_0
    tl.store(ptr_K_0, dK_0.to(tl.bfloat16), mask=mask_store[:, None])
    
    ptr_K_1 = base_ptr_K + row_idx_2d * 128 + col_idx_1
    tl.store(ptr_K_1, dK_1.to(tl.bfloat16), mask=mask_store[:, None])
    
    ptr_V_0 = base_ptr_V + row_idx_2d * 128 + col_idx_0
    tl.store(ptr_V_0, dV_0.to(tl.bfloat16), mask=mask_store[:, None])
    
    ptr_V_1 = base_ptr_V + row_idx_2d * 128 + col_idx_1
    tl.store(ptr_V_1, dV_1.to(tl.bfloat16), mask=mask_store[:, None])

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward pass dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    
    Q_f = Q.reshape(B * H * S, d)
    K_f = K.reshape(B * H * S, d)
    V_f = V.reshape(B * H * S, d)
    dO_f = dO.reshape(B * H * S, d)
    O_f = O.reshape(B * H * S, d)
    dQ_f = dQ.reshape(B * H * S, d)
    dK_f = dK.reshape(B * H * S, d)
    dV_f = dV.reshape(B * H * S, d)
    
    Q_desc = TensorDescriptor.from_tensor(Q_f, [64, 64])
    K_desc = TensorDescriptor.from_tensor(K_f, [64, 64])
    V_desc = TensorDescriptor.from_tensor(V_f, [64, 64])
    dO_desc = TensorDescriptor.from_tensor(dO_f, [64, 64])
    O_desc = TensorDescriptor.from_tensor(O_f, [64, 64])
    
    num_blocks = (S + 63) // 64
    grid = (B * H, num_blocks)
    
    _bwd_dq_kernel[grid](Q_desc, K_desc, V_desc, dO_desc, O_desc, L, dQ_f, S, 1.0 / math.sqrt(128), num_warps=4, num_stages=2)
    _bwd_dkv_kernel[grid](Q_desc, K_desc, V_desc, dO_desc, O_desc, L, dK_f, dV_f, S, 1.0 / math.sqrt(128), num_warps=4, num_stages=2)