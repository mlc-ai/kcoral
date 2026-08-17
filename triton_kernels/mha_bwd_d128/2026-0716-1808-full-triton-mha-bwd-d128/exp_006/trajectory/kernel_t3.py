import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

TILE = 128

@triton.jit
def load_tile(base_ptr, b, h, s_start, d_start, S, chunk_size: tl.constexpr):
    s_offset = (b * H + h) * S + s_start
    ptr = base_ptr + s_offset * 128 + d_start + tl.arange(0, chunk_size)
    row_idx = tl.arange(0, TILE)[:, None]
    col_idx = tl.arange(0, chunk_size)[None, :]
    return tl.load(ptr + row_idx * 128, mask=((s_start + row_idx) < S), other=0.0)

@triton.jit
def _compute_D_kernel(O_ptr, dO_ptr, D_ptr, B, H, S, BLOCK: tl.constexpr):
    row_idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    elem_idx = tl.arange(0, 128)
    off_O = row_idx[:, None] * 128 + elem_idx[None, :]
    off_dO = row_idx[:, None] * 128 + elem_idx[None, :]
    mask = row_idx[:, None] < B * H * S
    o = tl.load(O_ptr + off_O, mask=mask, other=0.0)
    do = tl.load(dO_ptr + off_dO, mask=mask, other=0.0)
    d = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    if row_idx < B * H * S:
        tl.store(D_ptr + row_idx, d)

@triton.jit
def _bwd_dq_kernel(Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, D_ptr, dQ_ptr, S, scale, B: tl.constexpr, H: tl.constexpr):
    b_h = tl.program_id(0)
    s_blk = tl.program_id(1)
    s_start = s_blk * TILE
    
    b = b_h // H
    h = b_h % H
    
    Q_0 = load_tile(Q_ptr, b, h, s_start, 0, S, 64)
    Q_1 = load_tile(Q_ptr, b, h, s_start, 64, S, 64)
    
    dO_0 = load_tile(dO_ptr, b, h, s_start, 0, S, 64)
    dO_1 = load_tile(dO_ptr, b, h, s_start, 64, S, 64)
    
    O_0 = load_tile(O_ptr, b, h, s_start, 0, S, 64)
    O_1 = load_tile(O_ptr, b, h, s_start, 64, S, 64)
    
    l_s = tl.load(L_ptr + (b_h * S + s_start) + tl.arange(0, TILE), mask=((s_start + tl.arange(0, TILE)) < S), other=0.0)
    d_s = tl.load(D_ptr + (b_h * S + s_start) + tl.arange(0, TILE), mask=((s_start + tl.arange(0, TILE)) < S), other=0.0)
    
    dQ_0 = tl.zeros((TILE, 64), dtype=tl.float32)
    dQ_1 = tl.zeros((TILE, 64), dtype=tl.float32)
    
    num_key_blks = (S + TILE - 1) // TILE
    for j_blk in range(num_key_blks):
        j_start = j_blk * TILE
        
        K_0 = load_tile(K_ptr, b, h, j_start, 0, S, 64)
        K_1 = load_tile(K_ptr, b, h, j_start, 64, S, 64)
        
        V_0 = load_tile(V_ptr, b, h, j_start, 0, S, 64)
        V_1 = load_tile(V_ptr, b, h, j_start, 64, S, 64)
        
        s = tl.dot(Q_0, K_0) + tl.dot(Q_1, K_1)
        dp = tl.dot(dO_0, V_0) + tl.dot(dO_1, V_1)
        
        q_idx = tl.arange(0, TILE)[:, None]
        k_idx = tl.arange(0, TILE)[None, :]
        mask_s = ((s_start + q_idx) < S) & ((j_start + k_idx) < S)
        s = s * mask_s
        dp = dp * mask_s
        
        p = tl.exp(s * scale - l_s[:, None])
        ds = p * (dp - d_s[:, None]) * scale
        ds = ds * mask_s
        
        dQ_0 = tl.dot(ds, K_0.T, dQ_0)
        dQ_1 = tl.dot(ds, K_1.T, dQ_1)
        
    s_offset = (b_h * S + s_start) * 128
    row_idx = tl.arange(0, TILE)[:, None]
    col_idx_0 = tl.arange(0, 64)[None, :]
    col_idx_1 = tl.arange(0, 64)[None, :]
    
    mask_store = ((s_start + tl.arange(0, TILE)) < S)
    
    ptr_0 = dQ_ptr + s_offset + row_idx * 128 + col_idx_0
    tl.store(ptr_0, dQ_0.to(tl.bfloat16), mask=mask_store[:, None])
    
    ptr_1 = dQ_ptr + s_offset + 64 + row_idx * 128 + col_idx_1
    tl.store(ptr_1, dQ_1.to(tl.bfloat16), mask=mask_store[:, None])

@triton.jit
def _bwd_dkv_kernel(Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, D_ptr, dK_ptr, dV_ptr, S, scale, B: tl.constexpr, H: tl.constexpr):
    b_h = tl.program_id(0)
    s_blk = tl.program_id(1)
    s_start = s_blk * TILE
    
    b = b_h // H
    h = b_h % H
    
    K_0 = load_tile(K_ptr, b, h, s_start, 0, S, 64)
    K_1 = load_tile(K_ptr, b, h, s_start, 64, S, 64)
    
    V_0 = load_tile(V_ptr, b, h, s_start, 0, S, 64)
    V_1 = load_tile(V_ptr, b, h, s_start, 64, S, 64)
    
    dK_0 = tl.zeros((TILE, 64), dtype=tl.float32)
    dK_1 = tl.zeros((TILE, 64), dtype=tl.float32)
    dV_0 = tl.zeros((TILE, 64), dtype=tl.float32)
    dV_1 = tl.zeros((TILE, 64), dtype=tl.float32)
    
    num_query_blks = (S + TILE - 1) // TILE
    for i_blk in range(num_query_blks):
        i_start = i_blk * TILE
        
        Q_0 = load_tile(Q_ptr, b, h, i_start, 0, S, 64)
        Q_1 = load_tile(Q_ptr, b, h, i_start, 64, S, 64)
        
        dO_0 = load_tile(dO_ptr, b, h, i_start, 0, S, 64)
        dO_1 = load_tile(dO_ptr, b, h, i_start, 64, S, 64)
        
        O_0 = load_tile(O_ptr, b, h, i_start, 0, S, 64)
        O_1 = load_tile(O_ptr, b, h, i_start, 64, S, 64)
        
        l_i = tl.load(L_ptr + (b_h * S + i_start) + tl.arange(0, TILE), mask=((i_start + tl.arange(0, TILE)) < S), other=0.0)
        d_i = tl.load(D_ptr + (b_h * S + i_start) + tl.arange(0, TILE), mask=((i_start + tl.arange(0, TILE)) < S), other=0.0)
        
        s = tl.dot(Q_0, K_0) + tl.dot(Q_1, K_1)
        dp = tl.dot(dO_0, V_0) + tl.dot(dO_1, V_1)
        
        q_idx = tl.arange(0, TILE)[:, None]
        k_idx = tl.arange(0, TILE)[None, :]
        mask_s = ((i_start + q_idx) < S) & ((s_start + k_idx) < S)
        s = s * mask_s
        dp = dp * mask_s
        
        p = tl.exp(s * scale - l_i[:, None])
        ds = p * (dp - d_i[:, None]) * scale
        ds = ds * mask_s
        
        p_T = p.T * mask_s.T
        ds_T = ds.T * mask_s.T
        
        dK_0 = tl.dot(ds_T, Q_0.T, dK_0)
        dK_1 = tl.dot(ds_T, Q_1.T, dK_1)
        
        dV_0 = tl.dot(p_T, dO_0.T, dV_0)
        dV_1 = tl.dot(p_T, dO_1.T, dV_1)
        
    s_offset = (b_h * S + s_start) * 128
    row_idx = tl.arange(0, TILE)[:, None]
    col_idx_0 = tl.arange(0, 64)[None, :]
    col_idx_1 = tl.arange(0, 64)[None, :]
    
    mask_store = ((s_start + tl.arange(0, TILE)) < S)
    
    ptr_K_0 = dK_ptr + s_offset + row_idx * 128 + col_idx_0
    tl.store(ptr_K_0, dK_0.to(tl.bfloat16), mask=mask_store[:, None])
    
    ptr_K_1 = dK_ptr + s_offset + 64 + row_idx * 128 + col_idx_1
    tl.store(ptr_K_1, dK_1.to(tl.bfloat16), mask=mask_store[:, None])
    
    ptr_V_0 = dV_ptr + s_offset + row_idx * 128 + col_idx_0
    tl.store(ptr_V_0, dV_0.to(tl.bfloat16), mask=mask_store[:, None])
    
    ptr_V_1 = dV_ptr + s_offset + 64 + row_idx * 128 + col_idx_1
    tl.store(ptr_V_1, dV_1.to(tl.bfloat16), mask=mask_store[:, None])

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward pass dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    
    D = torch.empty(B * H * S, device=Q.device, dtype=torch.float32)
    grid_D = (triton.cdiv(B * H * S, 32),)
    _compute_D_kernel[grid_D](O, dO, D, B, H, S, BLOCK=32)
    
    Q_desc = TensorDescriptor.from_tensor(
        Q, stride_row=Q.stride(2), stride_col=Q.stride(3), block_shape=[TILE, 128]
    )
    K_desc = TensorDescriptor.from_tensor(
        K, stride_row=K.stride(2), stride_col=K.stride(3), block_shape=[TILE, 128]
    )
    V_desc = TensorDescriptor.from_tensor(
        V, stride_row=V.stride(2), stride_col=V.stride(3), block_shape=[TILE, 128]
    )
    dO_desc = TensorDescriptor.from_tensor(
        dO, stride_row=dO.stride(2), stride_col=dO.stride(3), block_shape=[TILE, 128]
    )
    O_desc = TensorDescriptor.from_tensor(
        O, stride_row=O.stride(2), stride_col=O.stride(3), block_shape=[TILE, 128]
    )
    dQ_desc = TensorDescriptor.from_tensor(
        dQ, stride_row=dQ.stride(2), stride_col=dQ.stride(3), block_shape=[TILE, 128]
    )
    dK_desc = TensorDescriptor.from_tensor(
        dK, stride_row=dK.stride(2), stride_col=dK.stride(3), block_shape=[TILE, 128]
    )
    dV_desc = TensorDescriptor.from_tensor(
        dV, stride_row=dV.stride(2), stride_col=dV.stride(3), block_shape=[TILE, 128]
    )
    
    num_blocks = (S + TILE - 1) // TILE
    grid = (B * H, num_blocks)
    
    _bwd_dq_kernel[grid](Q, K, V, dO, O, L, D, dQ, S, 1.0 / math.sqrt(128), B=B, H=H, num_warps=4, num_stages=2)
    _bwd_dkv_kernel[grid](Q, K, V, dO, O, L, D, dK, dV, S, 1.0 / math.sqrt(128), B=B, H=H, num_warps=4, num_stages=2)