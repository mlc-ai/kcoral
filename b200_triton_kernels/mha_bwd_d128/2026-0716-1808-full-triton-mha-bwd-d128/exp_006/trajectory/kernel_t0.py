import torch
import triton
import triton.language as tl
import math

HEAD_DIM = 128
TILE = 16

@triton.jit
def load_q_tile(base_ptr, s_offset, d_offset, S):
    row_idx = tl.arange(0, TILE)
    col_idx = tl.arange(0, 64)
    ptr = base_ptr + (s_offset + row_idx[:, None]) * 128 + (d_offset + col_idx[None, :])
    return tl.load(ptr, mask=((s_offset + row_idx[:, None]) < S), other=0.0)

@triton.jit
def _bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ, S,
):
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    pid = tl.program_id(0)
    num_blocks = (S + TILE - 1) // TILE
    h = (pid // num_blocks) % 48
    s_offset = (pid % num_blocks) * TILE
    b = pid // (48 * num_blocks)
    
    row_idx = tl.arange(0, TILE)
    col_idx = tl.arange(0, 64)
    
    valid_i_flat = s_offset + row_idx < S
    
    q_h = load_q_tile(Q, (b * H + h) * S + s_offset, 0, S)
    q_f = load_q_tile(Q, (b * H + h) * S + s_offset, 64, S)
    do_h = load_q_tile(dO, (b * H + h) * S + s_offset, 0, S)
    do_f = load_q_tile(dO, (b * H + h) * S + s_offset, 64, S)
    o_h = load_q_tile(O, (b * H + h) * S + s_offset, 0, S)
    o_f = load_q_tile(O, (b * H + h) * S + s_offset, 64, S)
    
    l_h = tl.load(L + (b * H + h) * S + s_offset + row_idx, mask=valid_i_flat, other=0.0)
    
    d_val_h = o_h * do_h
    d_val_h = d_val_h.to(tl.float32)
    d_sum_h = tl.sum(d_val_h, axis=1)
    
    d_val_f = o_f * do_f
    d_val_f = d_val_f.to(tl.float32)
    d_sum_f = tl.sum(d_val_f, axis=1)
    
    valid_i = valid_i_flat[:, None]
    q_h *= valid_i
    q_f *= valid_i
    do_h *= valid_i
    do_f *= valid_i
    o_h *= valid_i
    o_f *= valid_i
    
    dQ_acc_h = tl.zeros((TILE, 64), dtype=tl.float32)
    dQ_acc_f = tl.zeros((TILE, 64), dtype=tl.float32)
    
    for j_block in range(num_blocks):
        j_offset = j_block * TILE
        
        k_h = load_q_tile(K, (b * H + h) * S + j_offset, 0, S)
        k_f = load_q_tile(K, (b * H + h) * S + j_offset, 64, S)
        v_h = load_q_tile(V, (b * H + h) * S + j_offset, 0, S)
        v_f = load_q_tile(V, (b * H + h) * S + j_offset, 64, S)
        
        valid_j_flat = j_offset + row_idx < S
        valid_j = valid_j_flat[:, None]
        k_h *= valid_j
        k_f *= valid_j
        v_h *= valid_j
        v_f *= valid_j
        
        k_h_T = k_h.T
        k_f_T = k_f.T
        v_h_T = v_h.T
        v_f_T = v_f.T
        
        s_h = tl.dot(q_h, k_h_T)
        s_f = tl.dot(q_f, k_f_T)
        s = s_h + s_f
        
        dp_h = tl.dot(do_h, v_h_T)
        dp_f = tl.dot(do_f, v_f_T)
        dp = dp_h + dp_f
        
        valid_j_2d = valid_j_flat[None, :]
        s_h *= valid_j_2d
        s_f *= valid_j_2d
        dp_h *= valid_j_2d
        dp_f *= valid_j_2d
        
        p_h = tl.exp(s * scale - l_h[:, None])
        
        ds_h = p_h * (dp - d_sum_h[:, None]) * scale
        
        dQ_acc_h = tl.dot(ds_h, k_h, dQ_acc_h)
        dQ_acc_f = tl.dot(ds_h, k_f, dQ_acc_f)
        
    dQ_ptr = dQ
    base_offset_h = ((b * H + h) * S + s_offset) * 128 + 0
    tl.store(dQ_ptr + base_offset_h + row_idx[:, None] * 128 + col_idx[None, :], dQ_acc_h.to(tl.bfloat16), mask=valid_i_flat[:, None])
    
    base_offset_f = ((b * H + h) * S + s_offset) * 128 + 64
    tl.store(dQ_ptr + base_offset_f + row_idx[:, None] * 128 + col_idx[None, :], dQ_acc_f.to(tl.bfloat16), mask=valid_i_flat[:, None])

@triton.jit
def _bwd_dkv_kernel(
    Q, K, V, O, dO, L, dK, dV, S,
):
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    pid = tl.program_id(0)
    num_blocks = (S + TILE - 1) // TILE
    h = (pid // num_blocks) % 48
    s_offset = (pid % num_blocks) * TILE
    b = pid // (48 * num_blocks)
    
    row_idx = tl.arange(0, TILE)
    col_idx = tl.arange(0, 64)
    
    valid_j_flat = s_offset + row_idx < S
    
    k_h = load_q_tile(K, (b * H + h) * S + s_offset, 0, S)
    k_f = load_q_tile(K, (b * H + h) * S + s_offset, 64, S)
    v_h = load_q_tile(V, (b * H + h) * S + s_offset, 0, S)
    v_f = load_q_tile(V, (b * H + h) * S + s_offset, 64, S)
    
    valid_j = valid_j_flat[:, None]
    k_h *= valid_j
    k_f *= valid_j
    v_h *= valid_j
    v_f *= valid_j
    
    k_h_T = k_h.T
    k_f_T = k_f.T
    v_h_T = v_h.T
    v_f_T = v_f.T
    
    dK_acc_h = tl.zeros((TILE, 64), dtype=tl.float32)
    dK_acc_f = tl.zeros((TILE, 64), dtype=tl.float32)
    dV_acc_h = tl.zeros((TILE, 64), dtype=tl.float32)
    dV_acc_f = tl.zeros((TILE, 64), dtype=tl.float32)
    
    for i_block in range(num_blocks):
        i_offset = i_block * TILE
        
        q_h = load_q_tile(Q, (b * H + h) * S + i_offset, 0, S)
        q_f = load_q_tile(Q, (b * H + h) * S + i_offset, 64, S)
        do_h = load_q_tile(dO, (b * H + h) * S + i_offset, 0, S)
        do_f = load_q_tile(dO, (b * H + h) * S + i_offset, 64, S)
        o_h = load_q_tile(O, (b * H + h) * S + i_offset, 0, S)
        o_f = load_q_tile(O, (b * H + h) * S + i_offset, 64, S)
        
        l_i = tl.load(L + (b * H + h) * S + i_offset + row_idx, mask=(i_offset + row_idx < S), other=0.0)
        
        valid_i_flat = i_offset + row_idx < S
        valid_i = valid_i_flat[:, None]
        q_h *= valid_i
        q_f *= valid_i
        do_h *= valid_i
        do_f *= valid_i
        o_h *= valid_i
        o_f *= valid_i
        
        d_val_h = o_h * do_h
        d_val_h = d_val_h.to(tl.float32)
        d_sum_h = tl.sum(d_val_h, axis=1)
        
        d_val_f = o_f * do_f
        d_val_f = d_val_f.to(tl.float32)
        d_sum_f = tl.sum(d_val_f, axis=1)
        
        s_h = tl.dot(q_h, k_h_T)
        s_f = tl.dot(q_f, k_f_T)
        s = s_h + s_f
        
        dp_h = tl.dot(do_h, v_h_T)
        dp_f = tl.dot(do_f, v_f_T)
        dp = dp_h + dp_f
        
        valid_i_2d = valid_i_flat[None, :]
        s_h *= valid_i_2d
        s_f *= valid_i_2d
        dp_h *= valid_i_2d
        dp_f *= valid_i_2d
        
        p_h = tl.exp(s * scale - l_i[:, None])
        
        ds_h = p_h * (dp - d_sum_h[:, None]) * scale
        
        ds_h_T = ds_h.T
        p_h_T = p_h.T
        
        dK_acc_h = tl.dot(ds_h_T, q_h, dK_acc_h)
        dK_acc_f = tl.dot(ds_h_T, q_f, dK_acc_f)
        
        dV_acc_h = tl.dot(p_h_T, do_h, dV_acc_h)
        dV_acc_f = tl.dot(p_h_T, do_f, dV_acc_f)
        
    dK_ptr = dK
    base_offset_h = ((b * H + h) * S + s_offset) * 128 + 0
    tl.store(dK_ptr + base_offset_h + row_idx[:, None] * 128 + col_idx[None, :], dK_acc_h.to(tl.bfloat16), mask=valid_j_flat[:, None])
    
    base_offset_f = ((b * H + h) * S + s_offset) * 128 + 64
    tl.store(dK_ptr + base_offset_f + row_idx[:, None] * 128 + col_idx[None, :], dK_acc_f.to(tl.bfloat16), mask=valid_j_flat[:, None])
    
    dV_ptr = dV
    tl.store(dV_ptr + base_offset_h + row_idx[:, None] * 128 + col_idx[None, :], dV_acc_h.to(tl.bfloat16), mask=valid_j_flat[:, None])
    
    tl.store(dV_ptr + base_offset_f + row_idx[:, None] * 128 + col_idx[None, :], dV_acc_f.to(tl.bfloat16), mask=valid_j_flat[:, None])

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward pass dQ, dK, dV into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    
    num_blocks = (S + TILE - 1) // TILE
    grid = (B * H * num_blocks,)
    
    _bwd_dq_kernel[grid](Q, K, V, O, dO, L, dQ, S, num_warps=4)
    _bwd_dkv_kernel[grid](Q, K, V, O, dO, L, dK, dV, S, num_warps=4)