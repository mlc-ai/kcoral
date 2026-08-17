import torch
import triton
import triton.language as tl


@triton.jit
def load_tile_64x128(base_ptr, bh, block_row_start, S):
    """Load a [64, 128] bf16 tile."""
    row_offs = block_row_start + tl.arange(0, 64)
    col_offs = tl.arange(0, 128)
    ptrs = base_ptr + (bh * S + row_offs[:, None]) * 128 + col_offs[None, :]
    return tl.load(ptrs, mask=(row_offs[:, None] < S), other=0.0)


@triton.jit
def store_tile_64x128(base_ptr, bh, row_start, value, S, col_start=0):
    """Store a [64, 64] bf16 tile."""
    row_offs = row_start + tl.arange(0, 64)
    col_offs = col_start + tl.arange(0, 64)
    ptrs = base_ptr + (bh * S + row_offs[:, None]) * 128 + col_offs[None, :]
    
    mask = (row_offs[:, None] < S)
    tl.store(ptrs, value, mask=mask)


@triton.jit
def dKdV_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
                S, scale):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    block_j = j * 64
    
    k_tile = load_tile_64x128(K_ptr, bh, block_j, S)
    v_tile = load_tile_64x128(V_ptr, bh, block_j, S)
    
    dk_acc0 = tl.zeros((64, 64), tl.float32)
    dk_acc1 = tl.zeros((64, 64), tl.float32)
    
    dv_acc0 = tl.zeros((64, 64), tl.float32)
    dv_acc1 = tl.zeros((64, 64), tl.float32)
    
    num_blocks = triton.cdiv(S, 64)
    
    for i in range(num_blocks):
        block_i = i * 64
        
        q_tile = load_tile_64x128(Q_ptr, bh, block_i, S)
        do_tile = load_tile_64x128(dO_ptr, bh, block_i, S)
        o_tile = load_tile_64x128(O_ptr, bh, block_i, S)
        
        q0 = q_tile[:, 0:64]
        q1 = q_tile[:, 64:128]
        
        do0 = do_tile[:, 0:64]
        do1 = do_tile[:, 64:128]
        
        o0 = o_tile[:, 0:64]
        o1 = o_tile[:, 64:128]
        
        d_i = tl.sum(do0 * o0, axis=1) + tl.sum(do1 * o1, axis=1)
        
        l_i = tl.load(L_ptr + bh * S + block_i + tl.arange(0, 64), mask=(block_i + tl.arange(0, 64)) < S, other=-float('inf'))
        
        k0 = k_tile[:, 0:64]
        k1 = k_tile[:, 64:128]
        
        v0 = v_tile[:, 0:64]
        v1 = v_tile[:, 64:128]
        
        acc_s = tl.dot(q0, k0.T)
        acc_s = tl.dot(q1, k1.T, acc_s)
        
        acc_dp = tl.dot(do0, v0.T)
        acc_dp = tl.dot(do1, v1.T, acc_dp)
        
        l_i_expanded = l_i[:, None]
        
        s_tmp = acc_s * scale
        
        p_tmp = tl.exp(s_tmp - l_i_expanded)
        
        row_offs_i = block_i + tl.arange(0, 64)
        col_offs_j = block_j + tl.arange(0, 64)
        pos_mask = (row_offs_i[:, None] < S) & (col_offs_j[None, :] < S)
        
        p_tmp = tl.where(pos_mask, p_tmp, 0.0)
        
        d_i_expanded = d_i[:, None]
        
        ds_tmp = p_tmp * (acc_dp - d_i_expanded) * scale
        
        ds_tmp = tl.where(pos_mask, ds_tmp, 0.0)
        
        dv_acc0 = tl.dot(p_tmp.T, do0, dv_acc0)
        dv_acc1 = tl.dot(p_tmp.T, do1, dv_acc1)
        
        dk_acc0 = tl.dot(ds_tmp.T, q0, dk_acc0)
        dk_acc1 = tl.dot(ds_tmp.T, q1, dk_acc1)
        
    store_tile_64x128(dK_ptr, bh, block_j, dk_acc0.to(tl.bfloat16), S, col_start=0)
    store_tile_64x128(dK_ptr, bh, block_j, dk_acc1.to(tl.bfloat16), S, col_start=64)
    store_tile_64x128(dV_ptr, bh, block_j, dv_acc0.to(tl.bfloat16), S, col_start=0)
    store_tile_64x128(dV_ptr, bh, block_j, dv_acc1.to(tl.bfloat16), S, col_start=64)


@triton.jit
def dQ_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
              S, scale):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    block_i = i * 64
    
    q_tile = load_tile_64x128(Q_ptr, bh, block_i, S)
    do_tile = load_tile_64x128(dO_ptr, bh, block_i, S)
    o_tile = load_tile_64x128(O_ptr, bh, block_i, S)
    
    q0 = q_tile[:, 0:64]
    q1 = q_tile[:, 64:128]
    
    do0 = do_tile[:, 0:64]
    do1 = do_tile[:, 64:128]
    
    o0 = o_tile[:, 0:64]
    o1 = o_tile[:, 64:128]
    
    d_i = tl.sum(do0 * o0, axis=1) + tl.sum(do1 * o1, axis=1)
    
    l_i = tl.load(L_ptr + bh * S + block_i + tl.arange(0, 64), mask=(block_i + tl.arange(0, 64)) < S, other=-float('inf'))
    
    dq_acc0 = tl.zeros((64, 64), tl.float32)
    dq_acc1 = tl.zeros((64, 64), tl.float32)
    
    num_blocks = triton.cdiv(S, 64)
    
    for j in range(num_blocks):
        block_j = j * 64
        
        k_tile = load_tile_64x128(K_ptr, bh, block_j, S)
        v_tile = load_tile_64x128(V_ptr, bh, block_j, S)
        
        k0 = k_tile[:, 0:64]
        k1 = k_tile[:, 64:128]
        
        v0 = v_tile[:, 0:64]
        v1 = v_tile[:, 64:128]
        
        acc_s = tl.dot(q0, k0.T)
        acc_s = tl.dot(q1, k1.T, acc_s)
        
        acc_dp = tl.dot(do0, v0.T)
        acc_dp = tl.dot(do1, v1.T, acc_dp)
        
        l_i_expanded = l_i[:, None]
        
        s_tmp = acc_s * scale
        
        p_tmp = tl.exp(s_tmp - l_i_expanded)
        
        row_offs_i = block_i + tl.arange(0, 64)
        col_offs_j = block_j + tl.arange(0, 64)
        pos_mask = (row_offs_i[:, None] < S) & (col_offs_j[None, :] < S)
        
        p_tmp = tl.where(pos_mask, p_tmp, 0.0)
        
        d_i_expanded = d_i[:, None]
        
        ds_tmp = p_tmp * (acc_dp - d_i_expanded) * scale
        
        ds_tmp = tl.where(pos_mask, ds_tmp, 0.0)
        
        dq_acc0 = tl.dot(ds_tmp, k0, dq_acc0)
        dq_acc1 = tl.dot(ds_tmp, k1, dq_acc1)
        
    store_tile_64x128(dQ_ptr, bh, block_i, dq_acc0.to(tl.bfloat16), S, col_start=0)
    store_tile_64x128(dQ_ptr, bh, block_i, dq_acc1.to(tl.bfloat16), S, col_start=64)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Computes the backward pass of multi-head attention natively on CUDA."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / (128 ** 0.5)
    
    grid = (triton.cdiv(S, 64), B * H)
    
    dKdV_kernel[grid](Q, K, V, O, dO, L, dK, dV, S, scale, num_warps=8, num_stages=3)
    dQ_kernel[grid](Q, K, V, O, dO, L, dQ, S, scale, num_warps=8, num_stages=3)