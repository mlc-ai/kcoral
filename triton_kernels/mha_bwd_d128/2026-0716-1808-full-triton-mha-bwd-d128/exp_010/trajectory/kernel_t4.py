import torch
import triton
import triton.language as tl


@triton.jit
def load_tile(base_ptr, bh, block_row, col_start, S):
    """Load a [64, 64] bf16 tile."""
    row_offs = block_row + tl.arange(0, 64)
    col_offs = col_start + tl.arange(0, 64)
    ptrs = base_ptr + (bh * S + row_offs[:, None]) * 128 + col_offs[None, :]
    return tl.load(ptrs, mask=(row_offs[:, None] < S), other=0.0)


@triton.jit
def store_tile(base_ptr, bh, row_start, value, S, col_start=0):
    """Store a [64, 64] bf16 tile."""
    row_offs = row_start + tl.arange(0, 64)
    col_offs = col_start + tl.arange(0, 64)
    ptrs = base_ptr + (bh * S + row_offs[:, None]) * 128 + col_offs[None, :]
    mask = (row_offs[:, None] < S)
    tl.store(ptrs, value, mask=mask)


@triton.jit(alias_bases=[(dO_ptr, dK_ptr), (O_ptr, dV_ptr)])
def dKdV_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
                S, scale):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    block_j = j * 64
    
    k0 = load_tile(K_ptr, bh, block_j, 0, S)
    k1 = load_tile(K_ptr, bh, block_j, 64, S)
    
    v0 = load_tile(V_ptr, bh, block_j, 0, S)
    v1 = load_tile(V_ptr, bh, block_j, 64, S)
    
    dk_acc0 = tl.zeros((64, 64), tl.float32)
    dk_acc1 = tl.zeros((64, 64), tl.float32)
    
    dv_acc0 = tl.zeros((64, 64), tl.float32)
    dv_acc1 = tl.zeros((64, 64), tl.float32)
    
    num_blocks = tl.cdiv(S, 64)
    
    for i in range(num_blocks):
        block_i = i * 64
        
        q0 = load_tile(Q_ptr, bh, block_i, 0, S)
        q1 = load_tile(Q_ptr, bh, block_i, 64, S)
        
        do0 = load_tile(dO_ptr, bh, block_i, 0, S)
        do1 = load_tile(dO_ptr, bh, block_i, 64, S)
        
        o0 = load_tile(O_ptr, bh, block_i, 0, S)
        o1 = load_tile(O_ptr, bh, block_i, 64, S)
        
        d_i = tl.sum(do0 * o0, axis=1) + tl.sum(do1 * o1, axis=1)
        
        l_i = tl.load(L_ptr + bh * S + block_i + tl.arange(0, 64), mask=(block_i + tl.arange(0, 64)) < S, other=-float('inf'))
        
        acc_s = tl.dot(q0, k0.T)
        acc_s = tl.dot(q1, k1.T, acc_s)
        
        acc_dp = tl.dot(do0, v0.T)
        acc_dp = tl.dot(do1, v1.T, acc_dp)
        
        s_tmp = acc_s * scale
        
        l_i_expanded = l_i[:, None]
        
        row_offs_i = block_i + tl.arange(0, 64)
        col_offs_j = block_j + tl.arange(0, 64)
        pos_mask = (row_offs_i[:, None] < S) & (col_offs_j[None, :] < S)
        
        p_tmp = tl.exp(s_tmp - l_i_expanded)
        p_tmp = tl.where(pos_mask, p_tmp, 0.0)
        
        d_i_expanded = d_i[:, None]
        
        ds_tmp = p_tmp * (acc_dp - d_i_expanded) * scale
        
        dv_acc0 = tl.dot(p_tmp.T, do0, dv_acc0)
        dv_acc1 = tl.dot(p_tmp.T, do1, dv_acc1)
        
        dk_acc0 = tl.dot(ds_tmp.T, q0, dk_acc0)
        dk_acc1 = tl.dot(ds_tmp.T, q1, dk_acc1)
        
    store_tile(dK_ptr, bh, block_j, dk_acc0.to(tl.bfloat16), S, 0)
    store_tile(dK_ptr, bh, block_j, dk_acc1.to(tl.bfloat16), S, 64)
    
    store_tile(dV_ptr, bh, block_j, dv_acc0.to(tl.bfloat16), S, 0)
    store_tile(dV_ptr, bh, block_j, dv_acc1.to(tl.bfloat16), S, 64)


@triton.jit
def dQ_kernel(Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
              S, scale):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    block_i = i * 64
    
    q0 = load_tile(Q_ptr, bh, block_i, 0, S)
    q1 = load_tile(Q_ptr, bh, block_i, 64, S)
    
    do0 = load_tile(dO_ptr, bh, block_i, 0, S)
    do1 = load_tile(dO_ptr, bh, block_i, 64, S)
    
    o0 = load_tile(O_ptr, bh, block_i, 0, S)
    o1 = load_tile(O_ptr, bh, block_i, 64, S)
    
    d_i = tl.sum(do0 * o0, axis=1) + tl.sum(do1 * o1, axis=1)
    
    l_i = tl.load(L_ptr + bh * S + block_i + tl.arange(0, 64), mask=(block_i + tl.arange(0, 64)) < S, other=-float('inf'))
    
    dq_acc0 = tl.zeros((64, 64), tl.float32)
    dq_acc1 = tl.zeros((64, 64), tl.float32)
    
    num_blocks = tl.cdiv(S, 64)
    
    for j in range(num_blocks):
        block_j = j * 64
        
        k0 = load_tile(K_ptr, bh, block_j, 0, S)
        k1 = load_tile(K_ptr, bh, block_j, 64, S)
        
        v0 = load_tile(V_ptr, bh, block_j, 0, S)
        v1 = load_tile(V_ptr, bh, block_j, 64, S)
        
        acc_s = tl.dot(q0, k0.T)
        acc_s = tl.dot(q1, k1.T, acc_s)
        
        acc_dp = tl.dot(do0, v0.T)
        acc_dp = tl.dot(do1, v1.T, acc_dp)
        
        s_tmp = acc_s * scale
        
        l_i_expanded = l_i[:, None]
        
        row_offs_i = block_i + tl.arange(0, 64)
        col_offs_j = block_j + tl.arange(0, 64)
        pos_mask = (row_offs_i[:, None] < S) & (col_offs_j[None, :] < S)
        
        p_tmp = tl.exp(s_tmp - l_i_expanded)
        p_tmp = tl.where(pos_mask, p_tmp, 0.0)
        
        d_i_expanded = d_i[:, None]
        
        ds_tmp = p_tmp * (acc_dp - d_i_expanded) * scale
        
        dq_acc0 = tl.dot(ds_tmp, k0, dq_acc0)
        dq_acc1 = tl.dot(ds_tmp, k1, dq_acc1)
        
    store_tile(dQ_ptr, bh, block_i, dq_acc0.to(tl.bfloat16), S, 0)
    store_tile(dQ_ptr, bh, block_i, dq_acc1.to(tl.bfloat16), S, 64)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Computes the backward pass of multi-head attention natively on CUDA."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / (128 ** 0.5)
    
    grid = (triton.cdiv(S, 64), B * H)
    
    # Allow dQ to execute cleanly on unaliased intact tensors initially 
    dQ_kernel[grid](Q, K, V, O, dO, L, dQ, S, scale, num_warps=8, num_stages=3)
    
    # Utilize aliasing base tricks to avoid allocating fresh shared memory for dK/dV output staging. 
    # Provide original O/dO pointers as destinations to leverage already-reservedSMEM spaces.
    dKdV_kernel[grid](Q, K, V, O, dO, L, dO, O, S, scale, num_warps=8, num_stages=3)