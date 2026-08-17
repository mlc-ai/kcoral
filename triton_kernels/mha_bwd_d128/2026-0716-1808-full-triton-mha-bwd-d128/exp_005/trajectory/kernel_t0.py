import torch
import triton
import triton.language as tl
import math


@triton.jit
def _bwd_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    S_len, scale,
    s_stride, h_stride, b_stride,
    l_h_stride, l_b_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    j_block = j * BLOCK_M
    
    row_off_j = j_block + tl.arange(0, BLOCK_M)
    row_off_q_base = tl.arange(0, BLOCK_M)
    col_off = tl.arange(0, BLOCK_N)
    
    q_base = Q + b * b_stride + h * h_stride
    k_base = K + b * b_stride + h * h_stride
    v_base = V + b * b_stride + h * h_stride
    o_base = O + b * b_stride + h * h_stride
    do_base = dO + b * b_stride + h * h_stride
    
    K_tile = tl.load(k_base + row_off_j[:, None] * s_stride + col_off[None, :],
                      mask=(row_off_j < S_len)[:, None], other=0.0)
    V_tile = tl.load(v_base + row_off_j[:, None] * s_stride + col_off[None, :],
                      mask=(row_off_j < S_len)[:, None], other=0.0)
    
    dK_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dV_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_q_blocks = tl.cdiv(S_len, BLOCK_M)
    for i in range(num_q_blocks):
        i_block = i * BLOCK_M
        row_off_q = i_block + row_off_q_base
        
        Q_tile = tl.load(q_base + row_off_q[:, None] * s_stride + col_off[None, :],
                          mask=(row_off_q < S_len)[:, None], other=0.0)
        dO_tile = tl.load(do_base + row_off_q[:, None] * s_stride + col_off[None, :],
                           mask=(row_off_q < S_len)[:, None], other=0.0)
        O_tile = tl.load(o_base + row_off_q[:, None] * s_stride + col_off[None, :],
                          mask=(row_off_q < S_len)[:, None], other=0.0)
        
        D = tl.sum(dO_tile * O_tile, axis=1)
        
        S = tl.dot(Q_tile, K_tile.T)
        
        dP = tl.dot(dO_tile, V_tile.T)
        
        L_tile = tl.load(L + b * l_b_stride + h * l_h_stride + row_off_q,
                          mask=(row_off_q < S_len), other=0.0)
        
        P = tl.exp(S * scale - L_tile[:, None])
        
        col_idx = tl.arange(0, BLOCK_M)
        key_mask = (j_block + col_idx < S_len)[None, :]
        P = P * key_mask
        
        dS = P * (dP - D[:, None]) * scale
        
        dV_acc = tl.dot(P.T, dO_tile, dV_acc)
        dK_acc = tl.dot(dS.T, Q_tile, dK_acc)
    
    out_K = dK + b * b_stride + h * h_stride + j_block * s_stride
    out_V = dV + b * b_stride + h * h_stride + j_block * s_stride
    
    store_mask = (row_off_j[:, None] < S_len)
    tl.store(out_K + row_off_j[:, None] * s_stride + col_off[None, :],
             dK_acc.to(tl.bfloat16), mask=store_mask)
    tl.store(out_V + row_off_j[:, None] * s_stride + col_off[None, :],
             dV_acc.to(tl.bfloat16), mask=store_mask)


@triton.jit
def _bwd_dq(
    Q, K, V, O, dO, L, dQ,
    S_len, scale,
    s_stride, h_stride, b_stride,
    l_h_stride, l_b_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    i_block = i * BLOCK_M
    
    row_off_q = i_block + tl.arange(0, BLOCK_M)
    row_off_k_base = tl.arange(0, BLOCK_M)
    col_off = tl.arange(0, BLOCK_N)
    
    q_base = Q + b * b_stride + h * h_stride
    k_base = K + b * b_stride + h * h_stride
    v_base = V + b * b_stride + h * h_stride
    o_base = O + b * b_stride + h * h_stride
    do_base = dO + b * b_stride + h * h_stride
    
    Q_tile = tl.load(q_base + row_off_q[:, None] * s_stride + col_off[None, :],
                      mask=(row_off_q < S_len)[:, None], other=0.0)
    dO_tile = tl.load(do_base + row_off_q[:, None] * s_stride + col_off[None, :],
                       mask=(row_off_q < S_len)[:, None], other=0.0)
    O_tile = tl.load(o_base + row_off_q[:, None] * s_stride + col_off[None, :],
                      mask=(row_off_q < S_len)[:, None], other=0.0)
    
    D = tl.sum(dO_tile * O_tile, axis=1)
    
    L_tile = tl.load(L + b * l_b_stride + h * l_h_stride + row_off_q,
                      mask=(row_off_q < S_len), other=0.0)
    
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_kv_blocks = tl.cdiv(S_len, BLOCK_M)
    for j in range(num_kv_blocks):
        j_block = j * BLOCK_M
        row_off_k = j_block + row_off_k_base
        
        K_tile = tl.load(k_base + row_off_k[:, None] * s_stride + col_off[None, :],
                          mask=(row_off_k < S_len)[:, None], other=0.0)
        V_tile = tl.load(v_base + row_off_k[:, None] * s_stride + col_off[None, :],
                          mask=(row_off_k < S_len)[:, None], other=0.0)
        
        S = tl.dot(Q_tile, K_tile.T)
        
        dP = tl.dot(dO_tile, V_tile.T)
        
        P = tl.exp(S * scale - L_tile[:, None])
        
        col_idx = tl.arange(0, BLOCK_M)
        key_mask = (j_block + col_idx < S_len)[None, :]
        P = P * key_mask
        
        dS = P * (dP - D[:, None]) * scale
        
        dQ_acc = tl.dot(dS, K_tile, dQ_acc)
    
    out_Q = dQ + b * b_stride + h * h_stride + i_block * s_stride
    store_mask = (row_off_q[:, None] < S_len)
    tl.store(out_Q + row_off_q[:, None] * s_stride + col_off[None, :],
             dQ_acc.to(tl.bfloat16), mask=store_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward gradients into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    s_stride = d
    h_stride = S * d
    b_stride = H * S * d
    l_h_stride = S
    l_b_stride = H * S
    
    block_m = 64
    block_n = 128 
    
    num_blocks = triton.cdiv(S, block_m)
    grid = (num_blocks, H, B)
    
    _bwd_dk_dv[grid](
        Q, K, V, O, dO, L, dK, dV,
        S, scale,
        s_stride, h_stride, b_stride,
        l_h_stride, l_b_stride,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4,
    )
    
    _bwd_dq[grid](
        Q, K, V, O, dO, L, dQ,
        S, scale,
        s_stride, h_stride, b_stride,
        l_h_stride, l_b_stride,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4,
    )