import torch
import triton
import triton.language as tl
import math


@triton.jit
def _bwd_dk_dv(
    Q, K, V, O, dO, L, dK, dV,
    S_len, scale, H, B,
    s_stride, h_stride, b_stride,
    l_h_stride, l_b_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0) 
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    j_block = j * BLOCK_M
    row_off_j = j_block + tl.arange(0, BLOCK_M)
    col_off = tl.arange(0, BLOCK_N)
    
    j_base = b * b_stride + h * h_stride + j_block * s_stride
    
    K_tile = tl.load(K + j_base + row_off_j[:, None] * s_stride + col_off[None, :],
                     mask=(row_off_j < S_len)[:, None], other=0.0)
    V_tile = tl.load(V + j_base + row_off_j[:, None] * s_stride + col_off[None, :],
                     mask=(row_off_j < S_len)[:, None], other=0.0)
    
    dK_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dV_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_q_blocks = tl.cdiv(S_len, BLOCK_M)
    for i in range(num_q_blocks):
        i_block = i * BLOCK_M
        row_off_q = i_block + tl.arange(0, BLOCK_M)
        
        q_base = b * b_stride + h * h_stride + i_block * s_stride
        
        Q_tile = tl.load(Q + q_base + row_off_q[:, None] * s_stride + col_off[None, :],
                          mask=(row_off_q < S_len)[:, None], other=0.0)
        dO_tile = tl.load(dO + q_base + row_off_q[:, None] * s_stride + col_off[None, :],
                           mask=(row_off_q < S_len)[:, None], other=0.0)
        O_tile = tl.load(O + q_base + row_off_q[:, None] * s_stride + col_off[None, :],
                          mask=(row_off_q < S_len)[:, None], other=0.0)
        
        D = tl.sum(dO_tile * O_tile, axis=1)
        
        l_base_idx = b * l_b_stride + h * l_h_stride
        L_tile = tl.load(L + l_base_idx + row_off_q,
                          mask=(row_off_q < S_len), other=0.0)
        
        S_acc = tl.dot(Q_tile, K_tile.T)
        
        dP_acc = tl.dot(dO_tile, V_tile.T)
        
        P = tl.exp(S_acc * scale - L_tile[:, None])
        
        col_idx = tl.arange(0, BLOCK_M)
        key_mask = ((j_block + col_idx) < S_len)[None, :]
        P = P * key_mask
        
        dS = P * (dP_acc - D[:, None]) * scale
        
        dK_acc = tl.dot(dS.T, Q_tile, dK_acc)
        dV_acc = tl.dot(P.T, dO_tile, dV_acc)
        
    store_mask = (row_off_j < S_len)[:, None]
    
    tl.store(dK + j_base + row_off_j[:, None] * s_stride + col_off[None, :],
             dK_acc.to(tl.bfloat16), mask=store_mask)
    tl.store(dV + j_base + row_off_j[:, None] * s_stride + col_off[None, :],
             dV_acc.to(tl.bfloat16), mask=store_mask)


@triton.jit
def _bwd_dq(
    Q, K, V, O, dO, L, dQ,
    S_len, scale, H, B,
    s_stride, h_stride, b_stride,
    l_h_stride, l_b_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0) 
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    i_block = i * BLOCK_M
    row_off_q = i_block + tl.arange(0, BLOCK_M)
    col_off = tl.arange(0, BLOCK_N)
    
    q_base = b * b_stride + h * h_stride + i_block * s_stride
    
    Q_tile = tl.load(Q + q_base + row_off_q[:, None] * s_stride + col_off[None, :],
                      mask=(row_off_q < S_len)[:, None], other=0.0)
    dO_tile = tl.load(dO + q_base + row_off_q[:, None] * s_stride + col_off[None, :],
                       mask=(row_off_q < S_len)[:, None], other=0.0)
    O_tile = tl.load(O + q_base + row_off_q[:, None] * s_stride + col_off[None, :],
                      mask=(row_off_q < S_len)[:, None], other=0.0)
    
    D = tl.sum(dO_tile * O_tile, axis=1)
    
    l_base_idx = b * l_b_stride + h * l_h_stride
    L_tile = tl.load(L + l_base_idx + row_off_q,
                      mask=(row_off_q < S_len), other=0.0)
    
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_kv_blocks = tl.cdiv(S_len, BLOCK_M)
    for j in range(num_kv_blocks):
        j_block = j * BLOCK_M
        row_off_k = j_block + tl.arange(0, BLOCK_M)
        
        k_base = b * b_stride + h * h_stride + j_block * s_stride
        
        K_tile = tl.load(K + k_base + row_off_k[:, None] * s_stride + col_off[None, :],
                          mask=(row_off_k < S_len)[:, None], other=0.0)
        V_tile = tl.load(V + k_base + row_off_k[:, None] * s_stride + col_off[None, :],
                          mask=(row_off_k < S_len)[:, None], other=0.0)
        
        S_acc = tl.dot(Q_tile, K_tile.T)
        
        dP_acc = tl.dot(dO_tile, V_tile.T)
        
        P = tl.exp(S_acc * scale - L_tile[:, None])
        
        col_idx = tl.arange(0, BLOCK_M)
        key_mask = ((j_block + col_idx) < S_len)[None, :]
        P = P * key_mask
        
        dS = P * (dP_acc - D[:, None]) * scale
        
        dQ_acc = tl.dot(dS, K_tile, dQ_acc)
    
    store_mask = (row_off_q < S_len)[:, None]
    
    tl.store(dQ + q_base + row_off_q[:, None] * s_stride + col_off[None, :],
             dQ_acc.to(tl.bfloat16), mask=store_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward gradients into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    s_stride = 128 
    h_stride = 128 * S
    b_stride = 128 * S * H
    l_h_stride = S
    l_b_stride = H * S
    
    block_m = 64
    block_n = 128 
    
    num_blocks = triton.cdiv(S, block_m)
    grid = (num_blocks, H, B)
    
    _bwd_dk_dv[grid](
        Q, K, V, O, dO, L, dK, dV,
        S, scale, H, B,
        s_stride, h_stride, b_stride,
        l_h_stride, l_b_stride,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=8, num_stages=3,
    )
    
    _bwd_dq[grid](
        Q, K, V, O, dO, L, dQ,
        S, scale, H, B,
        s_stride, h_stride, b_stride,
        l_h_stride, l_b_stride,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=8, num_stages=3,
    )