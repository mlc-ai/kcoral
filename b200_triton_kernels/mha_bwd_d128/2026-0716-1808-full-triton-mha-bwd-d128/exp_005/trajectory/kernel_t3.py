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
    half = tl.program_id(3)
    
    bh_idx = b * H + h
    j_block = j * BLOCK_M
    row_off_j = j_block + tl.arange(0, BLOCK_M)
    
    col_off0 = tl.arange(0, BLOCK_N)
    col_off1 = BLOCK_N + tl.arange(0, BLOCK_N)
    
    K_ptr0 = K + bh_idx * s_stride + j_block * s_stride
    K_ptr1 = K + bh_idx * s_stride + j_block * s_stride
    
    k_ptr0 = K_ptr0 + row_off_j[:, None] * s_stride + col_off0[None, :]
    k_ptr1 = K_ptr1 + row_off_j[:, None] * s_stride + col_off1[None, :]
    
    K0 = tl.load(k_ptr0, mask=(row_off_j < S_len)[:, None], other=0.0).to(tl.float32)
    K1 = tl.load(k_ptr1, mask=(row_off_j < S_len)[:, None], other=0.0).to(tl.float32)
    
    V0 = tl.load(V + bh_idx * s_stride + j_block * s_stride + row_off_j[:, None] * s_stride + col_off0[None, :],
                 mask=(row_off_j < S_len)[:, None], other=0.0).to(tl.float32)
    V1 = tl.load(V + bh_idx * s_stride + j_block * s_stride + row_off_j[:, None] * s_stride + col_off1[None, :],
                 mask=(row_off_j < S_len)[:, None], other=0.0).to(tl.float32)
    
    dK_acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dK_acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dV_acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dV_acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_q_blocks = tl.cdiv(S_len, BLOCK_M)
    for i in range(num_q_blocks):
        i_block = i * BLOCK_M
        row_off_q = i_block + tl.arange(0, BLOCK_M)
        
        q_ptr0 = Q + bh_idx * s_stride + i_block * s_stride
        
        Q0 = tl.load(q_ptr0 + row_off_q[:, None] * s_stride + col_off0[None, :],
                      mask=(row_off_q < S_len)[:, None], other=0.0).to(tl.float32)
        Q1 = tl.load(q_ptr0 + row_off_q[:, None] * s_stride + col_off1[None, :],
                      mask=(row_off_q < S_len)[:, None], other=0.0).to(tl.float32)
                      
        do_ptr0 = dO + bh_idx * s_stride + i_block * s_stride
        dO0 = tl.load(do_ptr0 + row_off_q[:, None] * s_stride + col_off0[None, :],
                       mask=(row_off_q < S_len)[:, None], other=0.0).to(tl.float32)
        dO1 = tl.load(do_ptr0 + row_off_q[:, None] * s_stride + col_off1[None, :],
                       mask=(row_off_q < S_len)[:, None], other=0.0).to(tl.float32)
                       
        o_ptr0 = O + bh_idx * s_stride + i_block * s_stride
        O0 = tl.load(o_ptr0 + row_off_q[:, None] * s_stride + col_off0[None, :],
                      mask=(row_off_q < S_len)[:, None], other=0.0).to(tl.float32)
        O1 = tl.load(o_ptr0 + row_off_q[:, None] * s_stride + col_off1[None, :],
                      mask=(row_off_q < S_len)[:, None], other=0.0).to(tl.float32)
        
        D = tl.sum(dO0 * O0, axis=1) + tl.sum(dO1 * O1, axis=1)
        
        l_base_idx = b * l_b_stride + h * l_h_stride
        L_tile = tl.load(L + l_base_idx + row_off_q,
                          mask=(row_off_q < S_len), other=0.0)
        
        S_acc = tl.dot(Q0, K0.T, input_precision="ieee")
        S_acc = tl.dot(Q1, K1.T, S_acc, input_precision="ieee")
        
        dP_acc = tl.dot(dO0, V0.T, input_precision="ieee")
        dP_acc = tl.dot(dO1, V1.T, dP_acc, input_precision="ieee")
        
        P = tl.exp(S_acc * scale - L_tile[:, None])
        
        col_idx = tl.arange(0, BLOCK_M)
        key_mask = ((j_block + col_idx) < S_len)[None, :]
        P = P * key_mask
        
        dS = P * (dP_acc - D[:, None]) * scale
        
        dK_acc0 = tl.dot(dS.T, Q0, dK_acc0, input_precision="ieee")
        dK_acc1 = tl.dot(dS.T, Q1, dK_acc1, input_precision="ieee")
        
        dV_acc0 = tl.dot(P.T, dO0, dV_acc0, input_precision="ieee")
        dV_acc1 = tl.dot(P.T, dO1, dV_acc1, input_precision="ieee")
        
    store_mask = (row_off_j < S_len)[:, None]
    
    dK_ptr0 = dK + bh_idx * s_stride + j_block * s_stride
    tl.store(dK_ptr0 + row_off_j[:, None] * s_stride + col_off0[None, :],
             dK_acc0.to(tl.bfloat16), mask=store_mask)
    tl.store(dK_ptr0 + row_off_j[:, None] * s_stride + col_off1[None, :],
             dK_acc1.to(tl.bfloat16), mask=store_mask)
             
    dV_ptr0 = dV + bh_idx * s_stride + j_block * s_stride
    tl.store(dV_ptr0 + row_off_j[:, None] * s_stride + col_off0[None, :],
             dV_acc0.to(tl.bfloat16), mask=store_mask)
    tl.store(dV_ptr0 + row_off_j[:, None] * s_stride + col_off1[None, :],
             dV_acc1.to(tl.bfloat16), mask=store_mask)


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
    half = tl.program_id(3)
    
    bh_idx = b * H + h
    i_block = i * BLOCK_M
    row_off_q = i_block + tl.arange(0, BLOCK_M)
    
    col_off0 = tl.arange(0, BLOCK_N)
    col_off1 = BLOCK_N + tl.arange(0, BLOCK_N)
    
    q_ptr0 = Q + bh_idx * s_stride + i_block * s_stride
    
    Q0 = tl.load(q_ptr0 + row_off_q[:, None] * s_stride + col_off0[None, :],
                  mask=(row_off_q < S_len)[:, None], other=0.0).to(tl.float32)
    Q1 = tl.load(q_ptr0 + row_off_q[:, None] * s_stride + col_off1[None, :],
                  mask=(row_off_q < S_len)[:, None], other=0.0).to(tl.float32)
                  
    do_ptr0 = dO + bh_idx * s_stride + i_block * s_stride
    dO0 = tl.load(do_ptr0 + row_off_q[:, None] * s_stride + col_off0[None, :],
                   mask=(row_off_q < S_len)[:, None], other=0.0).to(tl.float32)
    dO1 = tl.load(do_ptr0 + row_off_q[:, None] * s_stride + col_off1[None, :],
                   mask=(row_off_q < S_len)[:, None], other=0.0).to(tl.float32)
                   
    o_ptr0 = O + bh_idx * s_stride + i_block * s_stride
    O0 = tl.load(o_ptr0 + row_off_q[:, None] * s_stride + col_off0[None, :],
                  mask=(row_off_q < S_len)[:, None], other=0.0).to(tl.float32)
    O1 = tl.load(o_ptr0 + row_off_q[:, None] * s_stride + col_off1[None, :],
                  mask=(row_off_q < S_len)[:, None], other=0.0).to(tl.float32)
    
    D = tl.sum(dO0 * O0, axis=1) + tl.sum(dO1 * O1, axis=1)
    
    l_base_idx = b * l_b_stride + h * l_h_stride
    L_tile = tl.load(L + l_base_idx + row_off_q,
                      mask=(row_off_q < S_len), other=0.0)
    
    dQ_acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dQ_acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_kv_blocks = tl.cdiv(S_len, BLOCK_M)
    for j in range(num_kv_blocks):
        j_block = j * BLOCK_M
        row_off_k = j_block + tl.arange(0, BLOCK_M)
        
        k_ptr0 = K + bh_idx * s_stride + j_block * s_stride
        
        K0 = tl.load(k_ptr0 + row_off_k[:, None] * s_stride + col_off0[None, :],
                      mask=(row_off_k < S_len)[:, None], other=0.0).to(tl.float32)
        K1 = tl.load(k_ptr0 + row_off_k[:, None] * s_stride + col_off1[None, :],
                      mask=(row_off_k < S_len)[:, None], other=0.0).to(tl.float32)
                      
        v_ptr0 = V + bh_idx * s_stride + j_block * s_stride
        V0 = tl.load(v_ptr0 + row_off_k[:, None] * s_stride + col_off0[None, :],
                      mask=(row_off_k < S_len)[:, None], other=0.0).to(tl.float32)
        V1 = tl.load(v_ptr0 + row_off_k[:, None] * s_stride + col_off1[None, :],
                      mask=(row_off_k < S_len)[:, None], other=0.0).to(tl.float32)
        
        S_acc = tl.dot(Q0, K0.T, input_precision="ieee")
        S_acc = tl.dot(Q1, K1.T, S_acc, input_precision="ieee")
        
        dP_acc = tl.dot(dO0, V0.T, input_precision="ieee")
        dP_acc = tl.dot(dO1, V1.T, dP_acc, input_precision="ieee")
        
        P = tl.exp(S_acc * scale - L_tile[:, None])
        
        col_idx = tl.arange(0, BLOCK_M)
        key_mask = ((j_block + col_idx) < S_len)[None, :]
        P = P * key_mask
        
        dS = P * (dP_acc - D[:, None]) * scale
        
        dQ_acc0 = tl.dot(dS, K0, dQ_acc0, input_precision="ieee")
        dQ_acc1 = tl.dot(dS, K1, dQ_acc1, input_precision="ieee")
    
    store_mask = (row_off_q < S_len)[:, None]
    
    dQ_ptr0 = dQ + bh_idx * s_stride + i_block * s_stride
    tl.store(dQ_ptr0 + row_off_q[:, None] * s_stride + col_off0[None, :],
             dQ_acc0.to(tl.bfloat16), mask=store_mask)
    tl.store(dQ_ptr0 + row_off_q[:, None] * s_stride + col_off1[None, :],
             dQ_acc1.to(tl.bfloat16), mask=store_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward gradients into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    # Ensure tensors are row-major to guarantee uniform and predictable element strides.
    Q = Q.to(row_major=True)
    K = K.to(row_major=True)
    V = V.to(row_major=True)
    O = O.to(row_major=True)
    dO = dO.to(row_major=True)
    dQ = dQ.to(row_major=True)
    dK = dK.to(row_major=True)
    dV = dV.to(row_major=True)
    
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    s_stride = d
    h_stride = S * d
    b_stride = H * S * d
    l_h_stride = S
    l_b_stride = H * S
    
    block_m = 64
    block_n = 64 
    
    num_blocks = triton.cdiv(S, block_m)
    grid = (num_blocks, H, B, 2)
    
    _bwd_dk_dv[grid](
        Q, K, V, O, dO, L, dK, dV,
        S, scale, H, B,
        s_stride, h_stride, b_stride,
        l_h_stride, l_b_stride,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4, num_stages=2,
    )
    
    _bwd_dq[grid](
        Q, K, V, O, dO, L, dQ,
        S, scale, H, B,
        s_stride, h_stride, b_stride,
        l_h_stride, l_b_stride,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4, num_stages=2,
    )