import math
import torch
import triton
import triton.language as tl


BLOCK_Q = 128


@triton.jit
def dKdV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    S,
    SCALE: tl.constexpr,
):
    pid_j = tl.program_id(0)
    bh = tl.program_id(1)
    
    j_start = pid_j * BLOCK_Q
    
    row_idx = tl.arange(0, BLOCK_Q)
    cols_0 = tl.arange(0, 64)
    cols_1 = 64 + tl.arange(0, 64)
    
    k_row_ptr = K_ptr + bh * S * 128 + j_start * 128
    K_0 = tl.load(k_row_ptr + row_idx[:, None] * 128 + cols_0[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_0 < 128)[None, :], other=0.0)
    K_1 = tl.load(k_row_ptr + row_idx[:, None] * 128 + cols_1[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_1 < 128)[None, :], other=0.0)
    
    v_row_ptr = V_ptr + bh * S * 128 + j_start * 128
    V_0 = tl.load(v_row_ptr + row_idx[:, None] * 128 + cols_0[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_0 < 128)[None, :], other=0.0)
    V_1 = tl.load(v_row_ptr + row_idx[:, None] * 128 + cols_1[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_1 < 128)[None, :], other=0.0)
    
    dK_0 = tl.zeros((BLOCK_Q, 64), tl.bfloat16)
    dK_1 = tl.zeros((BLOCK_Q, 64), tl.bfloat16)
    dV_0 = tl.zeros((BLOCK_Q, 64), tl.bfloat16)
    dV_1 = tl.zeros((BLOCK_Q, 64), tl.bfloat16)
    
    num_Q_blocks = tl.cdiv(S, BLOCK_Q)
    for i in range(num_Q_blocks):
        j_start = i * BLOCK_Q
        
        q_row_ptr = Q_ptr + bh * S * 128 + j_start * 128
        Q_0 = tl.load(q_row_ptr + row_idx[:, None] * 128 + cols_0[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_0 < 128)[None, :], other=0.0)
        Q_1 = tl.load(q_row_ptr + row_idx[:, None] * 128 + cols_1[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_1 < 128)[None, :], other=0.0)
        
        o_row_ptr = O_ptr + bh * S * 128 + j_start * 128
        O_0 = tl.load(o_row_ptr + row_idx[:, None] * 128 + cols_0[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_0 < 128)[None, :], other=0.0)
        O_1 = tl.load(o_row_ptr + row_idx[:, None] * 128 + cols_1[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_1 < 128)[None, :], other=0.0)
        
        do_row_ptr = dO_ptr + bh * S * 128 + j_start * 128
        dO_0 = tl.load(do_row_ptr + row_idx[:, None] * 128 + cols_0[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_0 < 128)[None, :], other=0.0)
        dO_1 = tl.load(do_row_ptr + row_idx[:, None] * 128 + cols_1[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_1 < 128)[None, :], other=0.0)
        
        D = tl.sum(O_0 * dO_0 + O_1 * dO_1, axis=1)
        D = D[:, None]
        D = tl.where((row_idx + j_start < S)[:, None], D, 0.0)
        
        S_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        S_acc = tl.dot(Q_0, K_0.T, S_acc)
        S_acc = tl.dot(Q_1, K_1.T, S_acc)
        
        dP_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        dP_acc = tl.dot(dO_0, V_0.T, dP_acc)
        dP_acc = tl.dot(dO_1, V_1.T, dP_acc)
        
        L_j = tl.load(L_ptr + bh * S + j_start + row_idx, mask=(j_start + row_idx < S), other=float('inf'))
        
        P = tl.exp(S_acc * SCALE - L_j[:, None])
        
        dS = P * (dP_acc - D) * SCALE
        
        P_T = P.T
        dS_T = dS.T
        
        dV_0 = dV_0 + tl.dot(P_T, dO_0, out_dtype=tl.bfloat16)
        dV_1 = dV_1 + tl.dot(P_T, dO_1, out_dtype=tl.bfloat16)
        
        dK_0 = dK_0 + tl.dot(dS_T, Q_0, out_dtype=tl.bfloat16)
        dK_1 = dK_1 + tl.dot(dS_T, Q_1, out_dtype=tl.bfloat16)
        
    ptr_k0 = dK_ptr + bh * S * 128 + j_start * 128 + cols_0
    tl.store(ptr_k0, dK_0, mask=(row_idx + j_start < S)[:, None] & (cols_0 < 128)[None, :])
    
    ptr_k1 = dK_ptr + bh * S * 128 + j_start * 128 + cols_1
    tl.store(ptr_k1, dK_1, mask=(row_idx + j_start < S)[:, None] & (cols_1 < 128)[None, :])
    
    ptr_v0 = dV_ptr + bh * S * 128 + j_start * 128 + cols_0
    tl.store(ptr_v0, dV_0, mask=(row_idx + j_start < S)[:, None] & (cols_0 < 128)[None, :])
    
    ptr_v1 = dV_ptr + bh * S * 128 + j_start * 128 + cols_1
    tl.store(ptr_v1, dV_1, mask=(row_idx + j_start < S)[:, None] & (cols_1 < 128)[None, :])


@triton.jit
def dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S,
    SCALE: tl.constexpr,
):
    pid_i = tl.program_id(0)
    bh = tl.program_id(1)
    
    i_start = pid_i * BLOCK_Q
    
    row_idx = tl.arange(0, BLOCK_Q)
    cols_0 = tl.arange(0, 64)
    cols_1 = 64 + tl.arange(0, 64)
    
    q_row_ptr = Q_ptr + bh * S * 128 + i_start * 128
    Q_0 = tl.load(q_row_ptr + row_idx[:, None] * 128 + cols_0[None, :], mask=(row_idx + i_start < S)[:, None] & (cols_0 < 128)[None, :], other=0.0)
    Q_1 = tl.load(q_row_ptr + row_idx[:, None] * 128 + cols_1[None, :], mask=(row_idx + i_start < S)[:, None] & (cols_1 < 128)[None, :], other=0.0)
    
    o_row_ptr = O_ptr + bh * S * 128 + i_start * 128
    O_0 = tl.load(o_row_ptr + row_idx[:, None] * 128 + cols_0[None, :], mask=(row_idx + i_start < S)[:, None] & (cols_0 < 128)[None, :], other=0.0)
    O_1 = tl.load(o_row_ptr + row_idx[:, None] * 128 + cols_1[None, :], mask=(row_idx + i_start < S)[:, None] & (cols_1 < 128)[None, :], other=0.0)
    
    do_row_ptr = dO_ptr + bh * S * 128 + i_start * 128
    dO_0 = tl.load(do_row_ptr + row_idx[:, None] * 128 + cols_0[None, :], mask=(row_idx + i_start < S)[:, None] & (cols_0 < 128)[None, :], other=0.0)
    dO_1 = tl.load(do_row_ptr + row_idx[:, None] * 128 + cols_1[None, :], mask=(row_idx + i_start < S)[:, None] & (cols_1 < 128)[None, :], other=0.0)
    
    D = tl.sum(O_0 * dO_0 + O_1 * dO_1, axis=1)
    D = D[:, None]
    D = tl.where((row_idx + i_start < S)[:, None], D, 0.0)
    
    L_i = tl.load(L_ptr + bh * S + i_start + row_idx, mask=(i_start + row_idx < S), other=float('inf'))
    
    dQ_0 = tl.zeros((BLOCK_Q, 64), tl.bfloat16)
    dQ_1 = tl.zeros((BLOCK_Q, 64), tl.bfloat16)
    
    num_KV_blocks = tl.cdiv(S, BLOCK_Q)
    for j in range(num_KV_blocks):
        j_start = j * BLOCK_Q
        
        k_row_ptr = K_ptr + bh * S * 128 + j_start * 128
        K_0 = tl.load(k_row_ptr + row_idx[:, None] * 128 + cols_0[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_0 < 128)[None, :], other=0.0)
        K_1 = tl.load(k_row_ptr + row_idx[:, None] * 128 + cols_1[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_1 < 128)[None, :], other=0.0)
        
        v_row_ptr = V_ptr + bh * S * 128 + j_start * 128
        V_0 = tl.load(v_row_ptr + row_idx[:, None] * 128 + cols_0[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_0 < 128)[None, :], other=0.0)
        V_1 = tl.load(v_row_ptr + row_idx[:, None] * 128 + cols_1[None, :], mask=(row_idx + j_start < S)[:, None] & (cols_1 < 128)[None, :], other=0.0)
        
        S_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        S_acc = tl.dot(Q_0, K_0.T, S_acc)
        S_acc = tl.dot(Q_1, K_1.T, S_acc)
        
        dP_acc = tl.zeros((BLOCK_Q, BLOCK_Q), tl.float32)
        dP_acc = tl.dot(dO_0, V_0.T, dP_acc)
        dP_acc = tl.dot(dO_1, V_1.T, dP_acc)
        
        P = tl.exp(S_acc * SCALE - L_i[:, None])
        
        dS = P * (dP_acc - D) * SCALE
        
        dQ_0 = dQ_0 + tl.dot(dS, K_0, out_dtype=tl.bfloat16)
        dQ_1 = dQ_1 + tl.dot(dS, K_1, out_dtype=tl.bfloat16)
        
    ptr_q0 = dQ_ptr + bh * S * 128 + i_start * 128 + cols_0
    tl.store(ptr_q0, dQ_0, mask=(row_idx + i_start < S)[:, None] & (cols_0 < 128)[None, :])
    
    ptr_q1 = dQ_ptr + bh * S * 128 + i_start * 128 + cols_1
    tl.store(ptr_q1, dQ_1, mask=(row_idx + i_start < S)[:, None] & (cols_1 < 128)[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward pass for Multi-Head Attention."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(d)
    
    Q_flat = Q.view(-1, S, d)
    K_flat = K.view(-1, S, d)
    V_flat = V.view(-1, S, d)
    O_flat = O.view(-1, S, d)
    dO_flat = dO.view(-1, S, d)
    dQ_flat = dQ.view(-1, S, d)
    dK_flat = dK.view(-1, S, d)
    dV_flat = dV.view(-1, S, d)
    
    L_ptr = L.view(-1, S).data_ptr()
    
    num_blocks = triton.cdiv(S, BLOCK_Q)
    grid = (num_blocks, B * H)
    
    dKdV_kernel[grid](
        Q_flat.data_ptr(), K_flat.data_ptr(), V_flat.data_ptr(), O_flat.data_ptr(), 
        dO_flat.data_ptr(), L_ptr,
        dK_flat.data_ptr(), dV_flat.data_ptr(),
        S,
        SCALE=scale,
        num_warps=4, num_stages=2
    )
    
    dQ_kernel[grid](
        Q_flat.data_ptr(), K_flat.data_ptr(), V_flat.data_ptr(), O_flat.data_ptr(), 
        dO_flat.data_ptr(), L_ptr,
        dQ_flat.data_ptr(),
        S,
        SCALE=scale,
        num_warps=4, num_stages=2
    )