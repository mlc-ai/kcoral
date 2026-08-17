import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dKdV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, scale, num_blocks,
    STRIDE_BH, STRIDE_S, L_STRIDE_BH,
    BLOCK_N: tl.constexpr,
):
    b_h = tl.program_id(0)
    j = tl.program_id(1)
    start_i = j  # causal: only attend to existing Q blocks
    
    if start_i >= num_blocks:
        return
    
    dK_0 = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    dK_1 = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    dV_0 = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    dV_1 = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    
    row_idx = tl.arange(0, BLOCK_N)
    col_idx = tl.arange(0, BLOCK_N)
    
    k_base_0 = b_h * STRIDE_BH + j * BLOCK_N * STRIDE_S + 0
    k_offsets_0 = k_base_0 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
    k_mask_0 = (j * BLOCK_N + row_idx)[:, None] < S_len
    k_j_0 = tl.load(K_ptr + k_offsets_0, mask=k_mask_0, other=0.0)
    
    k_base_1 = b_h * STRIDE_BH + j * BLOCK_N * STRIDE_S + 64
    k_offsets_1 = k_base_1 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
    k_j_1 = tl.load(K_ptr + k_offsets_1, mask=k_mask_0, other=0.0)
    
    v_base_0 = b_h * STRIDE_BH + j * BLOCK_N * STRIDE_S + 0
    v_offsets_0 = v_base_0 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
    v_j_0 = tl.load(V_ptr + v_offsets_0, mask=k_mask_0, other=0.0)
    
    v_base_1 = b_h * STRIDE_BH + j * BLOCK_N * STRIDE_S + 64
    v_offsets_1 = v_base_1 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
    v_j_1 = tl.load(V_ptr + v_offsets_1, mask=k_mask_0, other=0.0)
    
    for i in range(start_i, num_blocks):
        q_base_0 = b_h * STRIDE_BH + i * BLOCK_N * STRIDE_S + 0
        q_offsets_0 = q_base_0 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
        q_mask_0 = (i * BLOCK_N + row_idx)[:, None] < S_len
        q_i_0 = tl.load(Q_ptr + q_offsets_0, mask=q_mask_0, other=0.0)
        
        q_base_1 = b_h * STRIDE_BH + i * BLOCK_N * STRIDE_S + 64
        q_offsets_1 = q_base_1 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
        q_i_1 = tl.load(Q_ptr + q_offsets_1, mask=q_mask_0, other=0.0)
        
        dO_base_0 = b_h * STRIDE_BH + i * BLOCK_N * STRIDE_S + 0
        dO_offsets_0 = dO_base_0 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
        dO_i_0 = tl.load(dO_ptr + dO_offsets_0, mask=q_mask_0, other=0.0)
        
        dO_base_1 = b_h * STRIDE_BH + i * BLOCK_N * STRIDE_S + 64
        dO_offsets_1 = dO_base_1 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
        dO_i_1 = tl.load(dO_ptr + dO_offsets_1, mask=q_mask_0, other=0.0)
        
        o_base_0 = b_h * STRIDE_BH + i * BLOCK_N * STRIDE_S + 0
        o_offsets_0 = o_base_0 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
        o_i_0 = tl.load(O_ptr + o_offsets_0, mask=q_mask_0, other=0.0)
        
        o_base_1 = b_h * STRIDE_BH + i * BLOCK_N * STRIDE_S + 64
        o_offsets_1 = o_base_1 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
        o_i_1 = tl.load(O_ptr + o_offsets_1, mask=q_mask_0, other=0.0)
        
        D_i = tl.sum(dO_i_0.to(tl.float32) * o_i_0.to(tl.float32) + dO_i_1.to(tl.float32) * o_i_1.to(tl.float32), axis=1)
        
        l_base = b_h * L_STRIDE_BH + i * BLOCK_N
        l_idx = tl.arange(0, BLOCK_N)
        l_offsets = l_base + l_idx
        l_mask = (i * BLOCK_N + l_idx) < S_len
        L_i = tl.load(L_ptr + l_offsets, mask=l_mask, other=0.0)
        
        S = tl.dot(q_i_0, k_j_0.T) + tl.dot(q_i_1, k_j_1.T)
        dP = tl.dot(dO_i_0, v_j_0.T) + tl.dot(dO_i_1, v_j_1.T)
        
        exp_val = tl.exp(S * scale - L_i[:, None])
        
        query_idx = i * BLOCK_N + row_idx
        key_idx = j * BLOCK_N + col_idx
        causal_mask = query_idx[:, None] >= key_idx[None, :]
        valid = causal_mask & (query_idx[:, None] < S_len) & (key_idx[None, :] < S_len)
        P = tl.where(valid, exp_val, 0.0)
        
        dS = P * (dP - D_i[:, None]) * scale
        
        P_bf16 = P.to(tl.bfloat16)
        dS_bf16 = dS.to(tl.bfloat16)
        
        dV_0 = tl.dot(P_bf16.T, dO_i_0, dV_0)
        dV_1 = tl.dot(P_bf16.T, dO_i_1, dV_1)
        dK_0 = tl.dot(dS_bf16.T, q_i_0, dK_0)
        dK_1 = tl.dot(dS_bf16.T, q_i_1, dK_1)
    
    dK_0_bf16 = dK_0.to(tl.bfloat16)
    dK_1_bf16 = dK_1.to(tl.bfloat16)
    dV_0_bf16 = dV_0.to(tl.bfloat16)
    dV_1_bf16 = dV_1.to(tl.bfloat16)
    
    store_mask = (j * BLOCK_N + row_idx)[:, None] < S_len
    tl.store(dK_ptr + k_offsets_0, dK_0_bf16, mask=store_mask)
    tl.store(dK_ptr + k_offsets_1, dK_1_bf16, mask=store_mask)
    tl.store(dV_ptr + v_offsets_0, dV_0_bf16, mask=store_mask)
    tl.store(dV_ptr + v_offsets_1, dV_1_bf16, mask=store_mask)


@triton.jit
def _bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, scale, num_blocks,
    STRIDE_BH, STRIDE_S, L_STRIDE_BH,
    BLOCK_N: tl.constexpr,
):
    b_h = tl.program_id(0)
    i = tl.program_id(1)
    start_j = 0
    end_j = i + 1  # causal: only attend to previous KV blocks
    
    if start_j >= end_j:
        return
    
    dQ_0 = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    dQ_1 = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    
    row_idx = tl.arange(0, BLOCK_N)
    col_idx = tl.arange(0, BLOCK_N)
    
    q_base_0 = b_h * STRIDE_BH + i * BLOCK_N * STRIDE_S + 0
    q_offsets_0 = q_base_0 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
    q_mask_0 = (i * BLOCK_N + row_idx)[:, None] < S_len
    q_i_0 = tl.load(Q_ptr + q_offsets_0, mask=q_mask_0, other=0.0)
    
    q_base_1 = b_h * STRIDE_BH + i * BLOCK_N * STRIDE_S + 64
    q_offsets_1 = q_base_1 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
    q_i_1 = tl.load(Q_ptr + q_offsets_1, mask=q_mask_0, other=0.0)
    
    dO_base_0 = b_h * STRIDE_BH + i * BLOCK_N * STRIDE_S + 0
    dO_offsets_0 = dO_base_0 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
    dO_i_0 = tl.load(dO_ptr + dO_offsets_0, mask=q_mask_0, other=0.0)
    
    dO_base_1 = b_h * STRIDE_BH + i * BLOCK_N * STRIDE_S + 64
    dO_offsets_1 = dO_base_1 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
    dO_i_1 = tl.load(dO_ptr + dO_offsets_1, mask=q_mask_0, other=0.0)
    
    o_base_0 = b_h * STRIDE_BH + i * BLOCK_N * STRIDE_S + 0
    o_offsets_0 = o_base_0 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
    o_i_0 = tl.load(O_ptr + o_offsets_0, mask=q_mask_0, other=0.0)
    
    o_base_1 = b_h * STRIDE_BH + i * BLOCK_N * STRIDE_S + 64
    o_offsets_1 = o_base_1 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
    o_i_1 = tl.load(O_ptr + o_offsets_1, mask=q_mask_0, other=0.0)
    
    D_i = tl.sum(dO_i_0.to(tl.float32) * o_i_0.to(tl.float32) + dO_i_1.to(tl.float32) * o_i_1.to(tl.float32), axis=1)
    
    l_base = b_h * L_STRIDE_BH + i * BLOCK_N
    l_idx = tl.arange(0, BLOCK_N)
    l_offsets = l_base + l_idx
    l_mask = (i * BLOCK_N + l_idx) < S_len
    L_i = tl.load(L_ptr + l_offsets, mask=l_mask, other=0.0)
    
    query_idx = i * BLOCK_N + row_idx
    
    for j in range(start_j, end_j):
        k_base_0 = b_h * STRIDE_BH + j * BLOCK_N * STRIDE_S + 0
        k_offsets_0 = k_base_0 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
        k_mask_0 = (j * BLOCK_N + row_idx)[:, None] < S_len
        k_j_0 = tl.load(K_ptr + k_offsets_0, mask=k_mask_0, other=0.0)
        
        k_base_1 = b_h * STRIDE_BH + j * BLOCK_N * STRIDE_S + 64
        k_offsets_1 = k_base_1 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
        k_j_1 = tl.load(K_ptr + k_offsets_1, mask=k_mask_0, other=0.0)
        
        v_base_0 = b_h * STRIDE_BH + j * BLOCK_N * STRIDE_S + 0
        v_offsets_0 = v_base_0 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
        v_j_0 = tl.load(V_ptr + v_offsets_0, mask=k_mask_0, other=0.0)
        
        v_base_1 = b_h * STRIDE_BH + j * BLOCK_N * STRIDE_S + 64
        v_offsets_1 = v_base_1 + row_idx[:, None] * STRIDE_S + col_idx[None, :]
        v_j_1 = tl.load(V_ptr + v_offsets_1, mask=k_mask_0, other=0.0)
        
        S = tl.dot(q_i_0, k_j_0.T) + tl.dot(q_i_1, k_j_1.T)
        dP = tl.dot(dO_i_0, v_j_0.T) + tl.dot(dO_i_1, v_j_1.T)
        
        exp_val = tl.exp(S * scale - L_i[:, None])
        
        key_idx = j * BLOCK_N + col_idx
        causal_mask = query_idx[:, None] >= key_idx[None, :]
        valid = causal_mask & (query_idx[:, None] < S_len) & (key_idx[None, :] < S_len)
        P = tl.where(valid, exp_val, 0.0)
        
        dS = P * (dP - D_i[:, None]) * scale
        
        dS_bf16 = dS.to(tl.bfloat16)
        
        dQ_0 = tl.dot(dS_bf16, k_j_0, dQ_0)
        dQ_1 = tl.dot(dS_bf16, k_j_1, dQ_1)
    
    dQ_0_bf16 = dQ_0.to(tl.bfloat16)
    dQ_1_bf16 = dQ_1.to(tl.bfloat16)
    
    store_mask = (i * BLOCK_N + row_idx)[:, None] < S_len
    tl.store(dQ_ptr + q_offsets_0, dQ_0_bf16, mask=store_mask)
    tl.store(dQ_ptr + q_offsets_1, dQ_1_bf16, mask=store_mask)


def run_mha_bwd(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S = Q.shape[0], Q.shape[1], Q.shape[2]
    d = Q.shape[3]
    
    num_blocks = triton.cdiv(S, 64)
    grid = (B * H, num_blocks)
    
    scale = 1.0 / math.sqrt(d)
    
    STRIDE_BH = Q.stride()[1]
    STRIDE_S = Q.stride()[2]
    L_STRIDE_BH = L.stride()[1]
    
    _bwd_dKdV_kernel[grid](
        Q, K, V, O, dO, L, dK, dV, S, scale, num_blocks,
        STRIDE_BH, STRIDE_S, L_STRIDE_BH,
        BLOCK_N=64,
        num_warps=8,
    )
    
    _bwd_dQ_kernel[grid](
        Q, K, V, O, dO, L, dQ, S, scale, num_blocks,
        STRIDE_BH, STRIDE_S, L_STRIDE_BH,
        BLOCK_N=64,
        num_warps=8,
    )