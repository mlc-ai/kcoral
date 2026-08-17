import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_pass1(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, scale, H_val,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    i = tl.program_id(0)
    
    row_offsets = tl.arange(0, 64)
    col_offsets = tl.arange(0, 64)
    
    q_base = (b * H_val + h) * S_len * 128 + i * 64 * 128
    Q0_ptr = Q_ptr + q_base + row_offsets[:, None] * 128 + col_offsets[None, :]
    Q1_ptr = Q_ptr + q_base + row_offsets[:, None] * 128 + (col_offsets[None, :] + 64)
    
    O0_ptr = O_ptr + q_base + row_offsets[:, None] * 128 + col_offsets[None, :]
    O1_ptr = O_ptr + q_base + row_offsets[:, None] * 128 + (col_offsets[None, :] + 64)
    
    dO0_ptr = dO_ptr + q_base + row_offsets[:, None] * 128 + col_offsets[None, :]
    dO1_ptr = dO_ptr + q_base + row_offsets[:, None] * 128 + (col_offsets[None, :] + 64)
    
    q_pos = i * 64 + row_offsets
    q_pos_mask = q_pos < S_len
    
    Q0 = tl.load(Q0_ptr, mask=q_pos_mask[:, None], other=0.0)
    Q1 = tl.load(Q1_ptr, mask=q_pos_mask[:, None], other=0.0)
    O0 = tl.load(O0_ptr, mask=q_pos_mask[:, None], other=0.0)
    O1 = tl.load(O1_ptr, mask=q_pos_mask[:, None], other=0.0)
    dO0 = tl.load(dO0_ptr, mask=q_pos_mask[:, None], other=0.0)
    dO1 = tl.load(dO1_ptr, mask=q_pos_mask[:, None], other=0.0)
    
    D = tl.sum(dO0 * O0 + dO1 * O1, axis=1)
    D = tl.where(q_pos_mask, D, 0.0)
    
    batch_offset = b * H_val + h
    l_off = batch_offset * S_len + i * 64 + row_offsets
    L_tile = tl.load(L_ptr + l_off, mask=q_pos_mask, other=0.0)
    
    dQ_acc0 = tl.zeros((64, 64), tl.float32)
    dQ_acc1 = tl.zeros((64, 64), tl.float32)
    
    for j in range(i + 1):
        k_base = (b * H_val + h) * S_len * 128 + j * 64 * 128
        K0_ptr = K_ptr + k_base + row_offsets[:, None] * 128 + col_offsets[None, :]
        K1_ptr = K_ptr + k_base + row_offsets[:, None] * 128 + (col_offsets[None, :] + 64)
        V0_ptr = V_ptr + k_base + row_offsets[:, None] * 128 + col_offsets[None, :]
        V1_ptr = V_ptr + k_base + row_offsets[:, None] * 128 + (col_offsets[None, :] + 64)
        
        k_pos = j * 64 + row_offsets
        k_pos_mask = k_pos < S_len
        
        K0 = tl.load(K0_ptr, mask=k_pos_mask[:, None], other=0.0)
        K1 = tl.load(K1_ptr, mask=k_pos_mask[:, None], other=0.0)
        V0 = tl.load(V0_ptr, mask=k_pos_mask[:, None], other=0.0)
        V1 = tl.load(V1_ptr, mask=k_pos_mask[:, None], other=0.0)
        
        S_matrix = (tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)) * scale
        
        valid_mask = (q_pos[:, None] >= k_pos[None, :]) & (q_pos < S_len)[:, None] & (k_pos < S_len)[None, :]
        
        P = tl.exp(S_matrix - L_tile[:, None])
        P = tl.where(valid_mask, P, 0.0)
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        dS = P * (dP - D[:, None]) * scale
        dS = tl.where(valid_mask, dS, 0.0)
        
        dQ_acc0 = tl.dot(dS, K0, dQ_acc0)
        dQ_acc1 = tl.dot(dS, K1, dQ_acc1)
    
    dQ_acc0_bf16 = dQ_acc0.to(tl.bfloat16)
    dQ_acc1_bf16 = dQ_acc1.to(tl.bfloat16)
    
    dQ0_ptr = dQ_ptr + q_base + row_offsets[:, None] * 128 + col_offsets[None, :]
    dQ1_ptr = dQ_ptr + q_base + row_offsets[:, None] * 128 + (col_offsets[None, :] + 64)
    
    tl.store(dQ0_ptr, dQ_acc0_bf16)
    tl.store(dQ1_ptr, dQ_acc1_bf16)


@triton.jit
def _bwd_pass2(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, scale, H_val,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    j = tl.program_id(0)
    
    row_offsets = tl.arange(0, 64)
    col_offsets = tl.arange(0, 64)
    
    k_base = (b * H_val + h) * S_len * 128 + j * 64 * 128
    K0_ptr = K_ptr + k_base + row_offsets[:, None] * 128 + col_offsets[None, :]
    K1_ptr = K_ptr + k_base + row_offsets[:, None] * 128 + (col_offsets[None, :] + 64)
    V0_ptr = V_ptr + k_base + row_offsets[:, None] * 128 + col_offsets[None, :]
    V1_ptr = V_ptr + k_base + row_offsets[:, None] * 128 + (col_offsets[None, :] + 64)
    
    K0 = tl.load(K0_ptr)
    K1 = tl.load(K1_ptr)
    V0 = tl.load(V0_ptr)
    V1 = tl.load(V1_ptr)
    
    dK_acc0 = tl.zeros((64, 64), tl.float32)
    dK_acc1 = tl.zeros((64, 64), tl.float32)
    dV_acc0 = tl.zeros((64, 64), tl.float32)
    dV_acc1 = tl.zeros((64, 64), tl.float32)
    
    num_blocks = tl.cdiv(S_len, 64)
    k_pos = j * 64 + row_offsets
    
    for i in range(j, num_blocks):
        q_base = (b * H_val + h) * S_len * 128 + i * 64 * 128
        Q0_ptr = Q_ptr + q_base + row_offsets[:, None] * 128 + col_offsets[None, :]
        Q1_ptr = Q_ptr + q_base + row_offsets[:, None] * 128 + (col_offsets[None, :] + 64)
        O0_ptr = O_ptr + q_base + row_offsets[:, None] * 128 + col_offsets[None, :]
        O1_ptr = O_ptr + q_base + row_offsets[:, None] * 128 + (col_offsets[None, :] + 64)
        dO0_ptr = dO_ptr + q_base + row_offsets[:, None] * 128 + col_offsets[None, :]
        dO1_ptr = dO_ptr + q_base + row_offsets[:, None] * 128 + (col_offsets[None, :] + 64)
        
        q_pos = i * 64 + row_offsets
        q_pos_mask = q_pos < S_len
        
        Q0_ = tl.load(Q0_ptr, mask=q_pos_mask[:, None], other=0.0)
        Q1_ = tl.load(Q1_ptr, mask=q_pos_mask[:, None], other=0.0)
        O0_ = tl.load(O0_ptr, mask=q_pos_mask[:, None], other=0.0)
        O1_ = tl.load(O1_ptr, mask=q_pos_mask[:, None], other=0.0)
        dO0_ = tl.load(dO0_ptr, mask=q_pos_mask[:, None], other=0.0)
        dO1_ = tl.load(dO1_ptr, mask=q_pos_mask[:, None], other=0.0)
        
        batch_offset = b * H_val + h
        l_off = batch_offset * S_len + i * 64 + row_offsets
        L_tile = tl.load(L_ptr + l_off, mask=q_pos_mask, other=0.0)
        
        D_ = tl.sum(dO0_ * O0_ + dO1_ * O1_, axis=1)
        D_ = tl.where(q_pos_mask, D_, 0.0)
        
        S_matrix = (tl.dot(Q0_, K0.T) + tl.dot(Q1_, K1.T)) * scale
        
        valid_mask = (q_pos[:, None] >= k_pos[None, :]) & (q_pos < S_len)[:, None] & (k_pos < S_len)[None, :]
        
        P_ = tl.exp(S_matrix - L_tile[:, None])
        P_ = tl.where(valid_mask, P_, 0.0)
        
        dP_ = tl.dot(dO0_, V0.T) + tl.dot(dO1_, V1.T)
        
        dS_ = P_ * (dP_ - D_[:, None]) * scale
        dS_ = tl.where(valid_mask, dS_, 0.0)
        
        dK_acc0 = tl.dot(dS_.T, Q0_, dK_acc0)
        dK_acc1 = tl.dot(dS_.T, Q1_, dK_acc1)
        
        dV_acc0 = tl.dot(P_.T, dO0_, dV_acc0)
        dV_acc1 = tl.dot(P_.T, dO1_, dV_acc1)
    
    dK_acc0_bf16 = dK_acc0.to(tl.bfloat16)
    dK_acc1_bf16 = dK_acc1.to(tl.bfloat16)
    dV_acc0_bf16 = dV_acc0.to(tl.bfloat16)
    dV_acc1_bf16 = dV_acc1.to(tl.bfloat16)
    
    dK0_ptr = dK_ptr + k_base + row_offsets[:, None] * 128 + col_offsets[None, :]
    dK1_ptr = dK_ptr + k_base + row_offsets[:, None] * 128 + (col_offsets[None, :] + 64)
    dV0_ptr = dV_ptr + k_base + row_offsets[:, None] * 128 + col_offsets[None, :]
    dV1_ptr = dV_ptr + k_base + row_offsets[:, None] * 128 + (col_offsets[None, :] + 64)
    
    tl.store(dK0_ptr, dK_acc0_bf16)
    tl.store(dK1_ptr, dK_acc1_bf16)
    tl.store(dV0_ptr, dV_acc0_bf16)
    tl.store(dV1_ptr, dV_acc1_bf16)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B_val, H_val, S_len, d_dim = Q.shape
    
    scale = 1.0 / math.sqrt(d_dim)
    num_blocks = triton.cdiv(S_len, 64)
    grid = (num_blocks, H_val, B_val)
    
    _bwd_pass1[grid](
        Q, K, V, O, dO, L, dQ,
        S_len, scale, H_val,
        BLOCK_N=64,
        num_warps=8,
        num_stages=2,
    )
    
    _bwd_pass2[grid](
        Q, K, V, O, dO, L, dK, dV,
        S_len, scale, H_val,
        BLOCK_N=64,
        num_warps=8,
        num_stages=2,
    )