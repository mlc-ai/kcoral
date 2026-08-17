import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_pass1(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr,
    dQ_desc, S_len, scale, H_val,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    q_idx = tl.program_id(0)
    
    batch_offset = b * H_val + h
    
    Q0 = Q_desc.load([batch_offset * S_len + q_idx * 64, 0])
    Q1 = Q_desc.load([batch_offset * S_len + q_idx * 64, 64])
    O0 = O_desc.load([batch_offset * S_len + q_idx * 64, 0])
    O1 = O_desc.load([batch_offset * S_len + q_idx * 64, 64])
    dO0 = dO_desc.load([batch_offset * S_len + q_idx * 64, 0])
    dO1 = dO_desc.load([batch_offset * S_len + q_idx * 64, 64])
    
    q_pos = q_idx * 64 + tl.arange(0, 64)
    row_mask = q_pos < S_len
    
    D = tl.sum(dO0 * O0 + dO1 * O1, axis=1)
    D = tl.where(row_mask, D, 0.0)
    
    l_off = batch_offset * S_len + q_idx * 64 + tl.arange(0, 64)
    L_tile = tl.load(L_ptr + l_off, mask=row_mask, other=0.0)
    
    dQ_acc0 = tl.zeros((64, 64), tl.float32)
    dQ_acc1 = tl.zeros((64, 64), tl.float32)
    
    q_idx_2d = q_idx * 64 + tl.arange(0, 64)
    
    for k_idx in range(q_idx + 1):
        K0 = K_desc.load([batch_offset * S_len + k_idx * 64, 0])
        K1 = K_desc.load([batch_offset * S_len + k_idx * 64, 64])
        V0 = V_desc.load([batch_offset * S_len + k_idx * 64, 0])
        V1 = V_desc.load([batch_offset * S_len + k_idx * 64, 64])
        
        k_idx_2d = k_idx * 64 + tl.arange(0, 64)
        
        S_matrix = (tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)) * scale
        
        mask = (q_idx_2d[:, None] >= k_idx_2d[None, :]) & (q_idx_2d[:, None] < S_len) & (k_idx_2d[None, :] < S_len)
        
        P = tl.exp(S_matrix - L_tile[:, None])
        P = tl.where(mask, P, 0.0)
        
        dP = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        dS = P * (dP - D[:, None]) * scale
        
        dQ_acc0 = tl.dot(dS, K0, dQ_acc0)
        dQ_acc1 = tl.dot(dS, K1, dQ_acc1)
    
    dQ_desc.store([batch_offset * S_len + q_idx * 64, 0], dQ_acc0.to(tl.bfloat16))
    dQ_desc.store([batch_offset * S_len + q_idx * 64, 64], dQ_acc1.to(tl.bfloat16))


@triton.jit
def _bwd_pass2(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr,
    dK_desc, dV_desc, S_len, scale, H_val,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    k_idx = tl.program_id(0)
    
    batch_offset = b * H_val + h
    
    K0 = K_desc.load([batch_offset * S_len + k_idx * 64, 0])
    K1 = K_desc.load([batch_offset * S_len + k_idx * 64, 64])
    V0 = V_desc.load([batch_offset * S_len + k_idx * 64, 0])
    V1 = V_desc.load([batch_offset * S_len + k_idx * 64, 64])
    
    dK_acc0 = tl.zeros((64, 64), tl.float32)
    dK_acc1 = tl.zeros((64, 64), tl.float32)
    dV_acc0 = tl.zeros((64, 64), tl.float32)
    dV_acc1 = tl.zeros((64, 64), tl.float32)
    
    num_blocks = tl.cdiv(S_len, 64)
    
    k_pos_2d = k_idx * 64 + tl.arange(0, 64)
    
    for q_idx in range(k_idx, num_blocks):
        Q0_ = Q_desc.load([batch_offset * S_len + q_idx * 64, 0])
        Q1_ = Q_desc.load([batch_offset * S_len + q_idx * 64, 64])
        O0_ = O_desc.load([batch_offset * S_len + q_idx * 64, 0])
        O1_ = O_desc.load([batch_offset * S_len + q_idx * 64, 64])
        dO0_ = dO_desc.load([batch_offset * S_len + q_idx * 64, 0])
        dO1_ = dO_desc.load([batch_offset * S_len + q_idx * 64, 64])
        
        q_pos_2d = q_idx * 64 + tl.arange(0, 64)
        q_pos_mask = q_pos_2d < S_len
        
        l_off = batch_offset * S_len + q_idx * 64 + tl.arange(0, 64)
        L_tile = tl.load(L_ptr + l_off, mask=q_pos_mask, other=0.0)
        
        D_ = tl.sum(dO0_ * O0_ + dO1_ * O1_, axis=1)
        D_ = tl.where(q_pos_mask, D_, 0.0)
        
        if q_idx > k_idx:
            S_matrix = (tl.dot(Q0_, K0.T) + tl.dot(Q1_, K1.T)) * scale
            valid_mask = q_pos_2d[:, None] < S_len
            P_ = tl.exp(S_matrix - L_tile[:, None])
            P_ = tl.where(valid_mask, P_, 0.0)
            dP_ = tl.dot(dO0_, V0.T) + tl.dot(dO1_, V1.T)
            dS_ = P_ * (dP_ - D_[:, None]) * scale
            
            dK_acc0 = tl.dot(dS_.T, Q0_, dK_acc0)
            dK_acc1 = tl.dot(dS_.T, Q1_, dK_acc1)
            dV_acc0 = tl.dot(P_.T, dO0_, dV_acc0)
            dV_acc1 = tl.dot(P_.T, dO1_, dV_acc1)
        else:
            q_idx_2d = q_idx * 64 + tl.arange(0, 64)
            S_matrix = (tl.dot(Q0_, K0.T) + tl.dot(Q1_, K1.T)) * scale
            mask = (q_idx_2d[:, None] >= k_pos_2d[None, :]) & (q_idx_2d[:, None] < S_len) & (k_pos_2d[None, :] < S_len)
            P_ = tl.exp(S_matrix - L_tile[:, None])
            P_ = tl.where(mask, P_, 0.0)
            dP_ = tl.dot(dO0_, V0.T) + tl.dot(dO1_, V1.T)
            dS_ = P_ * (dP_ - D_[:, None]) * scale
            
            dK_acc0 = tl.dot(dS_.T, Q0_, dK_acc0)
            dK_acc1 = tl.dot(dS_.T, Q1_, dK_acc1)
            dV_acc0 = tl.dot(P_.T, dO0_, dV_acc0)
            dV_acc1 = tl.dot(P_.T, dO1_, dV_acc1)
    
    dK_desc.store([batch_offset * S_len + k_idx * 64, 0], dK_acc0.to(tl.bfloat16))
    dK_desc.store([batch_offset * S_len + k_idx * 64, 64], dK_acc1.to(tl.bfloat16))
    dV_desc.store([batch_offset * S_len + k_idx * 64, 0], dV_acc0.to(tl.bfloat16))
    dV_desc.store([batch_offset * S_len + k_idx * 64, 64], dV_acc1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B_val, H_val, S_len, d_dim = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q, [B_val * H_val * S_len, 128])
    K_desc = TensorDescriptor.from_tensor(K, [B_val * H_val * S_len, 128])
    V_desc = TensorDescriptor.from_tensor(V, [B_val * H_val * S_len, 128])
    O_desc = TensorDescriptor.from_tensor(O, [B_val * H_val * S_len, 128])
    dO_desc = TensorDescriptor.from_tensor(dO, [B_val * H_val * S_len, 128])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [B_val * H_val * S_len, 128])
    dK_desc = TensorDescriptor.from_tensor(dK, [B_val * H_val * S_len, 128])
    dV_desc = TensorDescriptor.from_tensor(dV, [B_val * H_val * S_len, 128])
    
    scale = 1.0 / math.sqrt(d_dim)
    num_blocks = triton.cdiv(S_len, 64)
    grid = (num_blocks, H_val, B_val)
    
    _bwd_pass1[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L,
        dQ_desc, S_len, scale, H_val,
        BLOCK_N=64,
        num_warps=8,
        num_stages=2,
    )
    
    _bwd_pass2[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L,
        dK_desc, dV_desc, S_len, scale, H_val,
        BLOCK_N=64,
        num_warps=8,
        num_stages=2,
    )