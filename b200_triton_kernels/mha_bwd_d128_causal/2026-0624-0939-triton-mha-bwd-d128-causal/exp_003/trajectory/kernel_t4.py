import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_pass1(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr,
    dQ_desc, S_len, scale, H_val,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    i = tl.program_id(0)
    
    batch_h = b * H_val + h
    
    Q_full = tl.reshape(Q_desc.load([b, h, i * 64, 0]), (64, 128))
    O_full = tl.reshape(O_desc.load([b, h, i * 64, 0]), (64, 128))
    dO_full = tl.reshape(dO_desc.load([b, h, i * 64, 0]), (64, 128))
    
    q_pos = i * 64 + tl.arange(0, 64)
    q_pos_mask = q_pos < S_len
    
    D = tl.sum(dO_full * O_full, axis=1)
    D = tl.where(q_pos_mask, D, 0.0)
    
    l_off = batch_h * S_len + i * 64 + tl.arange(0, 64)
    L_tile = tl.load(L_ptr + l_off, mask=q_pos_mask, other=0.0)
    
    dQ_acc = tl.zeros((64, 128), tl.float32)
    
    for j in range(i + 1):
        K_full = tl.reshape(K_desc.load([b, h, j * 64, 0]), (64, 128))
        V_full = tl.reshape(V_desc.load([b, h, j * 64, 0]), (64, 128))
        
        k_pos = j * 64 + tl.arange(0, 64)
        
        S_matrix = tl.dot(Q_full, K_full.T) * scale
        
        valid_mask = (q_pos[:, None] >= k_pos[None, :]) & (q_pos < S_len)[:, None] & (k_pos < S_len)[None, :]
        S_matrix = tl.where(valid_mask, S_matrix, float("-inf"))
        
        P = tl.exp(S_matrix - L_tile[:, None])
        
        dP = tl.dot(dO_full, V_full.T)
        
        dS = P * (dP - D[:, None]) * scale
        
        dQ_acc = tl.dot(dS, K_full, dQ_acc)
    
    dQ_desc.store([b, h, i * 64, 0], tl.reshape(dQ_acc.to(tl.bfloat16), (1, 1, 64, 128)))


@triton.jit
def _bwd_pass2(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr,
    dK_desc, dV_desc, S_len, scale, H_val,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    j = tl.program_id(0)
    
    batch_h = b * H_val + h
    
    K_full = tl.reshape(K_desc.load([b, h, j * 64, 0]), (64, 128))
    V_full = tl.reshape(V_desc.load([b, h, j * 64, 0]), (64, 128))
    
    dK_acc = tl.zeros((64, 128), tl.float32)
    dV_acc = tl.zeros((64, 128), tl.float32)
    
    num_blocks = tl.cdiv(S_len, 64)
    k_pos = j * 64 + tl.arange(0, 64)
    
    for i in range(j, num_blocks):
        Q_full_ = tl.reshape(Q_desc.load([b, h, i * 64, 0]), (64, 128))
        O_full_ = tl.reshape(O_desc.load([b, h, i * 64, 0]), (64, 128))
        dO_full_ = tl.reshape(dO_desc.load([b, h, i * 64, 0]), (64, 128))
        
        q_pos = i * 64 + tl.arange(0, 64)
        q_pos_mask = q_pos < S_len
        
        l_off = batch_h * S_len + i * 64 + tl.arange(0, 64)
        L_tile = tl.load(L_ptr + l_off, mask=q_pos_mask, other=0.0)
        
        D_ = tl.sum(dO_full_ * O_full_, axis=1)
        D_ = tl.where(q_pos_mask, D_, 0.0)
        
        S_matrix = tl.dot(Q_full_, K_full.T) * scale
        
        valid_mask = (q_pos[:, None] >= k_pos[None, :]) & (q_pos < S_len)[:, None] & (k_pos < S_len)[None, :]
        S_matrix = tl.where(valid_mask, S_matrix, float("-inf"))
        
        P_ = tl.exp(S_matrix - L_tile[:, None])
        
        dP_ = tl.dot(dO_full_, V_full.T)
        
        dS_ = P_ * (dP_ - D_[:, None]) * scale
        
        dK_acc = tl.dot(dS_.T, Q_full_, dK_acc)
        dV_acc = tl.dot(P_.T, dO_full_, dV_acc)
    
    dK_desc.store([b, h, j * 64, 0], tl.reshape(dK_acc.to(tl.bfloat16), (1, 1, 64, 128)))
    dV_desc.store([b, h, j * 64, 0], tl.reshape(dV_acc.to(tl.bfloat16), (1, 1, 64, 128)))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B_val, H_val, S_len, d_dim = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 64, 128])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, 64, 128])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, 64, 128])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, 64, 128])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, 64, 128])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 1, 64, 128])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 1, 64, 128])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 1, 64, 128])
    
    scale = 1.0 / math.sqrt(d_dim)
    num_blocks = triton.cdiv(S_len, 64)
    grid = (num_blocks, H_val, B_val)
    
    _bwd_pass1[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L,
        dQ_desc, S_len, scale, H_val,
        num_warps=8,
        num_stages=2,
    )
    
    _bwd_pass2[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L,
        dK_desc, dV_desc, S_len, scale, H_val,
        num_warps=8,
        num_stages=2,
    )