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
    i = tl.program_id(0)
    
    # Load Q, O, dO for this query block
    Q0 = Q_desc.load([b, h, i * 64, 0])
    O0 = O_desc.load([b, h, i * 64, 0])
    dO0 = dO_desc.load([b, h, i * 64, 0])
    
    q_pos = i * 64 + tl.arange(0, 64)
    row_mask = q_pos < S_len
    
    # Precompute D = sum_d(dO * O) restricted to valid rows
    D = tl.sum(dO0 * O0, axis=1)
    D = tl.where(row_mask, D, 0.0)
    
    # Load L for this query block
    batch_offset = b * H_val + h
    l_off = batch_offset * S_len + i * 64 + tl.arange(0, 64)
    L_tile = tl.load(L_ptr + l_off, mask=row_mask, other=0.0)
    
    # Initialize dQ accumulators
    dQ_acc = tl.zeros((64, 128), tl.float32)
    
    # Iterate over all key blocks <= current query block
    for j in range(i + 1):
        K0 = K_desc.load([b, h, j * 64, 0])
        V0 = V_desc.load([b, h, j * 64, 0])
        
        k_pos = j * 64 + tl.arange(0, 64)
        
        # S = Q @ K^T * scale
        S_matrix = (tl.dot(Q0, K0.T)) * scale
        
        # Compute softmax probabilities with precise causal masking
        valid_mask = (q_pos[:, None] >= k_pos[None, :]) & (q_pos < S_len)[:, None] & (k_pos < S_len)[None, :]
        S_matrix = tl.where(valid_mask, S_matrix, float("-inf"))
        
        P = tl.exp(S_matrix - L_tile[:, None])
        
        # dP = dO @ V^T
        dP = tl.dot(dO0, V0.T)
        
        # dS = P * (dP - D) * scale
        dS = P * (dP - D[:, None]) * scale
        dS = tl.where(valid_mask, dS, 0.0)
        
        # dQ += dS @ K
        dQ_acc = tl.dot(dS, K0, dQ_acc)
    
    # Persist results back into the bfloat16 output tensor
    dQ_acc_bf16 = dQ_acc.to(tl.bfloat16)
    dQ_desc.store([b, h, i * 64, 0], dQ_acc_bf16)


@triton.jit
def _bwd_pass2(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr,
    dK_desc, dV_desc, S_len, scale, H_val,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    j = tl.program_id(0)
    
    # Load K, V for this key block
    K0 = K_desc.load([b, h, j * 64, 0])
    V0 = V_desc.load([b, h, j * 64, 0])
    
    dK_acc = tl.zeros((64, 128), tl.float32)
    dV_acc = tl.zeros((64, 128), tl.float32)
    
    num_blocks = tl.cdiv(S_len, 64)
    k_pos = j * 64 + tl.arange(0, 64)
    
    # Iterate over all query blocks >= current key block
    for i in range(j, num_blocks):
        Q0_ = Q_desc.load([b, h, i * 64, 0])
        O0_ = O_desc.load([b, h, i * 64, 0])
        dO0_ = dO_desc.load([b, h, i * 64, 0])
        
        q_pos = i * 64 + tl.arange(0, 64)
        
        batch_offset = b * H_val + h
        l_off = batch_offset * S_len + i * 64 + tl.arange(0, 64)
        L_tile = tl.load(L_ptr + l_off, mask=q_pos < S_len, other=0.0)
        
        D_ = tl.sum(dO0_ * O0_, axis=1)
        
        # S = Q @ K^T * scale
        S_matrix = (tl.dot(Q0_, K0.T)) * scale
        
        # Compute softmax probabilities with precise causal masking
        valid_mask = (q_pos[:, None] >= k_pos[None, :]) & (q_pos < S_len)[:, None] & (k_pos < S_len)[None, :]
        S_matrix = tl.where(valid_mask, S_matrix, float("-inf"))
        
        P_ = tl.exp(S_matrix - L_tile[:, None])
        
        dP_ = tl.dot(dO0_, V0.T)
        
        dS_ = P_ * (dP_ - D_[:, None]) * scale
        dS_ = tl.where(valid_mask, dS_, 0.0)
        
        dK_acc = tl.dot(dS_.T, Q0_, dK_acc)
        dV_acc = tl.dot(P_.T, dO0_, dV_acc)
    
    # Persist results back into the bfloat16 output tensors
    dK_acc_bf16 = dK_acc.to(tl.bfloat16)
    dV_acc_bf16 = dV_acc.to(tl.bfloat16)
    dK_desc.store([b, h, j * 64, 0], dK_acc_bf16)
    dV_desc.store([b, h, j * 64, 0], dV_acc_bf16)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B_val, H_val, S_len, d_dim = Q.shape
    
    # Round S up to the nearest power of two to satisfy TMA descriptor block shape constraints
    S_dim = 1 << (S_len - 1).bit_length()
    
    Q_desc = TensorDescriptor.from_tensor(Q, [B_val, H_val, S_dim, 128])
    K_desc = TensorDescriptor.from_tensor(K, [B_val, H_val, S_dim, 128])
    V_desc = TensorDescriptor.from_tensor(V, [B_val, H_val, S_dim, 128])
    O_desc = TensorDescriptor.from_tensor(O, [B_val, H_val, S_dim, 128])
    dO_desc = TensorDescriptor.from_tensor(dO, [B_val, H_val, S_dim, 128])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [B_val, H_val, S_dim, 128])
    dK_desc = TensorDescriptor.from_tensor(dK, [B_val, H_val, S_dim, 128])
    dV_desc = TensorDescriptor.from_tensor(dV, [B_val, H_val, S_dim, 128])
    
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