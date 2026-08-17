import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _preprocess_kernel(O_ptr, dO_ptr, D_ptr, S_len, H):
    b = tl.program_id(2)
    h = tl.program_id(1)
    s_idx = tl.program_id(0)
    
    if s_idx < S_len:
        off = ((b * H + h) * S_len + s_idx) * 128
        do_vals = tl.load(dO_ptr + off + tl.arange(0, 128))
        o_vals = tl.load(O_ptr + off + tl.arange(0, 128))
        d_val = tl.sum(do_vals.to(tl.float32) * o_vals.to(tl.float32))
        tl.store(D_ptr + (b * H + h) * S_len + s_idx, d_val)


@triton.jit
def _dkdv_kernel(
    Q_desc, K_desc, V_desc, dO_desc, L_ptr, D_ptr, dK_ptr, dV_ptr,
    S_len, H, tau,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    j = tl.program_id(0)
    
    num_blocks = tl.cdiv(S_len, BLOCK_K)
    
    K_tile = tl.reshape(K_desc.load([b_idx, h_idx, j * BLOCK_K, 0]), [BLOCK_K, BLOCK_D])
    V_tile = tl.reshape(V_desc.load([b_idx, h_idx, j * BLOCK_K, 0]), [BLOCK_K, BLOCK_D])
    
    dK_acc = tl.zeros((BLOCK_K, BLOCK_D), tl.float32)
    dV_acc = tl.zeros((BLOCK_K, BLOCK_D), tl.float32)
    
    for i in range(j, num_blocks):
        Q_tile = tl.reshape(Q_desc.load([b_idx, h_idx, i * BLOCK_Q, 0]), [BLOCK_Q, BLOCK_D])
        dO_tile = tl.reshape(dO_desc.load([b_idx, h_idx, i * BLOCK_Q, 0]), [BLOCK_Q, BLOCK_D])
        
        row_idx = tl.arange(0, BLOCK_Q)
        abs_i = i * BLOCK_Q + row_idx
        L_vec = tl.load(L_ptr + (b_idx * H + h_idx) * S_len + abs_i, mask=(abs_i < S_len), other=1e20)
        D_vec = tl.load(D_ptr + (b_idx * H + h_idx) * S_len + abs_i, mask=(abs_i < S_len), other=0.0)
        
        S = tl.dot(Q_tile, K_tile.T) * tau
        
        col_idx = tl.arange(0, BLOCK_K)
        abs_j_base = j * BLOCK_K + col_idx
        mask = (abs_j_base[None, :] <= abs_i[:, None]) & (abs_i[:, None] < S_len) & (abs_j_base[None, :] < S_len)
        
        P = tl.exp(S - L_vec[:, None])
        P = P * mask
        
        dP = tl.dot(dO_tile, V_tile.T)
        
        dS = P * (dP - D_vec[:, None]) * tau
        dS = dS * mask
        
        dV_acc = tl.dot(P.T, dO_tile, dV_acc)
        dK_acc = tl.dot(dS.T, Q_tile, dK_acc)
    
    row_idx_k = tl.arange(0, BLOCK_K)
    col_idx_d = tl.arange(0, BLOCK_D)
    out_off_k = (b_idx * H + h_idx) * S_len * 128 + j * BLOCK_K * 128
    
    dk_ptrs = dK_ptr + out_off_k + row_idx_k[:, None] * 128 + col_idx_d[None, :]
    dk_mask = (j * BLOCK_K + row_idx_k) < S_len
    tl.store(dk_ptrs, dK_acc.to(tl.bfloat16), mask=dk_mask[:, None])
    
    dv_ptrs = dV_ptr + out_off_k + row_idx_k[:, None] * 128 + col_idx_d[None, :]
    tl.store(dv_ptrs, dV_acc.to(tl.bfloat16), mask=dk_mask[:, None])


@triton.jit
def _dq_kernel(
    Q_desc, K_desc, V_desc, dO_desc, L_ptr, D_ptr, dQ_ptr,
    S_len, H, tau,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    i = tl.program_id(0)
    
    num_blocks = tl.cdiv(S_len, BLOCK_K)
    
    Q_tile = tl.reshape(Q_desc.load([b_idx, h_idx, i * BLOCK_Q, 0]), [BLOCK_Q, BLOCK_D])
    dO_tile = tl.reshape(dO_desc.load([b_idx, h_idx, i * BLOCK_Q, 0]), [BLOCK_Q, BLOCK_D])
    
    row_idx = tl.arange(0, BLOCK_Q)
    abs_i = i * BLOCK_Q + row_idx
    L_vec = tl.load(L_ptr + (b_idx * H + h_idx) * S_len + abs_i, mask=(abs_i < S_len), other=1e20)
    D_vec = tl.load(D_ptr + (b_idx * H + h_idx) * S_len + abs_i, mask=(abs_i < S_len), other=0.0)
    
    dQ_acc = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)
    
    for j in range(0, i + 1):
        K_tile = tl.reshape(K_desc.load([b_idx, h_idx, j * BLOCK_K, 0]), [BLOCK_K, BLOCK_D])
        V_tile = tl.reshape(V_desc.load([b_idx, h_idx, j * BLOCK_K, 0]), [BLOCK_K, BLOCK_D])
        
        S = tl.dot(Q_tile, K_tile.T) * tau
        
        col_idx = tl.arange(0, BLOCK_K)
        abs_j = j * BLOCK_K + col_idx
        mask = (abs_j[None, :] <= abs_i[:, None]) & (abs_i[:, None] < S_len) & (abs_j[None, :] < S_len)
        
        P = tl.exp(S - L_vec[:, None])
        P = P * mask
        
        dP = tl.dot(dO_tile, V_tile.T)
        
        dS = P * (dP - D_vec[:, None]) * tau
        dS = dS * mask
        
        dQ_acc = tl.dot(dS, K_tile, dQ_acc)
    
    row_idx_q = tl.arange(0, BLOCK_Q)
    col_idx_d = tl.arange(0, BLOCK_D)
    out_off_q = (b_idx * H + h_idx) * S_len * 128 + i * BLOCK_Q * 128
    
    dq_ptrs = dQ_ptr + out_off_q + row_idx_q[:, None] * 128 + col_idx_d[None, :]
    dq_mask = (i * BLOCK_Q + row_idx_q) < S_len
    tl.store(dq_ptrs, dQ_acc.to(tl.bfloat16), mask=dq_mask[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S_len, d = Q.shape
    tau = 1.0 / math.sqrt(d)
    
    D = torch.empty((B, H, S_len), dtype=torch.float32, device=Q.device)
    
    grid_pre = (S_len, H, B)
    _preprocess_kernel[grid_pre](O, dO, D, S_len, H)
    
    BLOCK_Q = 64
    BLOCK_K = 64
    BLOCK_D = 128
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_Q, BLOCK_D])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_K, BLOCK_D])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_K, BLOCK_D])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_Q, BLOCK_D])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_Q, BLOCK_D])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_Q, BLOCK_D])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_K, BLOCK_D])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_K, BLOCK_D])
    
    T_r = triton.cdiv(S_len, BLOCK_K)
    grid = (T_r, H, B)
    
    _dkdv_kernel[grid](
        Q_desc, K_desc, V_desc, dO_desc, L, D, dK, dV,
        S_len, H, tau,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=2
    )
    
    _dq_kernel[grid](
        Q_desc, K_desc, V_desc, dO_desc, L, D, dQ,
        S_len, H, tau,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=8, num_stages=2
    )