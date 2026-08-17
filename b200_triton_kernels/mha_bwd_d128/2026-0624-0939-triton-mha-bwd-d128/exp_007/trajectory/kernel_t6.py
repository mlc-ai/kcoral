import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _preprocess_kernel(
    dO_ptr, O_ptr, D_ptr,
    S, d,
    BLOCK: tl.constexpr
):
    b_h = tl.program_id(1)
    s_offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = s_offs < S
    
    k = tl.arange(0, 64)
    
    dO_left = tl.load(dO_ptr + b_h * S * d + s_offs[:, None] * d + k[None, :], mask=mask[:, None], other=0.0)
    dO_right = tl.load(dO_ptr + b_h * S * d + s_offs[:, None] * d + 64 + k[None, :], mask=mask[:, None], other=0.0)
    O_left = tl.load(O_ptr + b_h * S * d + s_offs[:, None] * d + k[None, :], mask=mask[:, None], other=0.0)
    O_right = tl.load(O_ptr + b_h * S * d + s_offs[:, None] * d + 64 + k[None, :], mask=mask[:, None], other=0.0)
    
    D = tl.sum(dO_left * O_left + dO_right * O_right, axis=1)
    
    tl.store(D_ptr + b_h * S + s_offs, D, mask=mask)


@triton.jit
def _bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, D_ptr, L_ptr, dK_ptr, dV_ptr,
    S, tau, d,
    NUM_STAGES: tl.constexpr,
):
    num_blocks_s = tl.cdiv(S, 64)
    j = tl.program_id(0)
    b_h = tl.program_id(1)
    
    row_base_k = b_h * S * d + j * 64 * d
    K_j_left = tl.load(K_ptr + row_base_k + tl.arange(0, 64)[:, None] * d + tl.arange(0, 64)[None, :], mask=(j*64+tl.arange(0,64))[:, None]<S, other=0.0)
    K_j_right = tl.load(K_ptr + row_base_k + tl.arange(0, 64)[:, None] * d + 64 + tl.arange(0, 64)[None, :], mask=(j*64+tl.arange(0,64))[:, None]<S, other=0.0)
    V_j_left = tl.load(V_ptr + row_base_k + tl.arange(0, 64)[:, None] * d + tl.arange(0, 64)[None, :], mask=(j*64+tl.arange(0,64))[:, None]<S, other=0.0)
    V_j_right = tl.load(V_ptr + row_base_k + tl.arange(0, 64)[:, None] * d + 64 + tl.arange(0, 64)[None, :], mask=(j*64+tl.arange(0,64))[:, None]<S, other=0.0)
    
    acc_dK_left = tl.zeros((64, 64), tl.float32)
    acc_dK_right = tl.zeros((64, 64), tl.float32)
    acc_dV_left = tl.zeros((64, 64), tl.float32)
    acc_dV_right = tl.zeros((64, 64), tl.float32)
    
    k_mask = (j * 64 + tl.arange(0, 64)) < S
    
    for i in tl.range(num_blocks_s, num_stages=NUM_STAGES):
        row_base_q = b_h * S * d + i * 64 * d
        Q_i_left = tl.load(Q_ptr + row_base_q + tl.arange(0, 64)[:, None] * d + tl.arange(0, 64)[None, :], mask=(i*64+tl.arange(0,64))[:, None]<S, other=0.0)
        Q_i_right = tl.load(Q_ptr + row_base_q + tl.arange(0, 64)[:, None] * d + 64 + tl.arange(0, 64)[None, :], mask=(i*64+tl.arange(0,64))[:, None]<S, other=0.0)
        dO_i_left = tl.load(dO_ptr + row_base_q + tl.arange(0, 64)[:, None] * d + tl.arange(0, 64)[None, :], mask=(i*64+tl.arange(0,64))[:, None]<S, other=0.0)
        dO_i_right = tl.load(dO_ptr + row_base_q + tl.arange(0, 64)[:, None] * d + 64 + tl.arange(0, 64)[None, :], mask=(i*64+tl.arange(0,64))[:, None]<S, other=0.0)
        
        q_mask = (i * 64 + tl.arange(0, 64)) < S
        mask_2d = (q_mask[:, None] & k_mask[None, :]).to(tl.float32)
        
        D_i = tl.load(D_ptr + b_h * S + i * 64 + tl.arange(0, 64), mask=q_mask, other=0.0)
        L_i = tl.load(L_ptr + b_h * S + i * 64 + tl.arange(0, 64), mask=q_mask, other=0.0)
        
        S_acc = tl.zeros((64, 64), tl.float32)
        S_acc = tl.dot(Q_i_left, K_j_left.T, S_acc)
        S_acc = tl.dot(Q_i_right, K_j_right.T, S_acc)
        S_scores = S_acc * tau
        
        P = tl.exp(S_scores - L_i[:, None])
        P = P * mask_2d
        
        dP_acc = tl.zeros((64, 64), tl.float32)
        dP_acc = tl.dot(dO_i_left, V_j_left.T, dP_acc)
        dP_acc = tl.dot(dO_i_right, V_j_right.T, dP_acc)
        
        dS = P * (dP_acc - D_i[:, None]) * tau
        dS = dS * mask_2d
        
        acc_dV_left = tl.dot(P.T, dO_i_left, acc_dV_left)
        acc_dV_right = tl.dot(P.T, dO_i_right, acc_dV_right)
        
        acc_dK_left = tl.dot(dS.T, Q_i_left, acc_dK_left)
        acc_dK_right = tl.dot(dS.T, Q_i_right, acc_dK_right)
        
    store_base_k = dK_ptr + b_h * S * d + j * 64 * d
    dK_left_ptrs = store_base_k + tl.arange(0, 64)[:, None] * d + tl.arange(0, 64)[None, :]
    dK_right_ptrs = store_base_k + tl.arange(0, 64)[:, None] * d + 64 + tl.arange(0, 64)[None, :]
    valid_row_k = (j * 64 + tl.arange(0, 64)[:, None]) < S
    tl.store(dK_left_ptrs, acc_dK_left, mask=valid_row_k)
    tl.store(dK_right_ptrs, acc_dK_right, mask=valid_row_k)
    
    store_base_k_v = dV_ptr + b_h * S * d + j * 64 * d
    dV_left_ptrs = store_base_k_v + tl.arange(0, 64)[:, None] * d + tl.arange(0, 64)[None, :]
    dV_right_ptrs = store_base_k_v + tl.arange(0, 64)[:, None] * d + 64 + tl.arange(0, 64)[None, :]
    tl.store(dV_left_ptrs, acc_dV_left, mask=valid_row_k)
    tl.store(dV_right_ptrs, acc_dV_right, mask=valid_row_k)


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, D_ptr, L_ptr, dQ_ptr,
    S, tau, d,
    NUM_STAGES: tl.constexpr,
):
    num_blocks_s = tl.cdiv(S, 64)
    i = tl.program_id(0)
    b_h = tl.program_id(1)
    
    row_base_q = b_h * S * d + i * 64 * d
    Q_i_left = tl.load(Q_ptr + row_base_q + tl.arange(0, 64)[:, None] * d + tl.arange(0, 64)[None, :], mask=(i*64+tl.arange(0,64))[:, None]<S, other=0.0)
    Q_i_right = tl.load(Q_ptr + row_base_q + tl.arange(0, 64)[:, None] * d + 64 + tl.arange(0, 64)[None, :], mask=(i*64+tl.arange(0,64))[:, None]<S, other=0.0)
    dO_i_left = tl.load(dO_ptr + row_base_q + tl.arange(0, 64)[:, None] * d + tl.arange(0, 64)[None, :], mask=(i*64+tl.arange(0,64))[:, None]<S, other=0.0)
    dO_i_right = tl.load(dO_ptr + row_base_q + tl.arange(0, 64)[:, None] * d + 64 + tl.arange(0, 64)[None, :], mask=(i*64+tl.arange(0,64))[:, None]<S, other=0.0)
    
    q_mask = (i * 64 + tl.arange(0, 64)) < S
    
    D_i = tl.load(D_ptr + b_h * S + i * 64 + tl.arange(0, 64), mask=q_mask, other=0.0)
    L_i = tl.load(L_ptr + b_h * S + i * 64 + tl.arange(0, 64), mask=q_mask, other=0.0)
    
    acc_dQ_left = tl.zeros((64, 64), tl.float32)
    acc_dQ_right = tl.zeros((64, 64), tl.float32)
    
    for j in tl.range(num_blocks_s, num_stages=NUM_STAGES):
        row_base_k = b_h * S * d + j * 64 * d
        K_j_left = tl.load(K_ptr + row_base_k + tl.arange(0, 64)[:, None] * d + tl.arange(0, 64)[None, :], mask=(j*64+tl.arange(0,64))[:, None]<S, other=0.0)
        K_j_right = tl.load(K_ptr + row_base_k + tl.arange(0, 64)[:, None] * d + 64 + tl.arange(0, 64)[None, :], mask=(j*64+tl.arange(0,64))[:, None]<S, other=0.0)
        V_j_left = tl.load(V_ptr + row_base_k + tl.arange(0, 64)[:, None] * d + tl.arange(0, 64)[None, :], mask=(j*64+tl.arange(0,64))[:, None]<S, other=0.0)
        V_j_right = tl.load(V_ptr + row_base_k + tl.arange(0, 64)[:, None] * d + 64 + tl.arange(0, 64)[None, :], mask=(j*64+tl.arange(0,64))[:, None]<S, other=0.0)
        
        k_mask = (j * 64 + tl.arange(0, 64)) < S
        mask_2d = (q_mask[:, None] & k_mask[None, :]).to(tl.float32)
        
        S_acc = tl.zeros((64, 64), tl.float32)
        S_acc = tl.dot(Q_i_left, K_j_left.T, S_acc)
        S_acc = tl.dot(Q_i_right, K_j_right.T, S_acc)
        S_scores = S_acc * tau
        
        P = tl.exp(S_scores - L_i[:, None])
        P = P * mask_2d
        
        dP_acc = tl.zeros((64, 64), tl.float32)
        dP_acc = tl.dot(dO_i_left, V_j_left.T, dP_acc)
        dP_acc = tl.dot(dO_i_right, V_j_right.T, dP_acc)
        
        dS = P * (dP_acc - D_i[:, None]) * tau
        dS = dS * mask_2d
        
        acc_dQ_left = tl.dot(dS, K_j_left, acc_dQ_left)
        acc_dQ_right = tl.dot(dS, K_j_right, acc_dQ_right)
        
    store_base_q = dQ_ptr + b_h * S * d + i * 64 * d
    dQ_left_ptrs = store_base_q + tl.arange(0, 64)[:, None] * d + tl.arange(0, 64)[None, :]
    dQ_right_ptrs = store_base_q + tl.arange(0, 64)[:, None] * d + 64 + tl.arange(0, 64)[None, :]
    valid_row_q = (i * 64 + tl.arange(0, 64)[:, None]) < S
    tl.store(dQ_left_ptrs, acc_dQ_left, mask=valid_row_q)
    tl.store(dQ_right_ptrs, acc_dQ_right, mask=valid_row_q)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    if S == 0:
        return

    D = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    tau = 1.0 / math.sqrt(d)
    
    grid_preprocess = (triton.cdiv(S, 64), B * H)
    _preprocess_kernel[grid_preprocess](dO, O, D, S, d, BLOCK=64)
    
    num_blocks_s = triton.cdiv(S, 64)
    
    grid_dkv = (num_blocks_s, B * H)
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, dO, D, L, dK, dV,
        S, tau, d,
        NUM_STAGES=3,
        num_warps=8
    )
    
    grid_dq = (num_blocks_s, B * H)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, dO, D, L, dQ,
        S, tau, d,
        NUM_STAGES=3,
        num_warps=8
    )