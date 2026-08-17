import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dK_ptr, dV_ptr,
    S, tau, d,
    NUM_STAGES: tl.constexpr,
):
    num_blocks_s = tl.cdiv(S, 128)
    j = tl.program_id(0)
    b_h = tl.program_id(1)
    
    k_row_offs = j * 128 + tl.arange(0, 128)
    k_mask = k_row_offs < S
    
    K_j = tl.load(K_ptr + b_h * S * d + k_row_offs[:, None] * d + tl.arange(0, 128)[None, :],
                  mask=k_mask[:, None], other=0.0)
    V_j = tl.load(V_ptr + b_h * S * d + k_row_offs[:, None] * d + tl.arange(0, 128)[None, :],
                  mask=k_mask[:, None], other=0.0)
    
    acc_dK = tl.zeros((128, 128), tl.float32)
    acc_dV = tl.zeros((128, 128), tl.float32)
    
    for i in tl.range(num_blocks_s, num_stages=NUM_STAGES):
        q_row_offs = i * 128 + tl.arange(0, 128)
        q_mask = q_row_offs < S
        
        Q_i = tl.load(Q_ptr + b_h * S * d + q_row_offs[:, None] * d + tl.arange(0, 128)[None, :],
                      mask=q_mask[:, None], other=0.0)
        dO_i = tl.load(dO_ptr + b_h * S * d + q_row_offs[:, None] * d + tl.arange(0, 128)[None, :],
                       mask=q_mask[:, None], other=0.0)
        O_i = tl.load(O_ptr + b_h * S * d + q_row_offs[:, None] * d + tl.arange(0, 128)[None, :],
                      mask=q_mask[:, None], other=0.0)
        
        mask_2d = (q_mask[:, None] & k_mask[None, :]).to(tl.float32)
        
        D_i = tl.sum(dO_i * O_i, axis=1)
        L_i = tl.load(L_ptr + b_h * S + q_row_offs, mask=q_mask, other=0.0)
        
        S_acc = tl.dot(Q_i, K_j.T)
        S_scores = S_acc * tau
        
        P = tl.exp(S_scores - L_i[:, None])
        P = P * mask_2d
        
        dP_acc = tl.dot(dO_i, V_j.T)
        
        dS = P * (dP_acc - D_i[:, None]) * tau
        dS = dS * mask_2d
        
        acc_dV = tl.dot(P.T, dO_i, acc_dV)
        acc_dK = tl.dot(dS.T, Q_i, acc_dK)
        
    store_k = dK_ptr + b_h * S * d + (j * 128 + tl.arange(0, 128))[:, None] * d + tl.arange(0, 128)[None, :]
    tl.store(store_k, acc_dK, mask=k_mask[:, None])
    
    store_v = dV_ptr + b_h * S * d + (j * 128 + tl.arange(0, 128))[:, None] * d + tl.arange(0, 128)[None, :]
    tl.store(store_v, acc_dV, mask=k_mask[:, None])


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, O_ptr, L_ptr, dQ_ptr,
    S, tau, d,
    NUM_STAGES: tl.constexpr,
):
    num_blocks_s = tl.cdiv(S, 128)
    i = tl.program_id(0)
    b_h = tl.program_id(1)
    
    q_row_offs = i * 128 + tl.arange(0, 128)
    q_mask = q_row_offs < S
    
    Q_i = tl.load(Q_ptr + b_h * S * d + q_row_offs[:, None] * d + tl.arange(0, 128)[None, :],
                  mask=q_mask[:, None], other=0.0)
    dO_i = tl.load(dO_ptr + b_h * S * d + q_row_offs[:, None] * d + tl.arange(0, 128)[None, :],
                   mask=q_mask[:, None], other=0.0)
    O_i = tl.load(O_ptr + b_h * S * d + q_row_offs[:, None] * d + tl.arange(0, 128)[None, :],
                  mask=q_mask[:, None], other=0.0)
    
    D_i = tl.sum(dO_i * O_i, axis=1)
    L_i = tl.load(L_ptr + b_h * S + q_row_offs, mask=q_mask, other=0.0)
    
    acc_dQ = tl.zeros((128, 128), tl.float32)
    
    for j in tl.range(num_blocks_s, num_stages=NUM_STAGES):
        k_row_offs = j * 128 + tl.arange(0, 128)
        k_mask = k_row_offs < S
        
        K_j = tl.load(K_ptr + b_h * S * d + k_row_offs[:, None] * d + tl.arange(0, 128)[None, :],
                      mask=k_mask[:, None], other=0.0)
        V_j = tl.load(V_ptr + b_h * S * d + k_row_offs[:, None] * d + tl.arange(0, 128)[None, :],
                      mask=k_mask[:, None], other=0.0)
        
        mask_2d = (q_mask[:, None] & k_mask[None, :]).to(tl.float32)
        
        S_acc = tl.dot(Q_i, K_j.T)
        S_scores = S_acc * tau
        
        P = tl.exp(S_scores - L_i[:, None])
        P = P * mask_2d
        
        dP_acc = tl.dot(dO_i, V_j.T)
        
        dS = P * (dP_acc - D_i[:, None]) * tau
        dS = dS * mask_2d
        
        acc_dQ = tl.dot(dS, K_j, acc_dQ)
        
    store_q = dQ_ptr + b_h * S * d + (i * 128 + tl.arange(0, 128))[:, None] * d + tl.arange(0, 128)[None, :]
    tl.store(store_q, acc_dQ, mask=q_mask[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    if S == 0:
        return
    
    tau = 1.0 / math.sqrt(d)
    
    num_blocks_s = triton.cdiv(S, 128)
    
    grid_dkv = (num_blocks_s, B * H)
    _bwd_dkv_kernel[grid_dkv](
        Q, K, V, dO, O, L, dK, dV,
        S, tau, d,
        NUM_STAGES=3,
        num_warps=8
    )
    
    grid_dq = (num_blocks_s, B * H)
    _bwd_dq_kernel[grid_dq](
        Q, K, V, dO, O, L, dQ,
        S, tau, d,
        NUM_STAGES=3,
        num_warps=8
    )