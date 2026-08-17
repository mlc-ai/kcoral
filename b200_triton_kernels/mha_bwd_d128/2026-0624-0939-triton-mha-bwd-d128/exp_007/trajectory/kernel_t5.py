import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _bwd_dkv_kernel(
    Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr, dK_desc, dV_desc,
    S, tau, b_h_max,
    NUM_STAGES: tl.constexpr,
):
    num_blocks_s = tl.cdiv(S, 128)
    start_pid = tl.program_id(0)
    
    if start_pid >= num_blocks_s:
        return
    
    j = start_pid
    b_h = tl.program_id(1)
    
    row_offset_k = b_h * S + j * 128
    
    K_j = K_desc.load([row_offset_k, 0])
    V_j = V_desc.load([row_offset_k, 0])
    
    acc_dK = tl.zeros((128, 128), tl.float32)
    acc_dV = tl.zeros((128, 128), tl.float32)
    
    k_mask = (j * 128 + tl.arange(0, 128)) < S
    
    for i in tl.range(num_blocks_s, num_stages=NUM_STAGES):
        row_offset_q = b_h * S + i * 128
        
        Q_i = Q_desc.load([row_offset_q, 0])
        dO_i = dO_desc.load([row_offset_q, 0])
        O_i = O_desc.load([row_offset_q, 0])
        
        q_mask = (i * 128 + tl.arange(0, 128)) < S
        mask_2d = (q_mask[:, None] & k_mask[None, :]).to(tl.float32)
        
        D_i = tl.sum(dO_i * O_i, axis=1)
        
        L_i = tl.load(L_ptr + b_h * S + i * 128 + tl.arange(0, 128), mask=q_mask, other=0.0)
        
        S_acc = tl.zeros((128, 128), tl.float32)
        S_acc = tl.dot(Q_i, K_j.T, S_acc)
        S_scores = S_acc * tau
        
        P = tl.exp(S_scores - L_i[:, None])
        P = P * mask_2d
        
        dP_acc = tl.zeros((128, 128), tl.float32)
        dP_acc = tl.dot(dO_i, V_j.T, dP_acc)
        
        dS = P * (dP_acc - D_i[:, None]) * tau
        dS = dS * mask_2d
        
        acc_dV = tl.dot(P.T, dO_i, acc_dV)
        acc_dK = tl.dot(dS.T, Q_i, acc_dK)
        
    dK_desc.store([row_offset_k, 0], acc_dK.to(tl.bfloat16))
    dV_desc.store([row_offset_k, 0], acc_dV.to(tl.bfloat16))


@triton.jit
def _bwd_dq_kernel(
    Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr, dQ_desc,
    S, tau, b_h_max,
    NUM_STAGES: tl.constexpr,
):
    num_blocks_s = tl.cdiv(S, 128)
    start_pid = tl.program_id(0)
    
    if start_pid >= num_blocks_s:
        return
    
    i = start_pid
    b_h = tl.program_id(1)
    
    row_offset_q = b_h * S + i * 128
    
    Q_i = Q_desc.load([row_offset_q, 0])
    dO_i = dO_desc.load([row_offset_q, 0])
    O_i = O_desc.load([row_offset_q, 0])
    
    q_mask = (i * 128 + tl.arange(0, 128)) < S
    
    D_i = tl.sum(dO_i * O_i, axis=1)
    
    L_i = tl.load(L_ptr + b_h * S + i * 128 + tl.arange(0, 128), mask=q_mask, other=0.0)
    
    acc_dQ = tl.zeros((128, 128), tl.float32)
    
    for j in tl.range(num_blocks_s, num_stages=NUM_STAGES):
        row_offset_k = b_h * S + j * 128
        
        K_j = K_desc.load([row_offset_k, 0])
        V_j = V_desc.load([row_offset_k, 0])
        
        k_mask = (j * 128 + tl.arange(0, 128)) < S
        mask_2d = (q_mask[:, None] & k_mask[None, :]).to(tl.float32)
        
        S_acc = tl.zeros((128, 128), tl.float32)
        S_acc = tl.dot(Q_i, K_j.T, S_acc)
        S_scores = S_acc * tau
        
        P = tl.exp(S_scores - L_i[:, None])
        P = P * mask_2d
        
        dP_acc = tl.zeros((128, 128), tl.float32)
        dP_acc = tl.dot(dO_i, V_j.T, dP_acc)
        
        dS = P * (dP_acc - D_i[:, None]) * tau
        dS = dS * mask_2d
        
        acc_dQ = tl.dot(dS, K_j, acc_dQ)
        
    dQ_desc.store([row_offset_q, 0], acc_dQ.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    if S == 0:
        return

    Q_2d = Q.reshape(B * H * S, 128)
    K_2d = K.reshape(B * H * S, 128)
    V_2d = V.reshape(B * H * S, 128)
    O_2d = O.reshape(B * H * S, 128)
    dO_2d = dO.reshape(B * H * S, 128)
    dQ_2d = dQ.reshape(B * H * S, 128)
    dK_2d = dK.reshape(B * H * S, 128)
    dV_2d = dV.reshape(B * H * S, 128)
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [128, 128])
    K_desc = TensorDescriptor.from_tensor(K_2d, [128, 128])
    V_desc = TensorDescriptor.from_tensor(V_2d, [128, 128])
    O_desc = TensorDescriptor.from_tensor(O_2d, [128, 128])
    dO_desc = TensorDescriptor.from_tensor(dO_2d, [128, 128])
    dQ_desc = TensorDescriptor.from_tensor(dQ_2d, [128, 128])
    dK_desc = TensorDescriptor.from_tensor(dK_2d, [128, 128])
    dV_desc = TensorDescriptor.from_tensor(dV_2d, [128, 128])
    
    tau = 1.0 / math.sqrt(d)
    
    num_blocks_s = triton.cdiv(S, 128)
    b_h_max = B * H
    
    grid_dkv = (num_blocks_s, b_h_max)
    _bwd_dkv_kernel[grid_dkv](
        Q_desc, K_desc, V_desc, dO_desc, O_desc, L, dK_desc, dV_desc,
        S, tau, b_h_max,
        NUM_STAGES=3,
        num_warps=8
    )
    
    grid_dq = (num_blocks_s, b_h_max)
    _bwd_dq_kernel[grid_dq](
        Q_desc, K_desc, V_desc, dO_desc, O_desc, L, dQ_desc,
        S, tau, b_h_max,
        NUM_STAGES=3,
        num_warps=8
    )