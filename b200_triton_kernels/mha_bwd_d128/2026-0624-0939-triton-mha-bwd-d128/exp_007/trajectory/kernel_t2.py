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
    Q_desc, K_desc, V_desc, dO_desc, D_ptr, L_ptr, dK_desc, dV_desc,
    S, tau,
    NUM_STAGES: tl.constexpr,
):
    b_h = tl.program_id(1)
    j = tl.program_id(0)
    num_blocks_s = tl.cdiv(S, 64)
    
    K_j_left = K_desc.load([b_h, j * 64, 0])
    K_j_right = K_desc.load([b_h, j * 64, 64])
    V_j_left = V_desc.load([b_h, j * 64, 0])
    V_j_right = V_desc.load([b_h, j * 64, 64])
    
    acc_dK_left = tl.zeros((64, 64), tl.float32)
    acc_dK_right = tl.zeros((64, 64), tl.float32)
    acc_dV_left = tl.zeros((64, 64), tl.float32)
    acc_dV_right = tl.zeros((64, 64), tl.float32)
    
    k_mask = (j * 64 + tl.arange(0, 64)) < S
    
    for i in tl.range(num_blocks_s, num_stages=NUM_STAGES):
        Q_i_left = Q_desc.load([b_h, i * 64, 0])
        Q_i_right = Q_desc.load([b_h, i * 64, 64])
        dO_i_left = dO_desc.load([b_h, i * 64, 0])
        dO_i_right = dO_desc.load([b_h, i * 64, 64])
        
        q_mask = (i * 64 + tl.arange(0, 64)) < S
        mask_2d = (q_mask[:, None] & k_mask[None, :])
        
        D_i = tl.load(D_ptr + b_h * S + i * 64 + tl.arange(0, 64), mask=q_mask, other=0.0)
        L_i = tl.load(L_ptr + b_h * S + i * 64 + tl.arange(0, 64), mask=q_mask, other=0.0)
        
        S_acc = tl.zeros((64, 64), tl.float32)
        S_acc = tl.dot(Q_i_left, K_j_left.T, S_acc)
        S_acc = tl.dot(Q_i_right, K_j_right.T, S_acc)
        S_scores = S_acc * tau
        
        P = tl.exp(S_scores - L_i[:, None])
        P = P * mask_2d.to(tl.float32)
        
        dP_acc = tl.zeros((64, 64), tl.float32)
        dP_acc = tl.dot(dO_i_left, V_j_left.T, dP_acc)
        dP_acc = tl.dot(dO_i_right, V_j_right.T, dP_acc)
        
        dS = P * (dP_acc - D_i[:, None]) * tau
        dS = dS * mask_2d.to(tl.float32)
        
        acc_dV_left = tl.dot(P.T, dO_i_left, acc_dV_left)
        acc_dV_right = tl.dot(P.T, dO_i_right, acc_dV_right)
        
        acc_dK_left = tl.dot(dS.T, Q_i_left, acc_dK_left)
        acc_dK_right = tl.dot(dS.T, Q_i_right, acc_dK_right)
        
    dK_desc.store([b_h, j * 64, 0], acc_dK_left.to(tl.bfloat16))
    dK_desc.store([b_h, j * 64, 64], acc_dK_right.to(tl.bfloat16))
    dV_desc.store([b_h, j * 64, 0], acc_dV_left.to(tl.bfloat16))
    dV_desc.store([b_h, j * 64, 64], acc_dV_right.to(tl.bfloat16))


@triton.jit
def _bwd_dq_kernel(
    Q_desc, K_desc, V_desc, dO_desc, D_ptr, L_ptr, dQ_desc,
    S, tau,
    NUM_STAGES: tl.constexpr,
):
    b_h = tl.program_id(1)
    i = tl.program_id(0)
    num_blocks_s = tl.cdiv(S, 64)
    
    Q_i_left = Q_desc.load([b_h, i * 64, 0])
    Q_i_right = Q_desc.load([b_h, i * 64, 64])
    dO_i_left = dO_desc.load([b_h, i * 64, 0])
    dO_i_right = dO_desc.load([b_h, i * 64, 64])
    
    q_mask = (i * 64 + tl.arange(0, 64)) < S
    
    D_i = tl.load(D_ptr + b_h * S + i * 64 + tl.arange(0, 64), mask=q_mask, other=0.0)
    L_i = tl.load(L_ptr + b_h * S + i * 64 + tl.arange(0, 64), mask=q_mask, other=0.0)
    
    acc_dQ_left = tl.zeros((64, 64), tl.float32)
    acc_dQ_right = tl.zeros((64, 64), tl.float32)
    
    for j in tl.range(num_blocks_s, num_stages=NUM_STAGES):
        K_j_left = K_desc.load([b_h, j * 64, 0])
        K_j_right = K_desc.load([b_h, j * 64, 64])
        V_j_left = V_desc.load([b_h, j * 64, 0])
        V_j_right = V_desc.load([b_h, j * 64, 64])
        
        k_mask = (j * 64 + tl.arange(0, 64)) < S
        mask_2d = (q_mask[:, None] & k_mask[None, :])
        
        S_acc = tl.zeros((64, 64), tl.float32)
        S_acc = tl.dot(Q_i_left, K_j_left.T, S_acc)
        S_acc = tl.dot(Q_i_right, K_j_right.T, S_acc)
        S_scores = S_acc * tau
        
        P = tl.exp(S_scores - L_i[:, None])
        P = P * mask_2d.to(tl.float32)
        
        dP_acc = tl.zeros((64, 64), tl.float32)
        dP_acc = tl.dot(dO_i_left, V_j_left.T, dP_acc)
        dP_acc = tl.dot(dO_i_right, V_j_right.T, dP_acc)
        
        dS = P * (dP_acc - D_i[:, None]) * tau
        dS = dS * mask_2d.to(tl.float32)
        
        acc_dQ_left = tl.dot(dS, K_j_left, acc_dQ_left)
        acc_dQ_right = tl.dot(dS, K_j_right, acc_dQ_right)
        
    dQ_desc.store([b_h, i * 64, 0], acc_dQ_left.to(tl.bfloat16))
    dQ_desc.store([b_h, i * 64, 64], acc_dQ_right.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    D = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    Q = Q.view(B * H, S, d)
    K = K.view(B * H, S, d)
    V = V.view(B * H, S, d)
    O = O.view(B * H, S, d)
    dO = dO.view(B * H, S, d)
    dQ = dQ.view(B * H, S, d)
    dK = dK.view(B * H, S, d)
    dV = dV.view(B * H, S, d)
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 64, 64])
    K_desc = TensorDescriptor.from_tensor(K, [1, 64, 64])
    V_desc = TensorDescriptor.from_tensor(V, [1, 64, 64])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 64, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 64, 64])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 64, 64])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 64, 64])
    
    tau = 1.0 / math.sqrt(d)
    
    grid_preprocess = (triton.cdiv(S, 256), B * H)
    _preprocess_kernel[grid_preprocess](dO, O, D, S, d, BLOCK=256)
    
    num_blocks_s = triton.cdiv(S, 64)
    
    grid_dkv = (num_blocks_s, B * H)
    _bwd_dkv_kernel[grid_dkv](
        Q_desc, K_desc, V_desc, dO_desc, D, L, dK_desc, dV_desc,
        S, tau,
        NUM_STAGES=3,
        num_warps=8
    )
    
    grid_dq = (num_blocks_s, B * H)
    _bwd_dq_kernel[grid_dq](
        Q_desc, K_desc, V_desc, dO_desc, D, L, dQ_desc,
        S, tau,
        NUM_STAGES=3,
        num_warps=8
    )