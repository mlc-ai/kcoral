import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _bwd_dkv_kernel(
    Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr, dK_desc, dV_desc,
    S, tau, b_h_max,
    NUM_SMS: tl.constexpr,
):
    num_blocks_s = tl.cdiv(S, 128)
    start_pid = tl.program_id(0)
    total_tiles = num_blocks_s * b_h_max
    
    if start_pid >= total_tiles:
        return
    
    for iteration in tl.range(start_pid, total_tiles, NUM_SMS):
        b_h = iteration // num_blocks_s
        j = iteration % num_blocks_s
        
        K_j_left = K_desc.load([b_h, j * 128, 0])
        K_j_right = K_desc.load([b_h, j * 128, 64])
        V_j_left = V_desc.load([b_h, j * 128, 0])
        V_j_right = V_desc.load([b_h, j * 128, 64])
        
        acc_dK_left = tl.zeros((1, 128, 64), tl.float32)
        acc_dK_right = tl.zeros((1, 128, 64), tl.float32)
        acc_dV_left = tl.zeros((1, 128, 64), tl.float32)
        acc_dV_right = tl.zeros((1, 128, 64), tl.float32)
        
        k_mask = (j * 128 + tl.arange(0, 128)) < S
        
        for i in range(num_blocks_s):
            Q_i_left = Q_desc.load([b_h, i * 128, 0])
            Q_i_right = Q_desc.load([b_h, i * 128, 64])
            dO_i_left = dO_desc.load([b_h, i * 128, 0])
            dO_i_right = dO_desc.load([b_h, i * 128, 64])
            O_i_left = O_desc.load([b_h, i * 128, 0])
            O_i_right = O_desc.load([b_h, i * 128, 64])
            
            q_mask = (i * 128 + tl.arange(0, 128)) < S
            mask_2d = (q_mask[:, None] & k_mask[None, :]).to(tl.float32)
            
            D_i = tl.sum(dO_i_left[0, :, :] * O_i_left[0, :, :] + dO_i_right[0, :, :] * O_i_right[0, :, :], axis=1)
            L_i = tl.load(L_ptr + b_h * S + i * 128 + tl.arange(0, 128), mask=q_mask, other=0.0)
            
            S_acc = tl.zeros((1, 128, 128), tl.float32)
            S_acc = tl.dot(Q_i_left, K_j_left.T, S_acc)
            S_acc = tl.dot(Q_i_right, K_j_right.T, S_acc)
            S_scores = S_acc[0, :, :] * tau
            
            P = tl.exp(S_scores - L_i[:, None])
            P = P * mask_2d
            
            dP_acc = tl.zeros((1, 128, 128), tl.float32)
            dP_acc = tl.dot(dO_i_left, V_j_left.T, dP_acc)
            dP_acc = tl.dot(dO_i_right, V_j_right.T, dP_acc)
            dP = dP_acc[0, :, :]
            
            dS = P * (dP - D_i[:, None]) * tau
            dS = dS * mask_2d
            
            acc_dV_left = tl.dot(P.T, dO_i_left[0, :, :], acc_dV_left)
            acc_dV_right = tl.dot(P.T, dO_i_right[0, :, :], acc_dV_right)
            
            acc_dK_left = tl.dot(dS.T, Q_i_left[0, :, :], acc_dK_left)
            acc_dK_right = tl.dot(dS.T, Q_i_right[0, :, :], acc_dK_right)
            
        dK_desc.store([b_h, j * 128, 0], acc_dK_left.to(tl.bfloat16))
        dK_desc.store([b_h, j * 128, 64], acc_dK_right.to(tl.bfloat16))
        dV_desc.store([b_h, j * 128, 0], acc_dV_left.to(tl.bfloat16))
        dV_desc.store([b_h, j * 128, 64], acc_dV_right.to(tl.bfloat16))


@triton.jit
def _bwd_dq_kernel(
    Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr, dQ_desc,
    S, tau, b_h_max,
    NUM_SMS: tl.constexpr,
):
    num_blocks_s = tl.cdiv(S, 128)
    start_pid = tl.program_id(0)
    total_tiles = num_blocks_s * b_h_max
    
    if start_pid >= total_tiles:
        return
    
    for iteration in tl.range(start_pid, total_tiles, NUM_SMS):
        b_h = iteration // num_blocks_s
        i = iteration % num_blocks_s
        
        Q_i_left = Q_desc.load([b_h, i * 128, 0])
        Q_i_right = Q_desc.load([b_h, i * 128, 64])
        dO_i_left = dO_desc.load([b_h, i * 128, 0])
        dO_i_right = dO_desc.load([b_h, i * 128, 64])
        O_i_left = O_desc.load([b_h, i * 128, 0])
        O_i_right = O_desc.load([b_h, i * 128, 64])
        
        q_mask = (i * 128 + tl.arange(0, 128)) < S
        
        D_i = tl.sum(dO_i_left[0, :, :] * O_i_left[0, :, :] + dO_i_right[0, :, :] * O_i_right[0, :, :], axis=1)
        L_i = tl.load(L_ptr + b_h * S + i * 128 + tl.arange(0, 128), mask=q_mask, other=0.0)
        
        acc_dQ_left = tl.zeros((1, 128, 64), tl.float32)
        acc_dQ_right = tl.zeros((1, 128, 64), tl.float32)
        
        for j in range(num_blocks_s):
            K_j_left = K_desc.load([b_h, j * 128, 0])
            K_j_right = K_desc.load([b_h, j * 128, 64])
            V_j_left = V_desc.load([b_h, j * 128, 0])
            V_j_right = V_desc.load([b_h, j * 128, 64])
            
            k_mask = (j * 128 + tl.arange(0, 128)) < S
            mask_2d = (q_mask[:, None] & k_mask[None, :]).to(tl.float32)
            
            S_acc = tl.zeros((1, 128, 128), tl.float32)
            S_acc = tl.dot(Q_i_left, K_j_left.T, S_acc)
            S_acc = tl.dot(Q_i_right, K_j_right.T, S_acc)
            S_scores = S_acc[0, :, :] * tau
            
            P = tl.exp(S_scores - L_i[:, None])
            P = P * mask_2d
            
            dP_acc = tl.zeros((1, 128, 128), tl.float32)
            dP_acc = tl.dot(dO_i_left, V_j_left.T, dP_acc)
            dP_acc = tl.dot(dO_i_right, V_j_right.T, dP_acc)
            dP = dP_acc[0, :, :]
            
            dS = P * (dP - D_i[:, None]) * tau
            dS = dS * mask_2d
            
            acc_dQ_left = tl.dot(dS, K_j_left[0, :, :], acc_dQ_left)
            acc_dQ_right = tl.dot(dS, K_j_right[0, :, :], acc_dQ_right)
            
        dQ_desc.store([b_h, i * 128, 0], acc_dQ_left.to(tl.bfloat16))
        dQ_desc.store([b_h, i * 128, 64], acc_dQ_right.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    if S == 0:
        return

    Q_3d = Q.view(B * H, S, 128)
    K_3d = K.view(B * H, S, 128)
    V_3d = V.view(B * H, S, 128)
    O_3d = O.view(B * H, S, 128)
    dO_3d = dO.view(B * H, S, 128)
    dQ_3d = dQ.view(B * H, S, 128)
    dK_3d = dK.view(B * H, S, 128)
    dV_3d = dV.view(B * H, S, 128)
    
    Q_desc = TensorDescriptor.from_tensor(Q_3d, [1, 128, 64])
    K_desc = TensorDescriptor.from_tensor(K_3d, [1, 128, 64])
    V_desc = TensorDescriptor.from_tensor(V_3d, [1, 128, 64])
    O_desc = TensorDescriptor.from_tensor(O_3d, [1, 128, 64])
    dO_desc = TensorDescriptor.from_tensor(dO_3d, [1, 128, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ_3d, [1, 128, 64])
    dK_desc = TensorDescriptor.from_tensor(dK_3d, [1, 128, 64])
    dV_desc = TensorDescriptor.from_tensor(dV_3d, [1, 128, 64])
    
    tau = 1.0 / math.sqrt(d)
    
    NUM_SMS = 132
    num_blocks_s = triton.cdiv(S, 128)
    b_h_max = B * H
    
    total_tiles = num_blocks_s * b_h_max
    num_blocks_kv = min(NUM_SMS, total_tiles)
    
    grid_dkv = (num_blocks_kv,)
    _bwd_dkv_kernel[grid_dkv](
        Q_desc, K_desc, V_desc, dO_desc, O_desc, L, dK_desc, dV_desc,
        S, tau, b_h_max,
        NUM_SMS=NUM_SMS,
        num_warps=8
    )
    
    num_blocks_q = min(NUM_SMS, total_tiles)
    grid_dq = (num_blocks_q,)
    _bwd_dq_kernel[grid_dq](
        Q_desc, K_desc, V_desc, dO_desc, O_desc, L, dQ_desc,
        S, tau, b_h_max,
        NUM_SMS=NUM_SMS,
        num_warps=8
    )