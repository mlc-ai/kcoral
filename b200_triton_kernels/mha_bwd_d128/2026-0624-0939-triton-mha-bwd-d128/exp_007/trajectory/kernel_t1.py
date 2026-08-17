import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _preprocess_kernel(
    dO_ptr, O_ptr, D_ptr,
    S, d, H,
    BLOCK: tl.constexpr
):
    b_h = tl.program_id(1)
    n_offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = n_offs < S
    
    stride_s = d
    stride_h = S * d
    
    k = tl.arange(0, 64)
    
    dO_left = tl.load(dO_ptr + b_h * stride_h + n_offs[:, None] * stride_s + k[None, :],
                 mask=mask[:, None], other=0.0)
    dO_right = tl.load(dO_ptr + b_h * stride_h + n_offs[:, None] * stride_s + 64 + k[None, :],
                 mask=mask[:, None], other=0.0)
                 
    O_left = tl.load(O_ptr + b_h * stride_h + n_offs[:, None] * stride_s + k[None, :],
                mask=mask[:, None], other=0.0)
    O_right = tl.load(O_ptr + b_h * stride_h + n_offs[:, None] * stride_s + 64 + k[None, :],
                mask=mask[:, None], other=0.0)
    
    D = tl.sum(dO_left * O_left + dO_right * O_right, axis=1)
    
    tl.store(D_ptr + b_h * S + n_offs, D, mask=mask)


@triton.jit
def _bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, D_ptr, L_ptr, dK_ptr, dV_ptr,
    S, d, H, tau, stride_h, stride_s, d_half,
    NUM_SMS: tl.constexpr,
):
    num_blocks_s = tl.cdiv(S, 64)
    start_pid = tl.program_id(0)
    num_blocks = min(NUM_SMS, B * H * num_blocks_s)
    
    if start_pid >= num_blocks:
        return
    
    idx = start_pid // num_blocks_s
    j = start_pid % num_blocks_s
    dh = tl.program_id(2)
    
    K_offs_j = j * 64 + tl.arange(0, 64)
    k_mask = K_offs_j < S
    
    K_j_left = tl.load(K_ptr + idx * stride_h + K_offs_j[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=k_mask[:, None], other=0.0)
    K_j_right = tl.load(K_ptr + idx * stride_h + K_offs_j[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=k_mask[:, None], other=0.0)
    V_j_left = tl.load(V_ptr + idx * stride_h + K_offs_j[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=k_mask[:, None], other=0.0)
    V_j_right = tl.load(V_ptr + idx * stride_h + K_offs_j[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=k_mask[:, None], other=0.0)
    
    K_j_left_4d = K_j_left[None, None, :, :]
    K_j_right_4d = K_j_right[None, None, :, :]
    V_j_left_4d = V_j_left[None, None, :, :]
    V_j_right_4d = V_j_right[None, None, :, :]
    
    acc_dK = tl.zeros((2, 1, 64, 64), tl.float32)
    acc_dV = tl.zeros((2, 1, 64, 64), tl.float32)
    
    for i in range(num_blocks_s):
        Q_offs_i = i * 64 + tl.arange(0, 64)
        q_mask = Q_offs_i < S
        
        Q_i_left = tl.load(Q_ptr + idx * stride_h + Q_offs_i[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=q_mask[:, None], other=0.0)
        Q_i_right = tl.load(Q_ptr + idx * stride_h + Q_offs_i[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=q_mask[:, None], other=0.0)
        dO_i_left = tl.load(dO_ptr + idx * stride_h + Q_offs_i[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=q_mask[:, None], other=0.0)
        dO_i_right = tl.load(dO_ptr + idx * stride_h + Q_offs_i[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=q_mask[:, None], other=0.0)
        
        Q_i_left_4d = Q_i_left[None, None, :, :]
        Q_i_right_4d = Q_i_right[None, None, :, :]
        dO_i_left_4d = dO_i_left[None, None, :, :]
        dO_i_right_4d = dO_i_right[None, None, :, :]
        
        D_i = tl.load(D_ptr + idx * S + Q_offs_i, mask=q_mask, other=0.0)
        L_i = tl.load(L_ptr + idx * S + Q_offs_i, mask=q_mask, other=0.0)
        
        D_i_4d = D_i[None, None, :, None]
        L_i_4d = L_i[None, None, :, None]
        
        mask_2d = (q_mask[:, None] & k_mask[None, :]).to(tl.float32)
        mask_4d = mask_2d[None, None, :, :]
        
        Q_batch = tl.join(Q_i_left_4d, Q_i_right_4d)
        K_batch = tl.join(K_j_left_4d, K_j_right_4d)
        
        S_acc = tl.zeros((2, 1, 64, 64), tl.float32)
        for k in range(2):
            S_acc = tl.dot(Q_batch[:, k:k+1, :, :], K_batch[:, k:k+1, :, :].transpose(2, 3), S_acc)
        
        S_scores = S_acc * tau
        
        dO_batch = tl.join(dO_i_left_4d, dO_i_right_4d)
        V_batch = tl.join(V_j_left_4d, V_j_right_4d)
        
        dP_acc = tl.zeros((2, 1, 64, 64), tl.float32)
        for k in range(2):
            dP_acc = tl.dot(dO_batch[:, k:k+1, :, :], V_batch[:, k:k+1, :, :].transpose(2, 3), dP_acc)
        
        dP = dP_acc[0, :, :, :]
        
        P = tl.exp(S_scores - L_i_4d)
        P = P * mask_4d
        
        dS = P * (dP - D_i_4d) * tau
        dS = dS * mask_4d
        
        P_4d = P
        dS_4d = dS
        
        acc_dV = tl.dot(P_4d.transpose(2, 3), dO_batch, acc_dV)
        acc_dK = tl.dot(dS_4d.transpose(2, 3), Q_batch, acc_dK)
        
    dh = tl.program_id(2)
    if dh == 0:
        dK_store_ptrs = dK_ptr + idx * stride_h + (j * 64 + tl.arange(0, 64)[:, None]) * stride_s + tl.arange(0, 64)[None, :]
        dV_store_ptrs = dV_ptr + idx * stride_h + (j * 64 + tl.arange(0, 64)[:, None]) * stride_s + tl.arange(0, 64)[None, :]
    else:
        dK_store_ptrs = dK_ptr + idx * stride_h + (j * 64 + tl.arange(0, 64)[:, None]) * stride_s + 64 + tl.arange(0, 64)[None, :]
        dV_store_ptrs = dV_ptr + idx * stride_h + (j * 64 + tl.arange(0, 64)[:, None]) * stride_s + 64 + tl.arange(0, 64)[None, :]

    valid_row = j * 64 + tl.arange(0, 64)[:, None] < S

    tl.store(dK_store_ptrs, acc_dK[0, 0, :, :], mask=valid_row)
    tl.store(dV_store_ptrs, acc_dV[0, 0, :, :], mask=valid_row)


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, D_ptr, L_ptr, dQ_ptr,
    S, d, H, tau, stride_h, stride_s, d_half,
    NUM_SMS: tl.constexpr,
):
    num_blocks_s = tl.cdiv(S, 64)
    start_pid = tl.program_id(0)
    num_blocks = min(NUM_SMS, B * H * num_blocks_s)
    
    if start_pid >= num_blocks:
        return
    
    idx = start_pid // num_blocks_s
    i = start_pid % num_blocks_s
    dh = tl.program_id(2)
    
    Q_offs_i = i * 64 + tl.arange(0, 64)
    q_mask = Q_offs_i < S
    
    Q_i_left = tl.load(Q_ptr + idx * stride_h + Q_offs_i[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=q_mask[:, None], other=0.0)
    Q_i_right = tl.load(Q_ptr + idx * stride_h + Q_offs_i[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=q_mask[:, None], other=0.0)
    dO_i_left = tl.load(dO_ptr + idx * stride_h + Q_offs_i[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=q_mask[:, None], other=0.0)
    dO_i_right = tl.load(dO_ptr + idx * stride_h + Q_offs_i[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=q_mask[:, None], other=0.0)
    
    Q_i_left_4d = Q_i_left[None, None, :, :]
    Q_i_right_4d = Q_i_right[None, None, :, :]
    dO_i_left_4d = dO_i_left[None, None, :, :]
    dO_i_right_4d = dO_i_right[None, None, :, :]
    
    D_i = tl.load(D_ptr + idx * S + Q_offs_i, mask=q_mask, other=0.0)
    L_i = tl.load(L_ptr + idx * S + Q_offs_i, mask=q_mask, other=0.0)
    D_i_4d = D_i[None, None, :, None]
    L_i_4d = L_i[None, None, :, None]
    
    Q_batch = tl.join(Q_i_left_4d, Q_i_right_4d)
    dO_batch = tl.join(dO_i_left_4d, dO_i_right_4d)
    
    acc_dQ = tl.zeros((2, 1, 64, 64), tl.float32)
    
    for j in range(num_blocks_s):
        K_offs_j = j * 64 + tl.arange(0, 64)
        k_mask = K_offs_j < S
        
        K_j_left = tl.load(K_ptr + idx * stride_h + K_offs_j[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=k_mask[:, None], other=0.0)
        K_j_right = tl.load(K_ptr + idx * stride_h + K_offs_j[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=k_mask[:, None], other=0.0)
        V_j_left = tl.load(V_ptr + idx * stride_h + K_offs_j[:, None] * stride_s + tl.arange(0, 64)[None, :], mask=k_mask[:, None], other=0.0)
        V_j_right = tl.load(V_ptr + idx * stride_h + K_offs_j[:, None] * stride_s + 64 + tl.arange(0, 64)[None, :], mask=k_mask[:, None], other=0.0)
        
        K_j_left_4d = K_j_left[None, None, :, :]
        K_j_right_4d = K_j_right[None, None, :, :]
        V_j_left_4d = V_j_left[None, None, :, :]
        V_j_right_4d = V_j_right[None, None, :, :]
        
        K_batch = tl.join(K_j_left_4d, K_j_right_4d)
        V_batch = tl.join(V_j_left_4d, V_j_right_4d)
        
        mask_2d = (q_mask[:, None] & k_mask[None, :]).to(tl.float32)
        mask_4d = mask_2d[None, None, :, :]
        
        S_acc = tl.zeros((2, 1, 64, 64), tl.float32)
        for k in range(2):
            S_acc = tl.dot(Q_batch[:, k:k+1, :, :], K_batch[:, k:k+1, :, :].transpose(2, 3), S_acc)
        
        S_scores = S_acc * tau
        
        dP_acc = tl.zeros((2, 1, 64, 64), tl.float32)
        for k in range(2):
            dP_acc = tl.dot(dO_batch[:, k:k+1, :, :], V_batch[:, k:k+1, :, :].transpose(2, 3), dP_acc)
        
        dP = dP_acc[0, :, :, :]
        
        P = tl.exp(S_scores - L_i_4d)
        P = P * mask_4d
        
        dS = P * (dP - D_i_4d) * tau
        dS = dS * mask_4d
        
        dS_4d = dS
        
        acc_dQ = tl.dot(dS_4d, K_batch, acc_dQ)

    dh = tl.program_id(2)
    if dh == 0:
        dQ_store_ptrs = dQ_ptr + idx * stride_h + (i * 64 + tl.arange(0, 64)[:, None]) * stride_s + tl.arange(0, 64)[None, :]
    else:
        dQ_store_ptrs = dQ_ptr + idx * stride_h + (i * 64 + tl.arange(0, 64)[:, None]) * stride_s + 64 + tl.arange(0, 64)[None, :]
        
    valid_row = i * 64 + tl.arange(0, 64)[:, None] < S
    tl.store(dQ_store_ptrs, acc_dQ[0, 0, :, :], mask=valid_row)


B = 4
H = 48
d = 128
BLOCK_Q = 64
BLOCK_K = 64

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    D = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 64, 64])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, 64, 64])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, 64, 64])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, 64, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 1, 64, 64])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 1, 64, 64])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 1, 64, 64])
    
    tau = 1.0 / math.sqrt(d)
    
    NUM_SMS = 132
    stride_h = S * d
    stride_s = d
    
    grid_preprocess = (triton.cdiv(S, 256), B * H)
    _preprocess_kernel[grid_preprocess](dO, O, D, S, d, H, BLOCK=256)
    torch.cuda.synchronize()
    
    num_blocks_s_kv = triton.cdiv(S, BLOCK_K)
    num_blocks_kv = min(NUM_SMS, B * H * num_blocks_s_kv)
    grid_dkv = (num_blocks_kv, 1, 2)
    _bwd_dkv_kernel[grid_dkv](
        Q_desc, K_desc, V_desc, dO_desc, D, L, dK_desc, dV_desc,
        S, d, H, tau, stride_h, stride_s, 0,
        NUM_SMS=NUM_SMS,
        num_warps=8
    )
    
    num_blocks_s_q = triton.cdiv(S, BLOCK_Q)
    num_blocks_q = min(NUM_SMS, B * H * num_blocks_s_q)
    grid_dq = (num_blocks_q, 1, 2)
    _bwd_dq_kernel[grid_dq](
        Q_desc, K_desc, V_desc, dO_desc, D, L, dQ_desc,
        S, d, H, tau, stride_h, stride_s, 0,
        NUM_SMS=NUM_SMS,
        num_warps=8
    )