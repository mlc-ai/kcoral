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
    b_h = tl.program_id(0)
    n_offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = n_offs < S
    
    k = tl.arange(0, 128)
    full_mask = (mask[:, None] & (k < 128)[None, :]).to_bytes()
    
    dO = tl.load(dO_ptr + b_h * S * d + n_offs[:, None] * d + k[None, :],
                 mask=full_mask, other=0.0f)
    O = tl.load(O_ptr + b_h * S * d + n_offs[:, None] * d + k[None, :],
                mask=full_mask, other=0.0f)
    
    D = tl.sum(dO * O, axis=1)
    
    tl.store(D_ptr + b_h * S + n_offs, D, mask=mask)


@triton.jit
def _bwd_dkv_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, D_ptr, L_ptr, dK_desc, dV_desc,
    S, d, H, tau,
    NUM_SMS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    num_blocks_s = tl.cdiv(S, BLOCK_K)
    start_pid = tl.program_id(0)
    num_blocks = min(NUM_SMS, B * H * num_blocks_s)
    
    if start_pid >= num_blocks:
        return
    
    idx = start_pid // num_blocks_s
    j = start_pid % num_blocks_s
    
    K_tile = K_desc.load([idx, j * BLOCK_K, 0])
    V_tile = V_desc.load([idx, j * BLOCK_K, 0])
    
    acc_dK = tl.zeros((1, 1, BLOCK_K, 128), tl.float32)
    acc_dV = tl.zeros((1, 1, BLOCK_K, 128), tl.float32)
    
    for i in tl.range(num_blocks_s, num_stages=2):
        Q_tile = Q_desc.load([idx, i * BLOCK_Q, 0])
        dO_tile = dO_desc.load([idx, i * BLOCK_Q, 0])
        
        q_mask = (i * BLOCK_Q + tl.arange(0, BLOCK_Q)) < S
        k_mask = (j * BLOCK_K + tl.arange(0, BLOCK_K)) < S
        mask_2d = (q_mask[:, None] & k_mask[None, :]).to(tl.float32)
        
        D_i = tl.load(D_ptr + idx * S + i * BLOCK_Q + tl.arange(0, BLOCK_Q),
                       mask=q_mask, other=0.0)
        L_i = tl.load(L_ptr + idx * S + i * BLOCK_Q + tl.arange(0, BLOCK_Q),
                       mask=q_mask, other=0.0)
        
        D_i_expanded = D_i[None, None, :, None]
        L_i_expanded = L_i[None, None, :, None]
        
        S_scores = tl.dot(Q_tile, K_tile.T) * tau
        P = tl.exp(S_scores - L_i_expanded)
        P = P * mask_2d[None, None, :, :]
        
        dP = tl.dot(dO_tile, V_tile.T)
        dS = P * (dP - D_i_expanded) * tau
        
        acc_dV = tl.dot(P.T, dO_tile, acc_dV)
        acc_dK = tl.dot(dS.T, Q_tile, acc_dK)
    
    dK_desc.store([idx, j * BLOCK_K, 0], acc_dK)
    dV_desc.store([idx, j * BLOCK_K, 0], acc_dV)


@triton.jit
def _bwd_dq_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, D_ptr, L_ptr, dQ_desc,
    S, d, H, tau,
    NUM_SMS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    num_blocks_s = tl.cdiv(S, BLOCK_Q)
    start_pid = tl.program_id(0)
    num_blocks = min(NUM_SMS, B * H * num_blocks_s)
    
    if start_pid >= num_blocks:
        return
    
    idx = start_pid // num_blocks_s
    i = start_pid % num_blocks_s
    
    Q_tile = Q_desc.load([idx, i * BLOCK_Q, 0])
    dO_tile = dO_desc.load([idx, i * BLOCK_Q, 0])
    
    q_mask = (i * BLOCK_Q + tl.arange(0, BLOCK_Q)) < S
    
    D_i = tl.load(D_ptr + idx * S + i * BLOCK_Q + tl.arange(0, BLOCK_Q),
                   mask=q_mask, other=0.0)
    L_i = tl.load(L_ptr + idx * S + i * BLOCK_Q + tl.arange(0, BLOCK_Q),
                   mask=q_mask, other=0.0)
    
    D_i_expanded = D_i[None, None, :, None]
    L_i_expanded = L_i[None, None, :, None]
    
    acc_dQ = tl.zeros((1, 1, BLOCK_Q, 128), tl.float32)
    
    for j in tl.range(num_blocks_s, num_stages=2):
        K_tile = K_desc.load([idx, j * BLOCK_K, 0])
        V_tile = V_desc.load([idx, j * BLOCK_K, 0])
        
        k_mask = (j * BLOCK_K + tl.arange(0, BLOCK_K)) < S
        mask_2d = (q_mask[:, None] & k_mask[None, :]).to(tl.float32)
        
        S_scores = tl.dot(Q_tile, K_tile.T) * tau
        P = tl.exp(S_scores - L_i_expanded)
        P = P * mask_2d[None, None, :, :]
        
        dP = tl.dot(dO_tile, V_tile.T)
        dS = P * (dP - D_i_expanded) * tau
        
        acc_dQ = tl.dot(dS, K_tile, acc_dQ)
    
    dQ_desc.store([idx, i * BLOCK_Q, 0], acc_dQ)


B = 4
H = 48
d = 128
BLOCK_Q = 64
BLOCK_K = 64

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    D = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_Q, 128])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_K, 128])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_K, 128])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_Q, 128])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_Q, 128])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_Q, 128])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_K, 128])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_K, 128])
    
    tau = 1.0 / math.sqrt(d)
    
    NUM_SMS = 132
    
    grid_preprocess = (B * H, triton.cdiv(S, 256))
    _preprocess_kernel[grid_preprocess](dO, O, D, S, d, H, BLOCK=256)
    torch.cuda.synchronize()
    
    num_blocks_s_kv = triton.cdiv(S, BLOCK_K)
    num_blocks_kv = min(NUM_SMS, B * H * num_blocks_s_kv)
    grid_dkv = (num_blocks_kv,)
    _bwd_dkv_kernel[grid_dkv](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, D, L, dK_desc, dV_desc,
        S, d, H, tau,
        NUM_SMS=NUM_SMS,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        num_warps=4
    )
    
    num_blocks_s_q = triton.cdiv(S, BLOCK_Q)
    num_blocks_q = min(NUM_SMS, B * H * num_blocks_s_q)
    grid_dq = (num_blocks_q,)
    _bwd_dq_kernel[grid_dq](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, D, L, dQ_desc,
        S, d, H, tau,
        NUM_SMS=NUM_SMS,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        num_warps=4
    )