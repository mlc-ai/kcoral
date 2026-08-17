import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dQ(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc,
    S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Computes the gradient w.r.t. the queries (dQ)."""
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    
    offset_m = pid_m * BLOCK_M
    if offset_m >= S:
        return
    
    q_0 = Q_desc.load([bh, offset_m, 0])
    q_1 = Q_desc.load([bh, offset_m, 64])
    
    o_0 = O_desc.load([bh, offset_m, 0])
    o_1 = O_desc.load([bh, offset_m, 64])
    
    do_0 = dO_desc.load([bh, offset_m, 0])
    do_1 = dO_desc.load([bh, offset_m, 64])
    
    d_val = (do_0 * o_0).sum(axis=1) + (do_1 * o_1).sum(axis=1)
    
    l_load = L_desc.load([bh, offset_m])
    
    acc_dQ_0 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    acc_dQ_1 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    q_row = tl.arange(0, BLOCK_M)
    k_row = tl.arange(0, BLOCK_N)
    
    for j in range(pid_m + 1):
        offset_n = j * BLOCK_N
        
        k_0 = K_desc.load([bh, offset_n, 0])
        k_1 = K_desc.load([bh, offset_n, 64])
        
        v_0 = V_desc.load([bh, offset_n, 0])
        v_1 = V_desc.load([bh, offset_n, 64])
        
        acc_S = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_S = tl.dot(q_0, k_0.T, acc_S)
        acc_S = tl.dot(q_1, k_1.T, acc_S)
        
        acc_dP = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_dP = tl.dot(do_0, v_0.T, acc_dP)
        acc_dP = tl.dot(do_1, v_1.T, acc_dP)
        
        p_unmasked = tl.exp(acc_S * scale - l_load[:, None])
        
        global_q_idx = (pid_m * BLOCK_M + q_row[:, None])
        global_k_idx = (j * BLOCK_N + k_row[None, :])
        mask_2d = (global_q_idx >= global_k_idx) & (global_k_idx < S) & (global_q_idx < S)
        
        p_unmasked = tl.where(mask_2d, p_unmasked, 0.0)
        ds = tl.where(mask_2d, p_unmasked * (acc_dP - d_val[:, None]) * scale, 0.0)
        
        acc_dQ_0 = tl.dot(ds, k_0, acc_dQ_0)
        acc_dQ_1 = tl.dot(ds, k_1, acc_dQ_1)
        
    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_D)
    row_mask_0 = (offset_m + row[:, None]) < S
    
    dQ_desc.store([bh, offset_m, 0], acc_dQ_0, mask=row_mask_0)
    dQ_desc.store([bh, offset_m, 64], acc_dQ_1, mask=row_mask_0)


@triton.jit
def _bwd_dK_dV(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
    S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Resolves gradients w.r.t. the memory bank keys and values (dK, dV)."""
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    
    offset_n = pid_n * BLOCK_N
    if offset_n >= S:
        return
    
    k_0 = K_desc.load([bh, offset_n, 0])
    k_1 = K_desc.load([bh, offset_n, 64])
    
    v_0 = V_desc.load([bh, offset_n, 0])
    v_1 = V_desc.load([bh, offset_n, 64])
    
    acc_dK_0 = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    acc_dK_1 = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    acc_dV_0 = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    acc_dV_1 = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    q_row = tl.arange(0, BLOCK_M)
    k_row = tl.arange(0, BLOCK_N)
    
    num_blocks_per_head = (S + BLOCK_M - 1) // BLOCK_M
    
    for i in range(pid_n, num_blocks_per_head):
        offset_m = i * BLOCK_M
        if offset_m >= S:
            break
            
        q_0 = Q_desc.load([bh, offset_m, 0])
        q_1 = Q_desc.load([bh, offset_m, 64])
        
        o_0 = O_desc.load([bh, offset_m, 0])
        o_1 = O_desc.load([bh, offset_m, 64])
        
        do_0 = dO_desc.load([bh, offset_m, 0])
        do_1 = dO_desc.load([bh, offset_m, 64])
        
        d_val = (do_0 * o_0).sum(axis=1) + (do_1 * o_1).sum(axis=1)
        
        l_load = L_desc.load([bh, offset_m])
        
        acc_S = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_S = tl.dot(q_0, k_0.T, acc_S)
        acc_S = tl.dot(q_1, k_1.T, acc_S)
        
        acc_dP = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_dP = tl.dot(do_0, v_0.T, acc_dP)
        acc_dP = tl.dot(do_1, v_1.T, acc_dP)
        
        p_unmasked = tl.exp(acc_S * scale - l_load[:, None])
        
        global_q_idx = (i * BLOCK_M + q_row[:, None])
        global_k_idx = (pid_n * BLOCK_N + k_row[None, :])
        mask_2d = (global_q_idx >= global_k_idx) & (global_k_idx < S) & (global_q_idx < S)
        
        p_unmasked = tl.where(mask_2d, p_unmasked, 0.0)
        ds = tl.where(mask_2d, p_unmasked * (acc_dP - d_val[:, None]) * scale, 0.0)
        
        ds_T = ds.T
        p_T = p_unmasked.T
        
        acc_dK_0 = tl.dot(ds_T, q_0, acc_dK_0)
        acc_dK_1 = tl.dot(ds_T, q_1, acc_dK_1)
        
        acc_dV_0 = tl.dot(p_T, do_0, acc_dV_0)
        acc_dV_1 = tl.dot(p_T, do_1, acc_dV_1)
        
    row = tl.arange(0, BLOCK_N)
    col = tl.arange(0, BLOCK_D)
    row_mask_n = (offset_n + row[:, None]) < S
    
    dK_desc.store([bh, offset_n, 0], acc_dK_0, mask=row_mask_n)
    dK_desc.store([bh, offset_n, 64], acc_dK_1, mask=row_mask_n)
    
    dV_desc.store([bh, offset_n, 0], acc_dV_0, mask=row_mask_n)
    dV_desc.store([bh, offset_n, 64], acc_dV_1, mask=row_mask_n)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Host bridge launching the two sequential Triton device-pass routines."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    
    BLOCK_M = 64
    BLOCK_N = 64
    HEAD_DIM = 128
    BLOCK_D = 64
    
    num_blocks_per_head = triton.cdiv(S, BLOCK_M)
    grid = (num_blocks_per_head, B * H)
    
    Q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_M, BLOCK_D])
    K_desc = TensorDescriptor.from_tensor(K, [BLOCK_N, BLOCK_D])
    V_desc = TensorDescriptor.from_tensor(V, [BLOCK_N, BLOCK_D])
    O_desc = TensorDescriptor.from_tensor(O, [BLOCK_M, BLOCK_D])
    dO_desc = TensorDescriptor.from_tensor(dO, [BLOCK_M, BLOCK_D])
    L_desc = TensorDescriptor.from_tensor(L, [BLOCK_M])
    
    dQ_desc = TensorDescriptor.from_tensor(dQ, [BLOCK_M, BLOCK_D])
    dK_desc = TensorDescriptor.from_tensor(dK, [BLOCK_N, BLOCK_D])
    dV_desc = TensorDescriptor.from_tensor(dV, [BLOCK_N, BLOCK_D])
    
    _bwd_dQ[grid](Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc, S, H, BLOCK_M, BLOCK_N, HEAD_DIM, BLOCK_D)
    _bwd_dK_dV[grid](Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc, S, H, BLOCK_M, BLOCK_N, HEAD_DIM, BLOCK_D)