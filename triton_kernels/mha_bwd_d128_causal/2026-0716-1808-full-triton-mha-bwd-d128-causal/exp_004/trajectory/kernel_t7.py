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
    b = bh // H
    h = bh % H
    
    offset_m = pid_m * BLOCK_M
    if offset_m >= S:
        return
    
    dim_offset_1 = HEAD_DIM // 2
    
    q0 = Q_desc.load([b, h, offset_m, 0])
    q1 = Q_desc.load([b, h, offset_m, dim_offset_1])
    
    o0 = O_desc.load([b, h, offset_m, 0])
    o1 = O_desc.load([b, h, offset_m, dim_offset_1])
    
    do0 = dO_desc.load([b, h, offset_m, 0])
    do1 = dO_desc.load([b, h, offset_m, dim_offset_1])
    
    d_val = (do0 * o0).sum(axis=1) + (do1 * o1).sum(axis=1)
    
    l_load = L_desc.load([b, h, offset_m])
    
    acc_dQ0 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    acc_dQ1 = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    q_row = tl.arange(0, BLOCK_M)
    k_row = tl.arange(0, BLOCK_N)
    
    j_max = (S - 1) // BLOCK_N if S > 0 else 0
    max_j = min(pid_m, j_max)
    
    for j in range(max_j + 1):
        offset_n = j * BLOCK_N
        
        k0 = K_desc.load([b, h, offset_n, 0])
        k1 = K_desc.load([b, h, offset_n, dim_offset_1])
        
        v0 = V_desc.load([b, h, offset_n, 0])
        v1 = V_desc.load([b, h, offset_n, dim_offset_1])
        
        acc_S = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_S = tl.dot(q0, k0.T, acc_S)
        acc_S = tl.dot(q1, k1.T, acc_S)
        
        acc_dP = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_dP = tl.dot(do0, v0.T, acc_dP)
        acc_dP = tl.dot(do1, v1.T, acc_dP)
        
        p_unmasked = tl.exp(acc_S * scale - l_load[:, None])
        
        global_q_idx = (pid_m * BLOCK_M + q_row[:, None])
        global_k_idx = (j * BLOCK_N + k_row[None, :])
        mask_2d = (global_q_idx >= global_k_idx) & (global_k_idx < S) & (global_q_idx < S)
        
        p_masked = tl.where(mask_2d, p_unmasked, 0.0)
        ds = tl.where(mask_2d, p_masked * (acc_dP - d_val[:, None]) * scale, 0.0)
        
        acc_dQ0 = tl.dot(ds, k0, acc_dQ0)
        acc_dQ1 = tl.dot(ds, k1, acc_dQ1)
        
    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_D)
    row_mask_0 = (offset_m + row[:, None]) < S
    
    dQ_desc.store([b, h, offset_m, 0], acc_dQ0, mask=row_mask_0)
    dQ_desc.store([b, h, offset_m, dim_offset_1], acc_dQ1, mask=row_mask_0)


@triton.jit
def _bwd_dK_dV(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc,
    S, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Resolves gradients w.r.t. the memory bank keys and values (dK, dV)."""
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    
    offset_n = pid_n * BLOCK_N
    if offset_n >= S:
        return
    
    dim_offset_1 = HEAD_DIM // 2
    
    k0 = K_desc.load([b, h, offset_n, 0])
    k1 = K_desc.load([b, h, offset_n, dim_offset_1])
    
    v0 = V_desc.load([b, h, offset_n, 0])
    v1 = V_desc.load([b, h, offset_n, dim_offset_1])
    
    acc_dK0 = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    acc_dK1 = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    acc_dV0 = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    acc_dV1 = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    q_row = tl.arange(0, BLOCK_M)
    k_row = tl.arange(0, BLOCK_N)
    
    num_blocks_per_head = (S + BLOCK_M - 1) // BLOCK_M
    
    for i in range(pid_n, num_blocks_per_head):
        offset_m = i * BLOCK_M
        if offset_m >= S:
            break
            
        q0 = Q_desc.load([b, h, offset_m, 0])
        q1 = Q_desc.load([b, h, offset_m, dim_offset_1])
        
        o0 = O_desc.load([b, h, offset_m, 0])
        o1 = O_desc.load([b, h, offset_m, dim_offset_1])
        
        do0 = dO_desc.load([b, h, offset_m, 0])
        do1 = dO_desc.load([b, h, offset_m, dim_offset_1])
        
        d_val = (do0 * o0).sum(axis=1) + (do1 * o1).sum(axis=1)
        
        l_load = L_desc.load([b, h, offset_m])
        
        acc_S = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_S = tl.dot(q0, k0.T, acc_S)
        acc_S = tl.dot(q1, k1.T, acc_S)
        
        acc_dP = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        acc_dP = tl.dot(do0, v0.T, acc_dP)
        acc_dP = tl.dot(do1, v1.T, acc_dP)
        
        p_unmasked = tl.exp(acc_S * scale - l_load[:, None])
        
        global_q_idx = (i * BLOCK_M + q_row[:, None])
        global_k_idx = (pid_n * BLOCK_N + k_row[None, :])
        mask_2d = (global_q_idx >= global_k_idx) & (global_k_idx < S) & (global_q_idx < S)
        
        p_masked = tl.where(mask_2d, p_unmasked, 0.0)
        ds = tl.where(mask_2d, p_masked * (acc_dP - d_val[:, None]) * scale, 0.0)
        
        ds_T = ds.T
        p_T = p_masked.T
        
        acc_dK0 = tl.dot(ds_T, q0, acc_dK0)
        acc_dK1 = tl.dot(ds_T, q1, acc_dK1)
        
        acc_dV0 = tl.dot(p_T, do0, acc_dV0)
        acc_dV1 = tl.dot(p_T, do1, acc_dV1)
        
    row = tl.arange(0, BLOCK_N)
    col = tl.arange(0, BLOCK_D)
    row_mask_n = (offset_n + row[:, None]) < S
    
    dK_desc.store([b, h, offset_n, 0], acc_dK0, mask=row_mask_n)
    dK_desc.store([b, h, offset_n, dim_offset_1], acc_dK1, mask=row_mask_n)
    
    dV_desc.store([b, h, offset_n, 0], acc_dV0, mask=row_mask_n)
    dV_desc.store([b, h, offset_n, dim_offset_1], acc_dV1, mask=row_mask_n)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Host bridge launching the two sequential Triton device-pass routines."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    
    BLOCK_M_dQ = 128
    BLOCK_N = 64
    HEAD_DIM = 128
    BLOCK_D = 64
    
    BLOCK_M_dKdV = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M_dQ, BLOCK_D])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, BLOCK_D])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, BLOCK_D])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M_dQ, BLOCK_D])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK_M_dQ, BLOCK_D])
    L_desc = TensorDescriptor.from_tensor(L, [1, 1, BLOCK_M_dQ])
    
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK_M_dQ, BLOCK_D])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK_N, BLOCK_D])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK_N, BLOCK_D])
    
    num_blocks_per_head = triton.cdiv(S, BLOCK_M_dQ)
    grid = (num_blocks_per_head, B * H)
    
    _bwd_dQ[grid](Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dQ_desc, S, H, BLOCK_M_dQ, BLOCK_N, HEAD_DIM, BLOCK_D, num_warps=4, num_stages=2)
    _bwd_dK_dV[grid](Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, dK_desc, dV_desc, S, H, BLOCK_M_dKdV, BLOCK_N, HEAD_DIM, BLOCK_D, num_warps=4, num_stages=2)