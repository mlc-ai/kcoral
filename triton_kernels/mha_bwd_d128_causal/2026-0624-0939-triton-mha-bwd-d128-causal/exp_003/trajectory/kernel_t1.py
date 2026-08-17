import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit(do_not_specialize=True)
def _bwd_pass1(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr,
    dQ_desc, S_len, scale, H_val,
    BLOCK_SEQ: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    q_block = tl.program_id(0)
    
    # Load Q, O, dO for this query block
    Q0 = Q_desc.load([b, h, q_block * 128, 0])
    Q1 = Q_desc.load([b, h, q_block * 128, 64])
    O0 = O_desc.load([b, h, q_block * 128, 0])
    O1 = O_desc.load([b, h, q_block * 128, 64])
    dO0 = dO_desc.load([b, h, q_block * 128, 0])
    dO1 = dO_desc.load([b, h, q_block * 128, 64])
    
    # Precompute D = sum_d(dO * O)
    D = tl.sum(dO0 * O0, axis=1) + tl.sum(dO1 * O1, axis=1)
    
    # Load L for this query block
    q_pos = q_block * 128 + tl.arange(0, 128)
    mask_l = q_pos < S_len
    l_off = (b * H_val + h) * S_len + q_pos
    L_tile = tl.load(L_ptr + l_off, mask=mask_l, other=0.0)
    
    # Initialize dQ accumulators
    dQ_acc0 = tl.zeros((BLOCK_SEQ, 64), tl.float32)
    dQ_acc1 = tl.zeros((BLOCK_SEQ, 64), tl.float32)
    
    # Row indices for the causal mask
    q_idx_2d = q_block * 128 + tl.arange(0, 128)
    
    # Iterate over all key blocks <= current query block
    K0_next = K1_next = V0_next = V1_next = None
    pre_load = False
    if q_block >= 0:
        pre_load = True
        K0_next = K_desc.load([b, h, 0, 0])
        K1_next = K_desc.load([b, h, 0, 64])
        V0_next = V_desc.load([b, h, 0, 0])
        V1_next = V_desc.load([b, h, 0, 64])
        
    for k_block in range(q_block + 1):
        if pre_load:
            if k_block + 1 <= q_block:
                K0_next = K_desc.load([b, h, (k_block + 1) * 128, 0])
                K1_next = K_desc.load([b, h, (k_block + 1) * 128, 64])
                V0_next = V_desc.load([b, h, (k_block + 1) * 128, 0])
                V1_next = V_desc.load([b, h, (k_block + 1) * 128, 64])
        
        K0, K1, V0, V1 = K0_next, K1_next, V0_next, V1_next
        
        k_idx_2d = k_block * 128 + tl.arange(0, 128)
        
        S_tile = (tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)) * scale
        
        valid_k = k_idx_2d[None, :] < S_len
        valid_q_row = q_idx_2d[:, None] < S_len
        mask = valid_q_row & valid_k & (q_idx_2d[:, None] >= k_idx_2d[None, :])
        
        P_tile = tl.exp(S_tile - L_tile[:, None])
        P_tile = tl.where(mask, P_tile, 0.0)
        
        dP_tile = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        dS_tile = P_tile * (dP_tile - D[:, None]) * scale
        
        dQ_acc0 = tl.dot(dS_tile, K0, dQ_acc0)
        dQ_acc1 = tl.dot(dS_tile, K1, dQ_acc1)
    
    # Persist results back into the bfloat16 output tensor
    dQ_desc.store([b, h, q_block * 128, 0], dQ_acc0.to(tl.bfloat16)[None, None, :, :])
    dQ_desc.store([b, h, q_block * 128, 64], dQ_acc1.to(tl.bfloat16)[None, None, :, :])


@triton.jit(do_not_specialize=True)
def _bwd_pass2(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr,
    dK_desc, dV_desc, S_len, scale, H_val,
    BLOCK_SEQ: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    k_block = tl.program_id(0)
    
    # Load K, V for this key block
    K0 = K_desc.load([b, h, k_block * 128, 0])
    K1 = K_desc.load([b, h, k_block * 128, 64])
    V0 = V_desc.load([b, h, k_block * 128, 0])
    V1 = V_desc.load([b, h, k_block * 128, 64])
    
    dK_acc0 = tl.zeros((BLOCK_SEQ, 64), tl.float32)
    dK_acc1 = tl.zeros((BLOCK_SEQ, 64), tl.float32)
    dV_acc0 = tl.zeros((BLOCK_SEQ, 64), tl.float32)
    dV_acc1 = tl.zeros((BLOCK_SEQ, 64), tl.float32)
    
    num_q_blocks = tl.cdiv(S_len, 128)
    
    k_idx_2d = k_block * 128 + tl.arange(0, 128)
    
    Q0_next = Q1_next = O0_next = O1_next = dO0_next = dO1_next = None
    pre_load = False
    if k_block < num_q_blocks:
        pre_load = True
        Q0_next = Q_desc.load([b, h, k_block * 128, 0])
        Q1_next = Q_desc.load([b, h, k_block * 128, 64])
        O0_next = O_desc.load([b, h, k_block * 128, 0])
        O1_next = O_desc.load([b, h, k_block * 128, 64])
        dO0_next = dO_desc.load([b, h, k_block * 128, 0])
        dO1_next = dO_desc.load([b, h, k_block * 128, 64])
    
    for q_block in range(k_block, num_q_blocks):
        if pre_load:
            if q_block + 1 < num_q_blocks:
                Q0_next = Q_desc.load([b, h, (q_block + 1) * 128, 0])
                Q1_next = Q_desc.load([b, h, (q_block + 1) * 128, 64])
                O0_next = O_desc.load([b, h, (q_block + 1) * 128, 0])
                O1_next = O_desc.load([b, h, (q_block + 1) * 128, 64])
                dO0_next = dO_desc.load([b, h, (q_block + 1) * 128, 0])
                dO1_next = dO_desc.load([b, h, (q_block + 1) * 128, 64])
        
        Q0, Q1, O0, O1, dO0, dO1 = Q0_next, Q1_next, O0_next, O1_next, dO0_next, dO1_next
        
        q_pos = q_block * 128 + tl.arange(0, 128)
        mask_l = q_pos < S_len
        l_off = (b * H_val + h) * S_len + q_pos
        L_tile = tl.load(L_ptr + l_off, mask=mask_l, other=0.0)
        
        D = tl.sum(dO0 * O0, axis=1) + tl.sum(dO1 * O1, axis=1)
        
        q_idx_2d = q_block * 128 + tl.arange(0, 128)
        
        S_tile = (tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)) * scale
        
        valid_k = k_idx_2d[None, :] < S_len
        valid_q_row = q_idx_2d[:, None] < S_len
        mask = valid_q_row & valid_k & (q_idx_2d[:, None] >= k_idx_2d[None, :])
        
        P_tile = tl.exp(S_tile - L_tile[:, None])
        P_tile = tl.where(mask, P_tile, 0.0)
        
        dP_tile = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        dS_tile = P_tile * (dP_tile - D[:, None]) * scale
        
        dK_acc0 = tl.dot(dS_tile.T, Q0, dK_acc0)
        dK_acc1 = tl.dot(dS_tile.T, Q1, dK_acc1)
        
        dV_acc0 = tl.dot(P_tile.T, dO0, dV_acc0)
        dV_acc1 = tl.dot(P_tile.T, dO1, dV_acc1)
    
    dK_desc.store([b, h, k_block * 128, 0], dK_acc0.to(tl.bfloat16)[None, None, :, :])
    dK_desc.store([b, h, k_block * 128, 64], dK_acc1.to(tl.bfloat16)[None, None, :, :])
    dV_desc.store([b, h, k_block * 128, 0], dV_acc0.to(tl.bfloat16)[None, None, :, :])
    dV_desc.store([b, h, k_block * 128, 64], dV_acc1.to(tl.bfloat16)[None, None, :, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B_val, H_val, S_len, d_dim = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 128, 128])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, 128, 128])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, 128, 128])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, 128, 128])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, 128, 128])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 1, 128, 128])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 1, 128, 128])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 1, 128, 128])
    
    scale = 1.0 / math.sqrt(d_dim)
    grid = (triton.cdiv(S_len, 128), H_val, B_val)
    
    _bwd_pass1[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L,
        dQ_desc, S_len, scale, H_val,
        BLOCK_SEQ=128,
        num_warps=8,
        num_stages=3,
    )
    
    _bwd_pass2[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L,
        dK_desc, dV_desc, S_len, scale, H_val,
        BLOCK_SEQ=128,
        num_warps=8,
        num_stages=3,
    )