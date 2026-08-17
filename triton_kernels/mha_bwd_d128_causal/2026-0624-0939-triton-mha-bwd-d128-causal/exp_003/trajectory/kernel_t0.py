import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_pass1(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr,
    dQ_desc, S, scale, H,
    BLOCK_SEQ: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    q_block = tl.program_id(0)
    
    # Load Q, O, dO for this query block (split d=128 into two 64-element halves)
    Q0 = tl.reshape(Q_desc.load([b, h, q_block * BLOCK_SEQ, 0]), (BLOCK_SEQ, 64))
    Q1 = tl.reshape(Q_desc.load([b, h, q_block * BLOCK_SEQ, 64]), (BLOCK_SEQ, 64))
    O0 = tl.reshape(O_desc.load([b, h, q_block * BLOCK_SEQ, 0]), (BLOCK_SEQ, 64))
    O1 = tl.reshape(O_desc.load([b, h, q_block * BLOCK_SEQ, 64]), (BLOCK_SEQ, 64))
    dO0 = tl.reshape(dO_desc.load([b, h, q_block * BLOCK_SEQ, 0]), (BLOCK_SEQ, 64))
    dO1 = tl.reshape(dO_desc.load([b, h, q_block * BLOCK_SEQ, 64]), (BLOCK_SEQ, 64))
    
    # Load L for this query block
    q_pos = q_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    mask_l = q_pos < S
    l_off = (b * H + h) * S + q_pos
    L_tile = tl.load(L_ptr + l_off, mask=mask_l, other=0.0)
    
    # Precompute D = sum_d(dO * O)
    D = tl.sum(dO0 * O0, axis=1) + tl.sum(dO1 * O1, axis=1)
    
    # Initialize dQ accumulators
    dQ_acc0 = tl.zeros((BLOCK_SEQ, 64), tl.float32)
    dQ_acc1 = tl.zeros((BLOCK_SEQ, 64), tl.float32)
    
    # Row indices for the causal mask
    q_idx_2d = q_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)[:, None]
    
    # Iterate over all key blocks <= current query block
    for k_block in range(q_block + 1):
        k_idx_2d = k_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)[None, :]
        
        K0 = tl.reshape(K_desc.load([b, h, k_block * BLOCK_SEQ, 0]), (BLOCK_SEQ, 64))
        K1 = tl.reshape(K_desc.load([b, h, k_block * BLOCK_SEQ, 64]), (BLOCK_SEQ, 64))
        V0 = tl.reshape(V_desc.load([b, h, k_block * BLOCK_SEQ, 0]), (BLOCK_SEQ, 64))
        V1 = tl.reshape(V_desc.load([b, h, k_block * BLOCK_SEQ, 64]), (BLOCK_SEQ, 64))
        
        # S = Q @ K^T * scale
        S_tile = (tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)) * scale
        
        # Causal mask restricted to valid token boundaries
        mask = (q_idx_2d >= k_idx_2d) & (q_idx_2d < S) & (k_idx_2d < S)
        
        # P = exp(S - L)
        P_tile = tl.exp(S_tile - L_tile[:, None])
        P_tile = tl.where(mask, P_tile, 0.0)
        
        # dP = dO @ V^T
        dP_tile = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        # dS = P * (dP - D) * scale
        dS_tile = P_tile * (dP_tile - D[:, None]) * scale
        
        # dQ += dS @ K
        dQ_acc0 = tl.dot(dS_tile, K0, dQ_acc0)
        dQ_acc1 = tl.dot(dS_tile, K1, dQ_acc1)
    
    # Persist results back into the bfloat16 output tensor
    dQ_desc.store([b, h, q_block * BLOCK_SEQ, 0], dQ_acc0.to(tl.bfloat16))
    dQ_desc.store([b, h, q_block * BLOCK_SEQ, 64], dQ_acc1.to(tl.bfloat16))


@triton.jit
def _bwd_pass2(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr,
    dK_desc, dV_desc, S, scale, H,
    BLOCK_SEQ: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    k_block = tl.program_id(0)
    
    # Load K, V for this key block
    K0 = tl.reshape(K_desc.load([b, h, k_block * BLOCK_SEQ, 0]), (BLOCK_SEQ, 64))
    K1 = tl.reshape(K_desc.load([b, h, k_block * BLOCK_SEQ, 64]), (BLOCK_SEQ, 64))
    V0 = tl.reshape(V_desc.load([b, h, k_block * BLOCK_SEQ, 0]), (BLOCK_SEQ, 64))
    V1 = tl.reshape(V_desc.load([b, h, k_block * BLOCK_SEQ, 64]), (BLOCK_SEQ, 64))
    
    dK_acc0 = tl.zeros((BLOCK_SEQ, 64), tl.float32)
    dK_acc1 = tl.zeros((BLOCK_SEQ, 64), tl.float32)
    dV_acc0 = tl.zeros((BLOCK_SEQ, 64), tl.float32)
    dV_acc1 = tl.zeros((BLOCK_SEQ, 64), tl.float32)
    
    num_q_blocks = tl.cdiv(S, BLOCK_SEQ)
    
    k_idx_2d = k_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)[None, :]
    
    # Iterate over all query blocks >= current key block
    for q_block in range(k_block, num_q_blocks):
        q_pos = q_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
        mask_l = q_pos < S
        l_off = (b * H + h) * S + q_pos
        L_tile = tl.load(L_ptr + l_off, mask=mask_l, other=0.0)
        
        Q0 = tl.reshape(Q_desc.load([b, h, q_block * BLOCK_SEQ, 0]), (BLOCK_SEQ, 64))
        Q1 = tl.reshape(Q_desc.load([b, h, q_block * BLOCK_SEQ, 64]), (BLOCK_SEQ, 64))
        O0 = tl.reshape(O_desc.load([b, h, q_block * BLOCK_SEQ, 0]), (BLOCK_SEQ, 64))
        O1 = tl.reshape(O_desc.load([b, h, q_block * BLOCK_SEQ, 64]), (BLOCK_SEQ, 64))
        dO0 = tl.reshape(dO_desc.load([b, h, q_block * BLOCK_SEQ, 0]), (BLOCK_SEQ, 64))
        dO1 = tl.reshape(dO_desc.load([b, h, q_block * BLOCK_SEQ, 64]), (BLOCK_SEQ, 64))
        
        D = tl.sum(dO0 * O0, axis=1) + tl.sum(dO1 * O1, axis=1)
        
        q_idx_2d = q_block * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)[:, None]
        
        S_tile = (tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)) * scale
        
        mask = (q_idx_2d >= k_idx_2d) & (q_idx_2d < S) & (k_idx_2d < S)
        
        P_tile = tl.exp(S_tile - L_tile[:, None])
        P_tile = tl.where(mask, P_tile, 0.0)
        
        dP_tile = tl.dot(dO0, V0.T) + tl.dot(dO1, V1.T)
        
        dS_tile = P_tile * (dP_tile - D[:, None]) * scale
        
        dK_acc0 = tl.dot(dS_tile.T, Q0, dK_acc0)
        dK_acc1 = tl.dot(dS_tile.T, Q1, dK_acc1)
        
        dV_acc0 = tl.dot(P_tile.T, dO0, dV_acc0)
        dV_acc1 = tl.dot(P_tile.T, dO1, dV_acc1)
    
    dK_desc.store([b, h, k_block * BLOCK_SEQ, 0], dK_acc0.to(tl.bfloat16))
    dK_desc.store([b, h, k_block * BLOCK_SEQ, 64], dK_acc1.to(tl.bfloat16))
    dV_desc.store([b, h, k_block * BLOCK_SEQ, 0], dV_acc0.to(tl.bfloat16))
    dV_desc.store([b, h, k_block * BLOCK_SEQ, 64], dV_acc1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 64, 64])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, 64, 64])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, 64, 64])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, 64, 64])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, 64, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 1, 64, 64])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 1, 64, 64])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 1, 64, 64])
    
    scale = 1.0 / math.sqrt(d)
    grid = (triton.cdiv(S, 64), H, B)
    
    _bwd_pass1[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L,
        dQ_desc, S, scale, H,
        BLOCK_SEQ=64,
        num_warps=4,
        num_stages=2,
    )
    
    _bwd_pass2[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L,
        dK_desc, dV_desc, S, scale, H,
        BLOCK_SEQ=64,
        num_warps=4,
        num_stages=2,
    )