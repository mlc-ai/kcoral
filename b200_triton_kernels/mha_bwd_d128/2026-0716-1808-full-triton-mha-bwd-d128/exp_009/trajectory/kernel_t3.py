import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def bwd_dq_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_ptr, dq_desc,
    S, scale, BLOCK_M, BLOCK_N, BLOCK_K, causal
):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    s_Q = [q_desc.load([bh * S + i * BLOCK_M, k * 32]) for k in range(4)]
    s_dO = [do_desc.load([bh * S + i * BLOCK_M, k * 32]) for k in range(4)]
    s_O = [o_desc.load([bh * S + i * BLOCK_M, k * 32]) for k in range(4)]
    
    D = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    for do_tile, o_tile in zip(s_dO, s_O):
        D += tl.sum(do_tile * o_tile, axis=1, keep_dims=True)
    
    mask_l = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
    L_i = tl.load(l_ptr + bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M), mask=mask_l, other=0.0)
    
    dq_acc = [tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32) for _ in range(4)]
    
    num_blocks = tl.cdiv(S, BLOCK_N)
    for j in range(num_blocks):
        s_K = [k_desc.load([bh * S + j * BLOCK_N, k * 32]) for k in range(4)]
        s_V = [v_desc.load([bh * S + j * BLOCK_N, k * 32]) for k in range(4)]
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for q_tile, k_tile in zip(s_Q, s_K):
            S_acc = tl.dot(q_tile, k_tile.T, S_acc)
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for do_tile, v_tile in zip(s_dO, s_V):
            dP_acc = tl.dot(do_tile, v_tile.T, dP_acc)
            
        P = tl.exp(S_acc * scale - L_i[:, None])
        
        if causal:
            row = i * BLOCK_M + tl.arange(0, BLOCK_M)
            col = j * BLOCK_N + tl.arange(0, BLOCK_N)
            valid = row[:, None] >= col[None, :]
            P = P * valid.to(tl.float32)
            
        dS = P * (dP_acc - D) * scale
        
        if causal:
            dS = dS * valid.to(tl.float32)
            
        for k_idx in range(4):
            k_tile = s_K[k_idx]
            dq_acc[k_idx] = tl.dot(dS, k_tile, dq_acc[k_idx])
            
    row_off_i = bh * S + i * BLOCK_M
    for k_idx in range(4):
        dq_desc.store([row_off_i, k_idx * 32], dq_acc[k_idx].to(tl.bfloat16))


@triton.jit
def bwd_dkv_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_ptr, dk_desc, dv_desc,
    S, scale, BLOCK_M, BLOCK_N, BLOCK_K, causal
):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    s_K = [k_desc.load([bh * S + j * BLOCK_N, k * 32]) for k in range(4)]
    s_V = [v_desc.load([bh * S + j * BLOCK_N, k * 32]) for k in range(4)]
    
    dk_acc = [tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32) for _ in range(4)]
    dv_acc = [tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32) for _ in range(4)]
    
    num_blocks = tl.cdiv(S, BLOCK_M)
    for i in range(num_blocks):
        s_Q = [q_desc.load([bh * S + i * BLOCK_M, k * 32]) for k in range(4)]
        s_dO = [do_desc.load([bh * S + i * BLOCK_M, k * 32]) for k in range(4)]
        s_O = [o_desc.load([bh * S + i * BLOCK_M, k * 32]) for k in range(4)]
        
        D = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
        for do_tile, o_tile in zip(s_dO, s_O):
            D += tl.sum(do_tile * o_tile, axis=1, keep_dims=True)
            
        mask_l = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
        L_i = tl.load(l_ptr + bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M), mask=mask_l, other=0.0)
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for q_tile, k_tile in zip(s_Q, s_K):
            S_acc = tl.dot(q_tile, k_tile.T, S_acc)
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for do_tile, v_tile in zip(s_dO, s_V):
            dP_acc = tl.dot(do_tile, v_tile.T, dP_acc)
            
        P = tl.exp(S_acc * scale - L_i[:, None])
        
        if causal:
            row = i * BLOCK_M + tl.arange(0, BLOCK_M)
            col = j * BLOCK_N + tl.arange(0, BLOCK_N)
            valid = row[:, None] >= col[None, :]
            P = P * valid.to(tl.float32)
            
        dS = P * (dP_acc - D) * scale
        
        if causal:
            dS = dS * valid.to(tl.float32)
            
        for k_idx in range(4):
            q_tile = s_Q[k_idx]
            dk_acc[k_idx] = tl.dot(dS.T, q_tile, dk_acc[k_idx])
            
            do_tile = s_dO[k_idx]
            dv_acc[k_idx] = tl.dot(P.T, do_tile, dv_acc[k_idx])
            
    row_off_j = bh * S + j * BLOCK_N
    for k_idx in range(4):
        dk_desc.store([row_off_j, k_idx * 32], dk_acc[k_idx].to(tl.bfloat16))
        dv_desc.store([row_off_j, k_idx * 32], dv_acc[k_idx].to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute the backward pass of Multi-Head Attention."""
    B, H, S, d = Q.shape
    assert B == 4 and H == 48 and d == 128
    assert O.shape == (B, H, S, d) and dO.shape == (B, H, S, d)
    assert L.shape == (B, H, S)
    assert dQ.shape == (B, H, S, d) and dK.shape == (B, H, S, d) and dV.shape == (B, H, S, d)
    
    torch.cuda.set_device(Q.device)
    
    total_S = B * H * S
    scale = 1.0 / (d ** 0.5)
    
    Q_2d = Q.view(total_S, d)
    K_2d = K.view(total_S, d)
    V_2d = V.view(total_S, d)
    O_2d = O.view(total_S, d)
    dO_2d = dO.view(total_S, d)
    dQ_2d = dQ.view(total_S, d)
    dK_2d = dK.view(total_S, d)
    dV_2d = dV.view(total_S, d)
    
    q_desc = TensorDescriptor.from_tensor(Q_2d, [64, 32])
    k_desc = TensorDescriptor.from_tensor(K_2d, [64, 32])
    v_desc = TensorDescriptor.from_tensor(V_2d, [64, 32])
    o_desc = TensorDescriptor.from_tensor(O_2d, [64, 32])
    do_desc = TensorDescriptor.from_tensor(dO_2d, [64, 32])
    dq_desc = TensorDescriptor.from_tensor(dQ_2d, [64, 32])
    dk_desc = TensorDescriptor.from_tensor(dK_2d, [64, 32])
    dv_desc = TensorDescriptor.from_tensor(dV_2d, [64, 32])
    
    num_blocks = triton.cdiv(S, 64)
    grid = (num_blocks, B * H)
    
    L_flat = L.view(B * H * S)
    
    bwd_dq_kernel[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L_flat, dq_desc,
                        S, scale, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, causal=False,
                        num_warps=8, num_stages=2)
                        
    bwd_dkv_kernel[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L_flat, dk_desc, dv_desc,
                         S, scale, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, causal=False,
                         num_warps=8, num_stages=2)