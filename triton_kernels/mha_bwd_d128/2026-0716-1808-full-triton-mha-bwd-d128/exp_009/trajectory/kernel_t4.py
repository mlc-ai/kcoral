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
    head_idx = tl.program_id(1)
    bh_idx = tl.program_id(2)
    bh = bh_idx * H + head_idx
    
    D = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    for k in range(0, 128, BLOCK_K):
        do_tile = do_desc.load([bh * S + i * BLOCK_M, k])
        o_tile = o_desc.load([bh * S + i * BLOCK_M, k])
        D += tl.sum(do_tile * o_tile, axis=1, keep_dims=True)
        
    mask_l = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
    L_i = tl.load(l_ptr + bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M), mask=mask_l, other=0.0)
    
    dq_acc = [tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32) for _ in range(4)]
    
    num_blocks = tl.cdiv(S, BLOCK_N)
    for j in range(num_blocks):
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, 128, BLOCK_K):
            q_tile = q_desc.load([bh * S + i * BLOCK_M, k])
            k_tile = k_desc.load([bh * S + j * BLOCK_N, k])
            S_acc = tl.dot(q_tile, k_tile.T, S_acc)
            
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, 128, BLOCK_K):
            do_tile = do_desc.load([bh * S + i * BLOCK_M, k])
            v_tile = v_desc.load([bh * S + j * BLOCK_N, k])
            dP_acc = tl.dot(do_tile, v_tile.T, dP_acc)
            
        P = tl.exp(S_acc * scale - L_i[:, None])
        
        if causal:
            row = i * BLOCK_M + tl.arange(0, BLOCK_M)
            col = j * BLOCK_N + tl.arange(0, BLOCK_N)
            valid = (row[:, None] >= col[None, :]) & (row[:, None] < S) & (col[None, :] < S)
            P = P * valid.to(tl.float32)
            
        valid_row = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
        P = P * valid_row[:, None].to(tl.float32)
        
        dS = P * (dP_acc - D) * scale
        
        if causal:
            dS = dS * valid.to(tl.float32)
            
        dS = dS[:, None, :] * scale  # Shape: [BLOCK_M, 1, BLOCK_N]
        for k_idx, k in enumerate(range(0, 128, BLOCK_K)):
            k_tile_b = k_desc.load([bh * S + j * BLOCK_N, k])  # Shape: [BLOCK_N, BLOCK_K]
            k_block = k_tile_b.T  # Shape: [BLOCK_K, BLOCK_N]
            temp = tl.zeros((BLOCK_M, BLOCK_K, BLOCK_N), dtype=tl.float32)
            temp = tl.dot(dS, k_block, temp)  # Accumulate outer-products across all warp lanes
            temp = temp.sum(axis=-1)  # Reduce over BLOCK_N dimension
            dq_acc[k_idx] = temp
            
    row_off_i = bh * S + i * BLOCK_M
    for k_idx in range(4):
        dq_desc.store([row_off_i, k_idx * 32], dq_acc[k_idx].to(tl.bfloat16))


@triton.jit
def bwd_dkv_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_ptr, dk_desc, dv_desc,
    S, scale, BLOCK_M, BLOCK_N, BLOCK_K, causal
):
    j = tl.program_id(0)
    head_idx = tl.program_id(1)
    bh_idx = tl.program_id(2)
    bh = bh_idx * H + head_idx
    
    dk_acc = [tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32) for _ in range(4)]
    dv_acc = [tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32) for _ in range(4)]
    
    num_blocks = tl.cdiv(S, BLOCK_M)
    for i in range(num_blocks):
        D = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
        for k in range(0, 128, BLOCK_K):
            do_tile = do_desc.load([bh * S + i * BLOCK_M, k])
            o_tile = o_desc.load([bh * S + i * BLOCK_M, k])
            D += tl.sum(do_tile * o_tile, axis=1, keep_dims=True)
            
        mask_l = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
        L_i = tl.load(l_ptr + bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M), mask=mask_l, other=0.0)
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, 128, BLOCK_K):
            q_tile = q_desc.load([bh * S + i * BLOCK_M, k])
            k_tile = k_desc.load([bh * S + j * BLOCK_N, k])
            S_acc = tl.dot(q_tile, k_tile.T, S_acc)
            
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, 128, BLOCK_K):
            do_tile = do_desc.load([bh * S + i * BLOCK_M, k])
            v_tile = v_desc.load([bh * S + j * BLOCK_N, k])
            dP_acc = tl.dot(do_tile, v_tile.T, dP_acc)
            
        P = tl.exp(S_acc * scale - L_i[:, None])
        
        if causal:
            row = i * BLOCK_M + tl.arange(0, BLOCK_M)
            col = j * BLOCK_N + tl.arange(0, BLOCK_N)
            valid = (row[:, None] >= col[None, :]) & (row[:, None] < S) & (col[None, :] < S)
            P = P * valid.to(tl.float32)
            
        valid_row = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
        P = P * valid_row[:, None].to(tl.float32)
        
        dS = P * (dP_acc - D) * scale
        
        if causal:
            dS = dS * valid.to(tl.float32)
            
        dS_T = dS.T[None, :, :]  # Shape: [1, BLOCK_N, BLOCK_M]
        for k_idx, k in enumerate(range(0, 128, BLOCK_K)):
            q_tile_b = q_desc.load([bh * S + i * BLOCK_M, k])  # Shape: [BLOCK_M, BLOCK_K]
            q_block = q_tile_b  
            temp = tl.zeros((BLOCK_N, BLOCK_K, BLOCK_M), dtype=tl.float32)
            temp = tl.dot(dS_T, q_block, temp)
            temp = temp.sum(axis=-1)
            dk_acc[k_idx] = temp
            
        P_T = P.T[None, :, :]  # Shape: [1, BLOCK_N, BLOCK_M]
        for k_idx, k in enumerate(range(0, 128, BLOCK_K)):
            do_tile_b = do_desc.load([bh * S + i * BLOCK_M, k])  # Shape: [BLOCK_M, BLOCK_K]
            do_block = do_tile_b
            temp = tl.zeros((BLOCK_N, BLOCK_K, BLOCK_M), dtype=tl.float32)
            temp = tl.dot(P_T, do_block, temp)
            temp = temp.sum(axis=-1)
            dv_acc[k_idx] = temp
            
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
    grid = (num_blocks, H, B)
    
    L_flat = L.view(B * H * S)
    
    bwd_dq_kernel[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L_flat, dq_desc,
                        S, scale, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, causal=False,
                        num_warps=8, num_stages=2)
                        
    bwd_dkv_kernel[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L_flat, dk_desc, dv_desc,
                         S, scale, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, causal=False,
                         num_warps=8, num_stages=2)