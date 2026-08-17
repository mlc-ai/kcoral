import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def bwd_dq_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_ptr, dq_desc,
    S_total, scale, total_len, BLOCK_M, BLOCK_N, BLOCK_K
):
    """
    dQ kernel: fixes query block i, iterates sequentially over key/value blocks j.
    Accumulates local contributions into a persistent register-resident accumulator.
    """
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    row_offset = bh * S_total + i * BLOCK_M
    
    s_Q = [q_desc.load([row_offset, k]) for k in range(0, 128, 32)]
    s_dO = [do_desc.load([row_offset, k]) for k in range(0, 128, 32)]
    s_O = [o_desc.load([row_offset, k]) for k in range(0, 128, 32)]
    
    D = 0.0
    for do_tile, o_tile in zip(s_dO, s_O):
        D += tl.sum(do_tile * o_tile, axis=1)
    
    row_offset_i = bh * S_total + i * BLOCK_M
    mask_l = row_offset_i + tl.arange(0, BLOCK_M) < total_len
    L_i = tl.load(l_ptr + row_offset_i + tl.arange(0, BLOCK_M), mask=mask_l, other=0.0)
    
    dQ_acc = [tl.zeros((BLOCK_M, 32), dtype=tl.float32) for _ in range(4)]
    
    num_blocks = tl.cdiv(S_total, BLOCK_N)
    for j in range(num_blocks):
        row_offset_j = bh * S_total + j * BLOCK_N
        
        s_K = [k_desc.load([row_offset_j, k]) for k in range(0, 128, 32)]
        s_V = [v_desc.load([row_offset_j, k]) for k in range(0, 128, 32)]
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for q_tile, k_tile in zip(s_Q, s_K):
            S_acc = tl.dot(q_tile, k_tile.T, S_acc)
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for do_tile, v_tile in zip(s_dO, s_V):
            dP_acc = tl.dot(do_tile, v_tile.T, dP_acc)
        
        P = tl.exp(S_acc * scale - L_i[:, None])
        
        col = j * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_col = col < S_total
        P = P * valid_col[None, :].to(tl.float32)
        
        dS = P * (dP_acc - D[:, None]) * scale
        
        for k_idx in range(4):
            dQ_acc[k_idx] = tl.dot(dS, s_K[k_idx], dQ_acc[k_idx])
            
    for k_idx in range(4):
        dq_desc.store([row_offset, k_idx * 32], dQ_acc[k_idx].to(tl.bfloat16))


@triton.jit
def bwd_dkv_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_ptr, dk_desc, dv_desc,
    S_total, scale, total_len, BLOCK_M, BLOCK_N, BLOCK_K
):
    """
    dKdV kernel: fixes key/value block j, iterates sequentially over query blocks i.
    Accumulates local contributions into persistent accumulators.
    """
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    row_offset_j = bh * S_total + j * BLOCK_N
    
    s_K = [k_desc.load([row_offset_j, k]) for k in range(0, 128, 32)]
    s_V = [v_desc.load([row_offset_j, k]) for k in range(0, 128, 32)]
    
    dK_acc = [tl.zeros((BLOCK_N, 32), dtype=tl.float32) for _ in range(4)]
    dV_acc = [tl.zeros((BLOCK_N, 32), dtype=tl.float32) for _ in range(4)]
    
    num_blocks = tl.cdiv(S_total, BLOCK_M)
    for i in range(num_blocks):
        row_offset_i = bh * S_total + i * BLOCK_M
        
        s_Q = [q_desc.load([row_offset_i, k]) for k in range(0, 128, 32)]
        s_dO = [do_desc.load([row_offset_i, k]) for k in range(0, 128, 32)]
        s_O = [o_desc.load([row_offset_i, k]) for k in range(0, 128, 32)]
        
        D = 0.0
        for do_tile, o_tile in zip(s_dO, s_O):
            D += tl.sum(do_tile * o_tile, axis=1)
        
        mask_l = row_offset_i + tl.arange(0, BLOCK_M) < total_len
        L_i = tl.load(l_ptr + row_offset_i + tl.arange(0, BLOCK_M), mask=mask_l, other=0.0)
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for q_tile, k_tile in zip(s_Q, s_K):
            S_acc = tl.dot(q_tile, k_tile.T, S_acc)
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for do_tile, v_tile in zip(s_dO, s_V):
            dP_acc = tl.dot(do_tile, v_tile.T, dP_acc)
        
        P = tl.exp(S_acc * scale - L_i[:, None])
        
        row = i * BLOCK_M + tl.arange(0, BLOCK_M)
        valid_row = row < S_total
        P = P * valid_row[:, None].to(tl.float32)
        
        dS = P * (dP_acc - D[:, None]) * scale
        dS = dS * valid_row[:, None].to(tl.float32)
        
        for k_idx in range(4):
            dK_acc[k_idx] = tl.dot(dS.T, s_Q[k_idx], dK_acc[k_idx])
            dV_acc[k_idx] = tl.dot(P.T, s_dO[k_idx], dV_acc[k_idx])
    
    for k_idx in range(4):
        dk_desc.store([row_offset_j, k_idx * 32], dK_acc[k_idx].to(tl.bfloat16))
        dv_desc.store([row_offset_j, k_idx * 32], dV_acc[k_idx].to(tl.bfloat16))


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
    
    q_desc = TensorDescriptor.from_tensor(Q_2d, [32, 32])
    k_desc = TensorDescriptor.from_tensor(K_2d, [32, 32])
    v_desc = TensorDescriptor.from_tensor(V_2d, [32, 32])
    o_desc = TensorDescriptor.from_tensor(O_2d, [32, 32])
    do_desc = TensorDescriptor.from_tensor(dO_2d, [32, 32])
    dq_desc = TensorDescriptor.from_tensor(dQ_2d, [32, 32])
    dk_desc = TensorDescriptor.from_tensor(dK_2d, [32, 32])
    dv_desc = TensorDescriptor.from_tensor(dV_2d, [32, 32])
    
    num_blocks = triton.cdiv(S, 32)
    grid = (num_blocks, B * H)
    
    bwd_dq_kernel[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L, dq_desc,
                        S, scale, B * H * S, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
                        num_warps=4, num_stages=2)
                        
    bwd_dkv_kernel[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L, dk_desc, dv_desc,
                         S, scale, B * H * S, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
                         num_warps=4, num_stages=2)