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
    
    s_Q = [q_desc.load([bh * S + i * BLOCK_M, k]) for k in range(0, 128, 32)]
    s_dO = [do_desc.load([bh * S + i * BLOCK_M, k]) for k in range(0, 128, 32)]
    s_O = [o_desc.load([bh * S + i * BLOCK_M, k]) for k in range(0, 128, 32)]
    
    D = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    for do_tile, o_tile in zip(s_dO, s_O):
        D += tl.sum(do_tile * o_tile, axis=1, keep_dims=True)
    
    mask_l = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
    L_i = tl.load(l_ptr + bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M), mask=mask_l, other=0.0)
    
    dq_acc = [tl.zeros((BLOCK_M, 32), dtype=tl.float32) for _ in range(4)]
    
    num_blocks = tl.cdiv(S, BLOCK_N)
    for j in range(num_blocks):
        s_K = [k_desc.load([bh * S + j * BLOCK_N, k]) for k in range(0, 128, 32)]
        s_V = [v_desc.load([bh * S + j * BLOCK_N, k]) for k in range(0, 128, 32)]
        
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
            
        valid_row = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
        P = P * valid_row[:, None].to(tl.float32)
        
        dS = P * (dP_acc - D) * scale
        
        for k_idx in range(4):
            k_tile = s_K[k_idx]
            dq_acc[k_idx] = tl.dot(dS, k_tile, dq_acc[k_idx])
            
    dq_ptr = dq_desc._ptr
    row_off_i = bh * S + i * BLOCK_M
    valid_row = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
    mask = valid_row[:, None]
    for k_idx in range(4):
        k_tile = dq_acc[k_idx].to(tl.bfloat16)
        col = k_idx * 32 + tl.arange(0, 32)
        ptr = dq_ptr + (row_off_i + tl.arange(0, BLOCK_M))[:, None] * 128 + col[None, :]
        tl.store(ptr, k_tile, mask=mask)


@triton.jit
def bwd_dk_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_ptr, dk_desc,
    S, scale, BLOCK_M, BLOCK_N, BLOCK_K, causal
):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    s_K = [k_desc.load([bh * S + j * BLOCK_N, k]) for k in range(0, 128, 32)]
    
    dk_acc = [tl.zeros((BLOCK_N, 32), dtype=tl.float32) for _ in range(4)]
    
    num_blocks = tl.cdiv(S, BLOCK_M)
    for i in range(num_blocks):
        s_Q = [q_desc.load([bh * S + i * BLOCK_M, k]) for k in range(0, 128, 32)]
        s_dO = [do_desc.load([bh * S + i * BLOCK_M, k]) for k in range(0, 128, 32)]
        s_O = [o_desc.load([bh * S + i * BLOCK_M, k]) for k in range(0, 128, 32)]
        
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
            
        valid_row = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
        P = P * valid_row[:, None].to(tl.float32)
        
        dS = P * (dP_acc - D) * scale
        
        for k_idx in range(4):
            q_tile = s_Q[k_idx]
            dk_acc[k_idx] = tl.dot(dS.T, q_tile, dk_acc[k_idx])
            
    dk_ptr = dk_desc._ptr
    row_off_j = bh * S + j * BLOCK_N
    valid_row = (j * BLOCK_N + tl.arange(0, BLOCK_N)) < S
    mask = valid_row[:, None]
    for k_idx in range(4):
        k_tile = dk_acc[k_idx].to(tl.bfloat16)
        col = k_idx * 32 + tl.arange(0, 32)
        ptr = dk_ptr + (row_off_j + tl.arange(0, BLOCK_N))[:, None] * 128 + col[None, :]
        tl.store(ptr, k_tile, mask=mask)


@triton.jit
def bwd_dv_kernel(
    q_desc, k_desc, v_desc, o_desc, do_desc, l_ptr, dv_desc,
    S, scale, BLOCK_M, BLOCK_N, BLOCK_K, causal
):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    s_K = [k_desc.load([bh * S + j * BLOCK_N, k]) for k in range(0, 128, 32)]
    s_V = [v_desc.load([bh * S + j * BLOCK_N, k]) for k in range(0, 128, 32)]
    
    dv_acc = [tl.zeros((BLOCK_N, 32), dtype=tl.float32) for _ in range(4)]
    
    num_blocks = tl.cdiv(S, BLOCK_M)
    for i in range(num_blocks):
        s_Q = [q_desc.load([bh * S + i * BLOCK_M, k]) for k in range(0, 128, 32)]
        s_dO = [do_desc.load([bh * S + i * BLOCK_M, k]) for k in range(0, 128, 32)]
        s_O = [o_desc.load([bh * S + i * BLOCK_M, k]) for k in range(0, 128, 32)]
        
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
            
        valid_row = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
        P = P * valid_row[:, None].to(tl.float32)
        
        for k_idx in range(4):
            do_tile = s_dO[k_idx]
            dv_acc[k_idx] = tl.dot(P.T, do_tile, dv_acc[k_idx])
            
    dv_ptr = dv_desc._ptr
    row_off_j = bh * S + j * BLOCK_N
    valid_row = (j * BLOCK_N + tl.arange(0, BLOCK_N)) < S
    mask = valid_row[:, None]
    for k_idx in range(4):
        k_tile = dv_acc[k_idx].to(tl.bfloat16)
        col = k_idx * 32 + tl.arange(0, 32)
        ptr = dv_ptr + (row_off_j + tl.arange(0, BLOCK_N))[:, None] * 128 + col[None, :]
        tl.store(ptr, k_tile, mask=mask)


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
                        S, scale, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, causal=False,
                        num_warps=4, num_stages=2)
                        
    bwd_dk_kernel[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L, dk_desc,
                        S, scale, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, causal=False,
                        num_warps=4, num_stages=2)
                        
    bwd_dv_kernel[grid](q_desc, k_desc, v_desc, o_desc, do_desc, L, dv_desc,
                        S, scale, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, causal=False,
                        num_warps=4, num_stages=2)