import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _dkdv_kernel(
    Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr, D_ptr_unused, dK_ptr, dV_ptr,
    S_len, tau,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0)
    b_idx = tl.program_id(1)
    num_blocks = tl.cdiv(S_len, BLOCK_M)
    
    row_idx = tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, BLOCK_N)
    
    k_smem_0 = tl.shared_memory(bfloat16, 2 * 128 * 128)
    v_smem_0 = tl.shared_memory(bfloat16, 2 * 128 * 128)
    q_smem_0 = tl.shared_memory(bfloat16, 2 * 128 * 128)
    do_smem_0 = tl.shared_memory(bfloat16, 2 * 128 * 128)
    o_smem_0 = tl.shared_memory(bfloat16, 2 * 128 * 128)
    
    d_smem = tl.shared_memory(float, 2 * 128)
    l_smem = tl.shared_memory(float, 2 * 128)
    
    step = tl.int_param(0)
    
    if step == 0:
        idx = 0
        K_tile = K_desc.load([b_idx, j * BLOCK_N, 0])
        V_tile = V_desc.load([b_idx, j * BLOCK_N, 0])
        k_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = K_tile
        v_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = V_tile
        
        Q_tile = Q_desc.load([b_idx, j * BLOCK_M, 0])
        dO_tile = dO_desc.load([b_idx, j * BLOCK_M, 0])
        O_tile = O_desc.load([b_idx, j * BLOCK_M, 0])
        q_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = Q_tile
        do_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = dO_tile
        o_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = O_tile
        step = 1
    elif step == 1:
        idx = 1
        K_tile = K_desc.load([b_idx, j * BLOCK_N, 0])
        V_tile = V_desc.load([b_idx, j * BLOCK_N, 0])
        k_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = K_tile
        v_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = V_tile
        
        Q_tile = Q_desc.load([b_idx, (j + 1) * BLOCK_M, 0])
        dO_tile = dO_desc.load([b_idx, (j + 1) * BLOCK_M, 0])
        O_tile = O_desc.load([b_idx, (j + 1) * BLOCK_M, 0])
        q_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = Q_tile
        do_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = dO_tile
        o_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = O_tile
        step = (j + 1) % 2
    else:
        step = (j + 1) % 2
    
    K_tile = k_smem_0[step * 128 * 128 : (step + 1) * 128 * 128]
    V_tile = v_smem_0[step * 128 * 128 : (step + 1) * 128 * 128]
    K_tile = tl.reshape(K_tile, [128, 128])
    V_tile = tl.reshape(V_tile, [128, 128])
    
    dK_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    dV_acc = tl.zeros((BLOCK_N, 128), tl.float32)
    
    for i in range(j, num_blocks):
        if step == 0:
            idx = 0
            Q_tile = Q_desc.load([b_idx, i * BLOCK_M, 0])
            dO_tile = dO_desc.load([b_idx, i * BLOCK_M, 0])
            O_tile = O_desc.load([b_idx, i * BLOCK_M, 0])
            q_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = Q_tile
            do_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = dO_tile
            o_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = O_tile
            step = 1
        elif step == 1:
            idx = 1
            Q_tile = Q_desc.load([b_idx, i * BLOCK_M, 0])
            dO_tile = dO_desc.load([b_idx, i * BLOCK_M, 0])
            O_tile = O_desc.load([b_idx, i * BLOCK_M, 0])
            q_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = Q_tile
            do_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = dO_tile
            o_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = O_tile
            step = (j + 1) % 2
        else:
            step = (j + 1) % 2
        
        read_idx = step ^ 1
        Q_tile = tl.reshape(q_smem_0[read_idx * 128 * 128 : (read_idx + 1) * 128 * 128], [128, 128])
        dO_tile = tl.reshape(do_smem_0[read_idx * 128 * 128 : (read_idx + 1) * 128 * 128], [128, 128])
        O_tile = tl.reshape(o_smem_0[read_idx * 128 * 128 : (read_idx + 1) * 128 * 128], [128, 128])
        
        abs_i = i * BLOCK_M + row_idx
        L_vec = tl.load(L_ptr + b_idx * S_len + abs_i, mask=(abs_i < S_len), other=1e20)
        
        dO_tile_T = tl.permute(dO_tile, [1, 0])
        O_tile_T = tl.permute(O_tile, [1, 0])
        d_val = tl.dot(dO_tile_T, O_tile_T) / math.sqrt(128) / math.sqrt(128)
        d_smem[idx] = d_val / math.sqrt(128) / math.sqrt(128)
        
        S = tl.dot(Q_tile, K_tile.T) * tau
        
        if i == j:
            mask = (col_idx[None, :] <= row_idx[:, None]) & (abs_i[:, None] < S_len) & ((j * BLOCK_N + col_idx)[None, :] < S_len)
        else:
            mask = (abs_i[:, None] < S_len) & ((j * BLOCK_N + col_idx)[None, :] < S_len)
        
        P = tl.exp(S - L_vec[:, None])
        P = P * mask
        
        dP = tl.dot(dO_tile, V_tile.T)
        
        dS = P * (dP - d_smem[row_idx][:, None]) * tau
        dS = dS * mask
        
        dV_acc = tl.dot(P.T, dO_tile, dV_acc)
        dK_acc = tl.dot(dS.T, Q_tile, dK_acc)
        
        K_tile = k_smem_0[step * 128 * 128 : (step + 1) * 128 * 128]
        V_tile = v_smem_0[step * 128 * 128 : (step + 1) * 128 * 128]
        K_tile = tl.reshape(K_tile, [128, 128])
        V_tile = tl.reshape(V_tile, [128, 128])
        
    col_idx_d = tl.arange(0, 128)
    out_off_k = b_idx * S_len * 128 + j * BLOCK_N * 128
    
    dk_ptrs = dK_ptr + out_off_k + row_idx[:, None] * 128 + col_idx_d[None, :]
    dk_mask = (j * BLOCK_N + row_idx) < S_len
    tl.store(dk_ptrs, dK_acc.to(tl.bfloat16), mask=dk_mask[:, None])
    
    dv_ptrs = dV_ptr + out_off_k + row_idx[:, None] * 128 + col_idx_d[None, :]
    tl.store(dv_ptrs, dV_acc.to(tl.bfloat16), mask=dk_mask[:, None])


@triton.jit
def _dq_kernel(
    Q_desc, K_desc, V_desc, dO_desc, O_desc, L_ptr, D_ptr_unused, dQ_ptr,
    S_len, tau,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    b_idx = tl.program_id(1)
    num_blocks = tl.cdiv(S_len, BLOCK_N)
    
    row_idx = tl.arange(0, BLOCK_M)
    col_idx = tl.arange(0, BLOCK_N)
    
    q_smem_0 = tl.shared_memory(bfloat16, 128 * 128)
    do_smem_0 = tl.shared_memory(bfloat16, 128 * 128)
    o_smem_0 = tl.shared_memory(bfloat16, 128 * 128)
    k_smem_0 = tl.shared_memory(bfloat16, 2 * 128 * 128)
    v_smem_0 = tl.shared_memory(bfloat16, 2 * 128 * 128)
    
    d_smem = tl.shared_memory(float, 128)
    l_smem = tl.shared_memory(float, 128)
    
    Q_tile = Q_desc.load([b_idx, i * BLOCK_M, 0])
    dO_tile = dO_desc.load([b_idx, i * BLOCK_M, 0])
    O_tile = O_desc.load([b_idx, i * BLOCK_M, 0])
    q_smem_0[0 : 128 * 128] = Q_tile
    do_smem_0[0 : 128 * 128] = dO_tile
    o_smem_0[0 : 128 * 128] = O_tile
    
    Q_tile = tl.reshape(Q_tile, [128, 128])
    dO_tile = tl.reshape(dO_tile, [128, 128])
    O_tile = tl.reshape(O_tile, [128, 128])
    
    abs_i = i * BLOCK_M + row_idx
    L_vec = tl.load(L_ptr + b_idx * S_len + abs_i, mask=(abs_i < S_len), other=1e20)
    l_smem[0:128] = L_vec
    
    dO_tile_T = tl.permute(dO_tile, [1, 0])
    O_tile_T = tl.permute(O_tile, [1, 0])
    d_val = tl.dot(dO_tile_T, O_tile_T) / math.sqrt(128) / math.sqrt(128)
    d_smem[0:128] = d_val / math.sqrt(128) / math.sqrt(128)
    
    step = tl.int_param(0)
    
    dQ_acc = tl.zeros((BLOCK_M, 128), tl.float32)
    
    for j_inner in range(0, i + 1):
        if step == 0:
            idx = 0
            K_tile = K_desc.load([b_idx, j_inner * BLOCK_N, 0])
            V_tile = V_desc.load([b_idx, j_inner * BLOCK_N, 0])
            k_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = K_tile
            v_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = V_tile
            step = 1
        elif step == 1:
            idx = 1
            K_tile = K_desc.load([b_idx, j_inner * BLOCK_N, 0])
            V_tile = V_desc.load([b_idx, j_inner * BLOCK_N, 0])
            k_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = K_tile
            v_smem_0[step * 128 * 128 : (step + 1) * 128 * 128] = V_tile
            step = (i + 1) % 2
        else:
            step = (i + 1) % 2
        
        read_idx = step ^ 1
        K_tile = tl.reshape(k_smem_0[read_idx * 128 * 128 : (read_idx + 1) * 128 * 128], [128, 128])
        V_tile = tl.reshape(v_smem_0[read_idx * 128 * 128 : (read_idx + 1) * 128 * 128], [128, 128])
        
        S = tl.dot(Q_tile, K_tile.T) * tau
        
        if i == j_inner:
            mask = (col_idx[None, :] <= row_idx[:, None]) & (abs_i[:, None] < S_len) & ((j_inner * BLOCK_N + col_idx)[None, :] < S_len)
        else:
            mask = (abs_i[:, None] < S_len) & ((j_inner * BLOCK_N + col_idx)[None, :] < S_len)
        
        P = tl.exp(S - l_smem[row_idx][:, None])
        P = P * mask
        
        dP = tl.dot(dO_tile, V_tile.T)
        
        dS = P * (dP - d_smem[row_idx][:, None]) * tau
        dS = dS * mask
        
        dQ_acc = tl.dot(dS, K_tile, dQ_acc)
        
        Q_tile = tl.reshape(q_smem_0[0 : 128 * 128], [128, 128])
        dO_tile = tl.reshape(do_smem_0[0 : 128 * 128], [128, 128])
        
    col_idx_d = tl.arange(0, 128)
    out_off_q = b_idx * S_len * 128 + i * BLOCK_M * 128
    
    dq_ptrs = dQ_ptr + out_off_q + row_idx[:, None] * 128 + col_idx_d[None, :]
    dq_mask = (i * BLOCK_M + row_idx) < S_len
    tl.store(dq_ptrs, dQ_acc.to(tl.bfloat16), mask=dq_mask[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S_len, d = Q.shape
    tau = 1.0 / math.sqrt(d)
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, BLOCK_M, 128])
    K_desc = TensorDescriptor.from_tensor(K, [1, BLOCK_N, 128])
    V_desc = TensorDescriptor.from_tensor(V, [1, BLOCK_N, 128])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, BLOCK_M, 128])
    O_desc = TensorDescriptor.from_tensor(O, [1, BLOCK_M, 128])
    
    T_r = triton.cdiv(S_len, BLOCK_M)
    grid = (T_r, B * H)
    
    _dkdv_kernel[grid](
        Q_desc, K_desc, V_desc, dO_desc, O_desc, L, None, dK, dV,
        S_len, tau,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3
    )
    
    _dq_kernel[grid](
        Q_desc, K_desc, V_desc, dO_desc, O_desc, L, None, dQ,
        S_len, tau,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=4
    )