import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# -------------------------------------------------------------------
# Precompute Row-Sum Scaling Factor D
# -------------------------------------------------------------------
@triton.jit
def precompute_D_kernel(
    O, dO, d_D,
    S_len,
    s_B: tl.constexpr,
    s_H: tl.constexpr,
    s_S: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Precompute D_i = rowsum(dO_i * O_i) for all batch/head/sequence blocks."""
    pid = tl.program_id(0)
    
    total_S = S_len
    seq_idx = pid % (total_S // BLOCK)
    batch_head = pid // (total_S // BLOCK)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    idx = tl.arange(0, BLOCK)
    valid = (seq_idx * BLOCK + idx) < total_S
    
    o_ptr = O + batch_idx * s_B + head_idx * s_H + seq_idx * BLOCK * s_S
    do_ptr = dO + batch_idx * s_B + head_idx * s_H + seq_idx * BLOCK * s_S
    
    O_block = tl.load(o_ptr + idx[:, None] * s_S + tl.arange(0, 128)[None, :] * 1, 
                      mask=valid[:, None] & (tl.arange(0, 128)[None, :] < 128), other=0.0)
    dO_block = tl.load(do_ptr + idx[:, None] * s_S + tl.arange(0, 128)[None, :] * 1, 
                       mask=valid[:, None] & (tl.arange(0, 128)[None, :] < 128), other=0.0)
    
    D_block = (O_block * dO_block).sum(dim=1)
    
    # Fixed store indexing bug. Ensure correct mapping into generic contiguous format.
    d_ptr = d_D + batch_idx * (H * BLOCK) + head_idx * BLOCK + seq_idx * BLOCK
    tl.store(d_ptr + idx, D_block, mask=valid)


# -------------------------------------------------------------------
# dQ Kernel
# -------------------------------------------------------------------
@triton.jit
def dQ_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, d_D_desc, L_desc, dQ_desc,
    S_len,
    scale,
    H,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b_head = tl.program_id(0)
    i = tl.program_id(1)
    
    offset_m = b_head * S_len + i * BLOCK_SIZE
    offset_n = 0
    
    Q0 = Q_desc.load([offset_m, offset_n])
    Q1 = Q_desc.load([offset_m, offset_n + 64])
    
    dO0 = dO_desc.load([offset_m, offset_n])
    dO1 = dO_desc.load([offset_m, offset_n + 64])
    
    O0 = O_desc.load([offset_m, offset_n])
    O1 = O_desc.load([offset_m, offset_n + 64])
    
    O0_block_f32 = O0.to(tl.float32)
    O1_block_f32 = O1.to(tl.float32)
    dO0_block_f32 = dO0.to(tl.float32)
    dO1_block_f32 = dO1.to(tl.float32)
    
    D_pre = (O0_block_f32 * dO0_block_f32).sum(axis=1, keep_dims=True) + (O1_block_f32 * dO1_block_f32).sum(axis=1, keep_dims=True)
    
    L_tile = L_desc.load([offset_m, 0])
    D_tile = d_D_desc.load([offset_m, 0])
    
    L_reshaped = L_tile[:, 0][:, None]
    D_reshaped = D_tile[:, 0][:, None]
    
    dQ0_acc = tl.zeros((BLOCK_SIZE, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK_SIZE, 64), tl.float32)
    
    idx_q = tl.arange(0, BLOCK_SIZE)
    idx_k = tl.arange(0, BLOCK_SIZE)
    
    total_blocks = i + 1
    
    k0_p = K_desc.load([offset_m, offset_n])
    k1_p = K_desc.load([offset_m, offset_n + 64])
    v0_p = V_desc.load([offset_m, offset_n])
    v1_p = V_desc.load([offset_m, offset_n + 64])
    
    k0_0 = k0_p; k1_0 = k1_p; v0_0 = v0_p; v1_0 = v1_p
    k0_1 = k0_p; k1_1 = k1_p; v0_1 = v0_p; v1_1 = v1_p
    k0_2 = k0_p; k1_2 = k1_p; v0_2 = v0_p; v1_2 = v1_p
    k0_3 = k0_p; k1_3 = k1_p; v0_3 = v0_p; v1_3 = v1_p
    
    for stage in range(min(4, total_blocks)):
        j_offset = b_head * S_len + stage * BLOCK_SIZE
        k0_p = K_desc.load([j_offset, offset_n])
        k1_p = K_desc.load([j_offset, offset_n + 64])
        v0_p = V_desc.load([j_offset, offset_n])
        v1_p = V_desc.load([j_offset, offset_n + 64])
        
        if stage == 0: k0_0 = k0_p; k1_0 = k1_p; v0_0 = v0_p; v1_0 = v1_p
        elif stage == 1: k0_1 = k0_p; k1_1 = k1_p; v0_1 = v0_p; v1_1 = v1_p
        elif stage == 2: k0_2 = k0_p; k1_2 = k1_p; v0_2 = v0_p; v1_2 = v1_p
        elif stage == 3: k0_3 = k0_p; k1_3 = k1_p; v0_3 = v0_p; v1_3 = v1_p

    for actual_j in range(total_blocks):
        next_stage = actual_j % 4
        if actual_j + 4 < total_blocks:
            next_j_offset = b_head * S_len + (actual_j + 4) * BLOCK_SIZE
            k0_p = K_desc.load([next_j_offset, offset_n])
            k1_p = K_desc.load([next_j_offset, offset_n + 64])
            v0_p = V_desc.load([next_j_offset, offset_n])
            v1_p = V_desc.load([next_j_offset, offset_n + 64])
            
            if next_stage == 0:
                k0_next = k0_p; k1_next = k1_p; v0_next = v0_p; v1_next = v1_p
            elif next_stage == 1:
                k0_next = k0_p; k1_next = k1_p; v0_next = v0_p; v1_next = v1_p
            elif next_stage == 2:
                k0_next = k0_p; k1_next = k1_p; v0_next = v0_p; v1_next = v1_p
            elif next_stage == 3:
                k0_next = k0_p; k1_next = k1_p; v0_next = v0_p; v1_next = v1_p
        
        k0 = k0_0 if (actual_j % 4) == 0 else k0_1 if (actual_j % 4) == 1 else k0_2 if (actual_j % 4) == 2 else k0_3
        k1 = k1_0 if (actual_j % 4) == 0 else k1_1 if (actual_j % 4) == 1 else k1_2 if (actual_j % 4) == 2 else k1_3
        v0 = v0_0 if (actual_j % 4) == 0 else v0_1 if (actual_j % 4) == 1 else v0_2 if (actual_j % 4) == 2 else v0_3
        v1 = v1_0 if (actual_j % 4) == 0 else v1_1 if (actual_j % 4) == 1 else v1_2 if (actual_j % 4) == 2 else v1_3
        
        j = actual_j * BLOCK_SIZE
        
        S = tl.dot(Q0, k0.T, input_precision="ieee") + tl.dot(Q1, k1.T, input_precision="ieee")
        dP = tl.dot(dO0, v0.T, input_precision="ieee") + tl.dot(dO1, v1.T, input_precision="ieee")
        
        valid = ((i * BLOCK_SIZE + idx_q[:, None]) >= (j + idx_k[None, :]))
        
        S_scaled = tl.where(valid, S * scale, float("-inf"))
        P = tl.math.exp(S_scaled - L_reshaped)
        P = tl.where(valid, P, 0.0)
        
        dS = P * (dP - D_reshaped) * scale
        dS = tl.where(valid, dS, 0.0)
        
        dS_local = dS.to(tl.bfloat16)
        
        dQ0_acc = tl.dot(dS_local, k0, dQ0_acc, input_precision="ieee")
        dQ1_acc = tl.dot(dS_local, k1, dQ1_acc, input_precision="ieee")
        
        if (actual_j % 4) == 0: k0_0 = k0_next; k1_0 = k1_next; v0_0 = v0_next; v1_0 = v1_next
        elif (actual_j % 4) == 1: k0_1 = k0_next; k1_1 = k1_next; v0_1 = v0_next; v1_1 = v1_next
        elif (actual_j % 4) == 2: k0_2 = k0_next; k1_2 = k1_next; v0_2 = v0_next; v1_2 = v1_next
        elif (actual_j % 4) == 3: k0_3 = k0_next; k1_3 = k1_next; v0_3 = v0_next; v1_3 = v1_next
        
    dQ0_out = dQ0_acc.to(tl.bfloat16)
    dQ1_out = dQ1_acc.to(tl.bfloat16)
    
    dQ_desc.store([offset_m, offset_n], dQ0_out)
    dQ_desc.store([offset_m, offset_n + 64], dQ1_out)


# -------------------------------------------------------------------
# dK / dV Kernel
# -------------------------------------------------------------------
@triton.jit
def dK_dV_kernel(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, d_D_desc, L_desc, dK_desc, dV_desc,
    S_len,
    scale,
    H,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b_head = tl.program_id(0)
    j = tl.program_id(1)
    
    offset_m = b_head * S_len + j * BLOCK_SIZE
    offset_n = 0
    
    K0 = K_desc.load([offset_m, offset_n])
    K1 = K_desc.load([offset_m, offset_n + 64])
    
    V0 = V_desc.load([offset_m, offset_n])
    V1 = V_desc.load([offset_m, offset_n + 64])
    
    dK0_acc = tl.zeros((BLOCK_SIZE, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK_SIZE, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK_SIZE, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK_SIZE, 64), tl.float32)
    
    idx_q = tl.arange(0, BLOCK_SIZE)
    idx_k = tl.arange(0, BLOCK_SIZE)
    
    num_blocks = (S_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = num_blocks - j
    
    Q0_p = Q_desc.load([offset_m, offset_n])
    Q1_p = Q_desc.load([offset_m, offset_n + 64])
    dO0_p = dO_desc.load([offset_m, offset_n])
    dO1_p = dO_desc.load([offset_m, offset_n + 64])
    O0_p = O_desc.load([offset_m, offset_n])
    O1_p = O_desc.load([offset_m, offset_n + 64])
    
    Q0_0 = Q0_p; Q1_0 = Q1_p; dO0_0 = dO0_p; dO1_0 = dO1_p; O0_0 = O0_p; O1_0 = O1_p
    Q0_1 = Q0_p; Q1_1 = Q1_p; dO0_1 = dO0_p; dO1_1 = dO1_p; O0_1 = O0_p; O1_1 = O1_p
    Q0_2 = Q0_p; Q1_2 = Q1_p; dO0_2 = dO0_p; dO1_2 = dO1_p; O0_2 = O0_p; O1_2 = O1_p
    Q0_3 = Q0_p; Q1_3 = Q1_p; dO0_3 = dO0_p; dO1_3 = dO1_p; O0_3 = O0_p; O1_3 = O1_p
    
    for stage in range(min(4, total_blocks)):
        i_offset = b_head * S_len + (j + stage) * BLOCK_SIZE
        Q0_p = Q_desc.load([i_offset, offset_n])
        Q1_p = Q_desc.load([i_offset, offset_n + 64])
        dO0_p = dO_desc.load([i_offset, offset_n])
        dO1_p = dO_desc.load([i_offset, offset_n + 64])
        O0_p = O_desc.load([i_offset, offset_n])
        O1_p = O_desc.load([i_offset, offset_n + 64])
        
        if stage == 0: Q0_0 = Q0_p; Q1_0 = Q1_p; dO0_0 = dO0_p; dO1_0 = dO1_p; O0_0 = O0_p; O1_0 = O1_p
        elif stage == 1: Q0_1 = Q0_p; Q1_1 = Q1_p; dO0_1 = dO0_p; dO1_1 = dO1_p; O0_1 = O0_p; O1_1 = O1_p
        elif stage == 2: Q0_2 = Q0_p; Q1_2 = Q1_p; dO0_2 = dO0_p; dO1_2 = dO1_p; O0_2 = O0_p; O1_2 = O1_p
        elif stage == 3: Q0_3 = Q0_p; Q1_3 = Q1_p; dO0_3 = dO0_p; dO1_3 = dO1_p; O0_3 = O0_p; O1_3 = O1_p
        
    for actual_i in range(total_blocks):
        next_stage = actual_i % 4
        if actual_i + 4 < total_blocks:
            next_i = j + actual_i + 4
            i_offset = b_head * S_len + next_i * BLOCK_SIZE
            Q0_p = Q_desc.load([i_offset, offset_n])
            Q1_p = Q_desc.load([i_offset, offset_n + 64])
            dO0_p = dO_desc.load([i_offset, offset_n])
            dO1_p = dO_desc.load([i_offset, offset_n + 64])
            O0_p = O_desc.load([i_offset, offset_n])
            O1_p = O_desc.load([i_offset, offset_n + 64])
            
            if next_stage == 0:
                Q0_next = Q0_p; Q1_next = Q1_p; dO0_next = dO0_p; dO1_next = dO1_p; O0_next = O0_p; O1_next = O1_p
            elif next_stage == 1:
                Q0_next = Q0_p; Q1_next = Q1_p; dO0_next = dO0_p; dO1_next = dO1_p; O0_next = O0_p; O1_next = O1_p
            elif next_stage == 2:
                Q0_next = Q0_p; Q1_next = Q1_p; dO0_next = dO0_p; dO1_next = dO1_p; O0_next = O0_p; O1_next = O1_p
            elif next_stage == 3:
                Q0_next = Q0_p; Q1_next = Q1_p; dO0_next = dO0_p; dO1_next = dO1_p; O0_next = O0_p; O1_next = O1_p
                
        Q0 = Q0_0 if (actual_i % 4) == 0 else Q0_1 if (actual_i % 4) == 1 else Q0_2 if (actual_i % 4) == 2 else Q0_3
        Q1 = Q1_0 if (actual_i % 4) == 0 else Q1_1 if (actual_i % 4) == 1 else Q1_2 if (actual_i % 4) == 2 else Q1_3
        dO0 = dO0_0 if (actual_i % 4) == 0 else dO0_1 if (actual_i % 4) == 1 else dO0_2 if (actual_i % 4) == 2 else dO0_3
        dO1 = dO1_0 if (actual_i % 4) == 0 else dO1_1 if (actual_i % 4) == 1 else dO1_2 if (actual_i % 4) == 2 else dO1_3
        O0 = O0_0 if (actual_i % 4) == 0 else O0_1 if (actual_i % 4) == 1 else O0_2 if (actual_i % 4) == 2 else O0_3
        O1 = O1_0 if (actual_i % 4) == 0 else O1_1 if (actual_i % 4) == 1 else O1_2 if (actual_i % 4) == 2 else O1_3
        
        i = j + actual_i
        
        O0_block_f32 = O0.to(tl.float32)
        O1_block_f32 = O1.to(tl.float32)
        dO0_block_f32 = dO0.to(tl.float32)
        dO1_block_f32 = dO1.to(tl.float32)
        
        D_pre = (O0_block_f32 * dO0_block_f32).sum(axis=1, keep_dims=True) + (O1_block_f32 * dO1_block_f32).sum(axis=1, keep_dims=True)
        
        i_offset = b_head * S_len + i * BLOCK_SIZE
        L_tile = L_desc.load([i_offset, 0])
        D_tile = d_D_desc.load([i_offset, 0])
        
        L_reshaped = L_tile[:, 0][:, None]
        D_reshaped = D_tile[:, 0][:, None]
        
        S = tl.dot(Q0, K0.T, input_precision="ieee") + tl.dot(Q1, K1.T, input_precision="ieee")
        dP = tl.dot(dO0, V0.T, input_precision="ieee") + tl.dot(dO1, V1.T, input_precision="ieee")
        
        valid = ((i * BLOCK_SIZE + idx_q[:, None]) >= (j * BLOCK_SIZE + idx_k[None, :]))
        
        S_scaled = tl.where(valid, S * scale, float("-inf"))
        P = tl.math.exp(S_scaled - L_reshaped)
        P = tl.where(valid, P, 0.0)
        
        dS = P * (dP - D_reshaped) * scale
        dS = tl.where(valid, dS, 0.0)
        
        dS_T = dS.T.to(tl.bfloat16)
        
        dK0_acc = tl.dot(dS_T, Q0, dK0_acc, input_precision="ieee")
        dK1_acc = tl.dot(dS_T, Q1, dK1_acc, input_precision="ieee")
        
        P_T = P.T.to(tl.bfloat16)
        
        dV0_acc = tl.dot(P_T, dO0, dV0_acc, input_precision="ieee")
        dV1_acc = tl.dot(P_T, dO1, dV1_acc, input_precision="ieee")
        
        if (actual_i % 4) == 0: Q0_0 = Q0_next; Q1_0 = Q1_next; dO0_0 = dO0_next; dO1_0 = dO1_next; O0_0 = O0_next; O1_0 = O1_next
        elif (actual_i % 4) == 1: Q0_1 = Q0_next; Q1_1 = Q1_next; dO0_1 = dO0_next; dO1_1 = dO1_next; O0_1 = O0_next; O1_1 = O1_next
        elif (actual_i % 4) == 2: Q0_2 = Q0_next; Q1_2 = Q1_next; dO0_2 = dO0_next; dO1_2 = dO1_next; O0_2 = O0_next; O1_2 = O1_next
        elif (actual_i % 4) == 3: Q0_3 = Q0_next; Q1_3 = Q1_next; dO0_3 = dO0_next; dO1_3 = dO1_next; O0_3 = O0_next; O1_3 = O1_next
        
    dK0_out = dK0_acc.to(tl.bfloat16)
    dK1_out = dK1_acc.to(tl.bfloat16)
    dV0_out = dV0_acc.to(tl.bfloat16)
    dV1_out = dV1_acc.to(tl.bfloat16)
    
    dK_desc.store([offset_m, offset_n], dK0_out)
    dK_desc.store([offset_m, offset_n + 64], dK1_out)
    dV_desc.store([offset_m, offset_n], dV0_out)
    dV_desc.store([offset_m, offset_n + 64], dV1_out)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute causal multi-head attention backward (dQ, dK, dV) using TMA and optimally staged pipelines."""
    torch.cuda.set_device(Q.device)
    
    B, H, S_len, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    BLOCK_SIZE = 128
    BLOCK_D = 128
    
    s_B = Q.stride()[0]
    s_H = Q.stride()[1]
    s_S = Q.stride()[2]
    
    num_blocks = (S_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    D_tensor = torch.empty((B, H, S_len), device=Q.device, dtype=torch.float32)
    
    total_elems = B * H * S_len
    grid_D = (triton.cdiv(total_elems, 128),)
    precompute_D_kernel[grid_D](
        O, dO, D_tensor, S_len, s_B, s_H, s_S, H, BLOCK=128, num_warps=4
    )
    
    Q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_SIZE, BLOCK_D])
    K_desc = TensorDescriptor.from_tensor(K, [BLOCK_SIZE, BLOCK_D])
    V_desc = TensorDescriptor.from_tensor(V, [BLOCK_SIZE, BLOCK_D])
    O_desc = TensorDescriptor.from_tensor(O, [BLOCK_SIZE, BLOCK_D])
    dO_desc = TensorDescriptor.from_tensor(dO, [BLOCK_SIZE, BLOCK_D])
    d_D_desc = TensorDescriptor.from_tensor(D_tensor, [BLOCK_SIZE, BLOCK_D])
    L_desc = TensorDescriptor.from_tensor(L, [BLOCK_SIZE, BLOCK_D])
    
    dQ_desc = TensorDescriptor.from_tensor(dQ, [BLOCK_SIZE, BLOCK_D])
    dK_desc = TensorDescriptor.from_tensor(dK, [BLOCK_SIZE, BLOCK_D])
    dV_desc = TensorDescriptor.from_tensor(dV, [BLOCK_SIZE, BLOCK_D])
    
    grid = (B * H, num_blocks)
    
    dQ_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, d_D_desc, L_desc, dQ_desc,
        S_len, scale, H, BLOCK_SIZE, BLOCK_D, num_warps=4, num_stages=4
    )
    
    dK_dV_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, d_D_desc, L_desc, dK_desc, dV_desc,
        S_len, scale, H, BLOCK_SIZE, BLOCK_D, num_warps=4, num_stages=4
    )