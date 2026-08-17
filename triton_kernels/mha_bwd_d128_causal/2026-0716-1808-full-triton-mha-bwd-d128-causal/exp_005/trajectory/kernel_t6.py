import torch
import triton
import triton.language as tl


# -------------------------------------------------------------------
# dQ Kernel
# -------------------------------------------------------------------
@triton.jit
def dQ_kernel(
    Q, K, V, O, dO, L, dQ,
    S_len,
    scale,
    H,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    s_B: tl.constexpr,
    s_H: tl.constexpr,
    s_S: tl.constexpr,
    l_s_B: tl.constexpr,
    l_s_H: tl.constexpr,
):
    """Compute the contribution to the Q-block gradient."""
    b_head = tl.program_id(0)
    i = tl.program_id(1)
    
    batch_idx = b_head // H
    head_idx = b_head % H
    
    idx_q = tl.arange(0, BLOCK_SIZE)
    idx_k = tl.arange(0, BLOCK_SIZE)
    idx_d = tl.arange(0, BLOCK_D)
    
    q_base = Q + batch_idx * s_B + head_idx * s_H + i * BLOCK_SIZE * s_S
    do_base = dO + batch_idx * s_B + head_idx * s_H + i * BLOCK_SIZE * s_S
    o_base = O + batch_idx * s_B + head_idx * s_H + i * BLOCK_SIZE * s_S
    
    Q_tile = tl.load(
        q_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
        mask=((i * BLOCK_SIZE + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
        other=0.0
    )
    dO_tile = tl.load(
        do_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
        mask=((i * BLOCK_SIZE + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
        other=0.0
    )
    O_tile = tl.load(
        o_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
        mask=((i * BLOCK_SIZE + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
        other=0.0
    )
    
    # Fixed precision computation: cast to fp32 explicitly before reduction
    O_block_f32 = O_tile.to(tl.float32)
    dO_block_f32 = dO_tile.to(tl.float32)
    D_pre = (O_block_f32 * dO_block_f32).sum(axis=1, keep_dims=True)
    
    l_base = L + batch_idx * l_s_B + head_idx * l_s_H + i * BLOCK_SIZE
    L_tile = tl.load(l_base + idx_q, mask=(i * BLOCK_SIZE + idx_q) < S_len, other=0.0)
    L_reshaped = L_tile[:, None]
    D_reshaped = D_pre
    
    dQ_acc = tl.full((BLOCK_SIZE, BLOCK_D), 0.0, tl.bfloat16)
    
    for block_j in range(0, i + 1):
        j = block_j * BLOCK_SIZE
        k_base = K + batch_idx * s_B + head_idx * s_H + j * s_S
        v_base = V + batch_idx * s_B + head_idx * s_H + j * s_S
        
        K_tile = tl.load(
            k_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
            mask=((j + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0
        )
        V_tile = tl.load(
            v_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
            mask=((j + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0
        )
        
        # Utilize tf32 inputs mapping strictly to native IEEE 754 expectations 
        S = tl.dot(Q_tile, K_tile.T, input_precision="ieee", out_dtype=tl.float32)
        dP = tl.dot(dO_tile, V_tile.T, input_precision="ieee", out_dtype=tl.float32)
        
        scale_s32 = scale.to(tl.int32)
        S_scaled = (S * scale_s32) >> 8
        
        P_unmasked = tl.math.exp(S_scaled - L_reshaped)
        valid_row = ((i * BLOCK_SIZE + idx_q[:, None]) >= (j + idx_k[None, :]))
        P_block = tl.where(valid_row, P_unmasked, 0.0)
        
        dP_block = dP_raw
        D_reshaped = D_block[:, None].to(tl.bfloat16)
        
        dS = P_block * (dP_block - D_reshaped) * scale_s32
        dS = tl.where(valid_row, dS, 0.0)
        
        K_tile_to_d = K_tile 
        dS_local = dS.to(tl.bfloat16)
        
        dQ_acc = tl.dot(dS_local, K_tile_to_d, input_precision="ieee", out_dtype=tl.bfloat16)
        
    dq_base = dQ + batch_idx * s_B + head_idx * s_H + i * BLOCK_SIZE * s_S
    tl.store(
        dq_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
        dQ_acc.to(tl.bfloat16),
        mask=((i * BLOCK_SIZE + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :]
    )


# -------------------------------------------------------------------
# dK / dV Kernel
# -------------------------------------------------------------------
@triton.jit
def dK_dV_kernel(
    Q, K, V, O, dO, L, dK, dV,
    S_len,
    scale,
    H,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    s_B: tl.constexpr,
    s_H: tl.constexpr,
    s_S: tl.constexpr,
    l_s_B: tl.constexpr,
    l_s_H: tl.constexpr,
):
    """Compute the accumulated contributions for the K-block and V-block gradients."""
    b_head = tl.program_id(0)
    j = tl.program_id(1)
    
    batch_idx = b_head // H
    head_idx = b_head % H
    
    idx_q = tl.arange(0, BLOCK_SIZE)
    idx_k = tl.arange(0, BLOCK_SIZE)
    idx_d = tl.arange(0, BLOCK_D)
    
    k_base = K + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
    v_base = V + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
    
    K_tile = tl.load(
        k_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
        mask=((j * BLOCK_SIZE + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
        other=0.0
    )
    V_tile = tl.load(
        v_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
        mask=((j * BLOCK_SIZE + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
        other=0.0
    )
    
    dK_acc = tl.full((BLOCK_SIZE, BLOCK_D), 0.0, tl.bfloat16)
    dV_acc = tl.full((BLOCK_SIZE, BLOCK_D), 0.0, tl.bfloat16)
    
    num_blocks = (S_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for block_i in range(j, num_blocks):
        i = block_i * BLOCK_SIZE
        q_base = Q + batch_idx * s_B + head_idx * s_H + i * s_S
        do_base = dO + batch_idx * s_B + head_idx * s_H + i * s_S
        o_base = O + batch_idx * s_B + head_idx * s_H + i * s_S
        
        Q_tile = tl.load(
            q_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
            mask=((i + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0
        )
        dO_tile = tl.load(
            do_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
            mask=((i + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0
        )
        O_tile = tl.load(
            o_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
            mask=((i + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0
        )
        
        # Fixed precision computation: cast to fp32 explicitly before reduction
        O_block_f32 = O_tile.to(tl.float32)
        dO_block_f32 = dO_tile.to(tl.float32)
        D_pre = (O_block_f32 * dO_block_f32).sum(axis=1, keep_dims=True)
        
        l_base = L + batch_idx * l_s_B + head_idx * l_s_H + i
        L_tile = tl.load(l_base + idx_q, mask=(i + idx_q) < S_len, other=0.0)
        L_reshaped = L_tile[:, None]
        D_reshaped = D_pre
        
        scale_s32 = scale.to(tl.int32)
        
        S = tl.dot(Q_tile, K_tile.T, input_precision="ieee", out_dtype=tl.float32)
        dP = tl.dot(dO_tile, V_tile.T, input_precision="ieee", out_dtype=tl.float32)
        
        valid_row = ((i + idx_q[:, None]) >= (j * BLOCK_SIZE + idx_k[None, :]))
        
        S_scaled = (S * scale_s32) >> 8
        P_unmasked = tl.math.exp(S_scaled - L_reshaped)
        P_block = tl.where(valid_row, P_unmasked, 0.0)
        
        dP_block = dP_raw
        
        D_block = D_pre.sum(dim=-1, keepdim=True) 
        D_reshaped = D_block[:, None].to(tl.bfloat16)
        
        dS = P_block * (dP_block - D_reshaped) * scale_s32
        dS = tl.where(valid_row, dS, 0.0)
        
        Q_tile_to_d = Q_tile 
        dS_local_T = dS.T.to(tl.bfloat16)
        
        dK_acc = tl.dot(dS_local_T, Q_tile_to_d, input_precision="ieee", out_dtype=tl.bfloat16)
        
        dO_tile_to_d = dO_tile 
        P_local_T = P_block.T.to(tl.bfloat16)
        
        dV_acc = tl.dot(P_local_T, dO_tile_to_d, input_precision="ieee", out_dtype=tl.bfloat16)
        
    dk_base = dK + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
    dv_base = dV + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
    
    tl.store(
        dk_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
        dK_acc.to(tl.bfloat16),
        mask=((j * BLOCK_SIZE + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :]
    )
    tl.store(
        dv_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
        dV_acc.to(tl.bfloat16),
        mask=((j * BLOCK_SIZE + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :]
    )


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute causal multi-head attention backward (dQ, dK, dV) using explicit TMA-free loads."""
    torch.cuda.set_device(Q.device)
    
    B, H, S_len, d = Q.shape
    scale = 1.0 / (d ** 0.5)
    
    BLOCK_SIZE = 128
    BLOCK_D = 128
    
    s_B = Q.stride()[0]
    s_H = Q.stride()[1]
    s_S = Q.stride()[2]
    l_s_B = L.stride()[0]
    l_s_H = L.stride()[1]
    
    num_blocks = (S_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (B * H, num_blocks)
    
    dQ_kernel[grid](
        Q, K, V, O, dO, L, dQ, S_len, scale, H,
        BLOCK_SIZE, BLOCK_D, s_B, s_H, s_S, l_s_B, l_s_H,
        num_warps=4, num_stages=1
    )
    
    dK_dV_kernel[grid](
        Q, K, V, O, dO, L, dK, dV, S_len, scale, H,
        BLOCK_SIZE, BLOCK_D, s_B, s_H, s_S, l_s_B, l_s_H,
        num_warps=4, num_stages=1
    )