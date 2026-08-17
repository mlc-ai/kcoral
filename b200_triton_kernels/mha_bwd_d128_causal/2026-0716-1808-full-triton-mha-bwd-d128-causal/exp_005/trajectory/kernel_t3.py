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
    
    # Compute scaling factor block locally 
    D_pre = O_tile * dO_tile
    D_pre = D_pre.sum(axis=1, keep_dims=True)
    
    l_base = L + batch_idx * l_s_B + head_idx * l_s_H + i * BLOCK_SIZE
    L_tile = tl.load(l_base + idx_q, mask=(i * BLOCK_SIZE + idx_q) < S_len, other=0.0)
    L_reshaped = L_tile[:, None]
    D_reshaped = D_pre[:, :, None]
    D_reshaped_bf16 = D_reshaped.to(tl.bfloat16)
    
    dQ_acc = tl.zeros((BLOCK_SIZE, BLOCK_D), tl.float32)
    
    for j in range(0, i + 1):
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
        
        S = tl.dot(Q_tile, K_tile.T, input_precision="tf32", out_dtype=tl.float32)
        dP = tl.dot(dO_tile, V_tile.T, input_precision="tf32", out_dtype=tl.float32)
        
        valid = ((i * BLOCK_SIZE + idx_q[:, None]) >= (j * BLOCK_SIZE + idx_k[None, :]))
        
        S_scaled = tl.where(valid, S * scale, float("-inf"))
        P = tl.exp(S_scaled - L_reshaped)
        P = tl.where(valid, P, 0.0)
        
        dP_fp32 = dP.to(tl.float32)
        
        dS = P * (dP_fp32 - D_reshaped) * scale
        dS = tl.where(valid, dS, 0.0)
        
        K_tile_to_d = K_tile # [key_rows, d]
        dS_local = dS.to(tl.bfloat16)
        
        dQ_acc = tl.dot(dS_local, K_tile_to_d, input_precision="ieee", out_dtype=tl.float32)
        
    dq_base = dQ + batch_idx * s_B + head_idx * s_H + i * BLOCK_SIZE * s_S
    tl.store(
        dq_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
        dQ_acc.to(tl.bfloat16),
        mask=((i * BLOCK_SIZE + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :]
    )


# -------------------------------------------------------------------
# dK Kernel
# -------------------------------------------------------------------
@triton.jit
def dK_kernel(
    Q, K, V, O, dO, L, dK,
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
    """Compute the accumulated contributions for the K-block gradient."""
    b_head = tl.program_id(0)
    j = tl.program_id(1)
    
    batch_idx = b_head // H
    head_idx = b_head % H
    
    idx_q = tl.arange(0, BLOCK_SIZE)
    idx_k = tl.arange(0, BLOCK_SIZE)
    idx_d = tl.arange(0, BLOCK_D)
    
    k_base = K + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
    
    K_tile = tl.load(
        k_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
        mask=((j * BLOCK_SIZE + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
        other=0.0
    )
    
    dK_acc = tl.zeros((BLOCK_SIZE, BLOCK_D), tl.float32)
    
    num_blocks = (S_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for i in range(j, num_blocks):
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
        v_base = V + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
        V_tile = tl.load(
            v_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
            mask=((j * BLOCK_SIZE + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0
        )
        
        # Compute scaling factor block locally 
        D_pre = O_tile * dO_tile
        D_pre = D_pre.sum(axis=1, keep_dims=True)
        
        l_base = L + batch_idx * l_s_B + head_idx * l_s_H + i * BLOCK_SIZE
        L_tile = tl.load(l_base + idx_q, mask=(i * BLOCK_SIZE + idx_q) < S_len, other=0.0)
        L_reshaped = L_tile[:, None]
        D_reshaped = D_pre[:, :, None]
        D_reshaped_bf16 = D_reshaped.to(tl.bfloat16)
        
        S = tl.dot(Q_tile, K_tile.T, input_precision="tf32", out_dtype=tl.float32)
        dP = tl.dot(dO_tile, V_tile.T, input_precision="tf32", out_dtype=tl.float32)
        
        valid = ((i * BLOCK_SIZE + idx_q[:, None]) >= (j * BLOCK_SIZE + idx_k[None, :]))
        
        S_scaled = tl.where(valid, S * scale, float("-inf"))
        P = tl.exp(S_scaled - L_reshaped)
        P = tl.where(valid, P, 0.0)
        
        dP_fp32 = dP.to(tl.float32)
        
        dS = P * (dP_fp32 - D_reshaped) * scale
        dS = tl.where(valid, dS, 0.0)
        
        Q_tile_to_d = Q_tile # [query_rows, d]
        dS_local_T = dS.T.to(tl.bfloat16)
        
        dK_acc = tl.dot(dS_local_T, Q_tile_to_d, input_precision="ieee", out_dtype=tl.float32)
        
    dk_base = dK + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
    
    tl.store(
        dk_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
        dK_acc.to(tl.bfloat16),
        mask=((j * BLOCK_SIZE + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :]
    )


# -------------------------------------------------------------------
# dV Kernel
# -------------------------------------------------------------------
@triton.jit
def dV_kernel(
    Q, K, V, O, dO, L, dV,
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
    """Compute the accumulated contributions for the V-block gradient."""
    b_head = tl.program_id(0)
    j = tl.program_id(1)
    
    batch_idx = b_head // H
    head_idx = b_head % H
    
    idx_q = tl.arange(0, BLOCK_SIZE)
    idx_k = tl.arange(0, BLOCK_SIZE)
    idx_d = tl.arange(0, BLOCK_D)
    
    v_base = V + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
    
    V_tile = tl.load(
        v_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
        mask=((j * BLOCK_SIZE + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
        other=0.0
    )
    
    dV_acc = tl.zeros((BLOCK_SIZE, BLOCK_D), tl.float32)
    
    num_blocks = (S_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for i in range(j, num_blocks):
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
        k_base = K + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
        K_tile = tl.load(
            k_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
            mask=((j * BLOCK_SIZE + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0
        )
        
        # Compute scaling factor block locally 
        D_pre = O_tile * dO_tile
        D_pre = D_pre.sum(axis=1, keep_dims=True)
        
        l_base = L + batch_idx * l_s_B + head_idx * l_s_H + i * BLOCK_SIZE
        L_tile = tl.load(l_base + idx_q, mask=(i * BLOCK_SIZE + idx_q) < S_len, other=0.0)
        L_reshaped = L_tile[:, None]
        D_reshaped = D_pre[:, :, None]
        D_reshaped_bf16 = D_reshaped.to(tl.bfloat16)
        
        S = tl.dot(Q_tile, K_tile.T, input_precision="tf32", out_dtype=tl.float32)
        dP = tl.dot(dO_tile, V_tile.T, input_precision="tf32", out_dtype=tl.float32)
        
        valid = ((i * BLOCK_SIZE + idx_q[:, None]) >= (j * BLOCK_SIZE + idx_k[None, :]))
        
        S_scaled = tl.where(valid, S * scale, float("-inf"))
        P = tl.exp(S_scaled - L_reshaped)
        P = tl.where(valid, P, 0.0)
        
        dP_fp32 = dP.to(tl.float32)
        
        dS = P * (dP_fp32 - D_reshaped) * scale
        dS = tl.where(valid, dS, 0.0)
        
        dO_tile_to_d = dO_tile # [query_rows, d]
        P_local_T = P.T.to(tl.bfloat16)
        
        dV_acc = tl.dot(P_local_T, dO_tile_to_d, input_precision="ieee", out_dtype=tl.float32)
        
    dv_base = dV + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
    
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
        num_warps=8, num_stages=4
    )
    
    dK_kernel[grid](
        Q, K, V, O, dO, L, dK, S_len, scale, H,
        BLOCK_SIZE, BLOCK_D, s_B, s_H, s_S, l_s_B, l_s_H,
        num_warps=8, num_stages=4
    )
    
    dV_kernel[grid](
        Q, K, V, O, dO, L, dV, S_len, scale, H,
        BLOCK_SIZE, BLOCK_D, s_B, s_H, s_S, l_s_B, l_s_H,
        num_warps=8, num_stages=4
    )