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
    
    # Decompose the flattened B*H grid coordinate into concrete batch / head indices
    batch_idx = b_head // H
    head_idx = b_head % H
    
    idx_q = tl.arange(0, BLOCK_SIZE)
    idx_k = tl.arange(0, BLOCK_SIZE)
    idx_d = tl.arange(0, BLOCK_D)
    
    q_base = Q + batch_idx * s_B + head_idx * s_H + i * BLOCK_SIZE * s_S
    do_base = dO + batch_idx * s_B + head_idx * s_H + i * BLOCK_SIZE * s_S
    o_base = O + batch_idx * s_B + head_idx * s_H + i * BLOCK_SIZE * s_S
    
    # Load the entire i-th block of Q, dO, and O.
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
    
    # Correction term D_i = rowsum(dO_i * O_i) over the feature dimension.
    D_pre = O_tile * dO_tile
    D_pre = D_pre.sum(axis=1, keep_dims=True)
    
    l_base = L + batch_idx * l_s_B + head_idx * l_s_H + i * BLOCK_SIZE
    L_tile = tl.load(l_base + idx_q, mask=(i * BLOCK_SIZE + idx_q) < S_len, other=0.0)
    L_reshaped = L_tile[:, None]
    D_reshaped = D_pre
    
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
        
        # Standard GEMMs for the unscaled attention scores and the upstream dP.
        S_local = tl.dot(Q_tile, K_tile.T, input_precision="ieee")
        dP_local = tl.dot(dO_tile, V_tile.T, input_precision="ieee")
        
        valid = ((i * BLOCK_SIZE + idx_q[:, None]) >= (j * BLOCK_SIZE + idx_k[None, :]))
        
        S_local = tl.where(valid, S_local * scale, float("-inf"))
        P_local = tl.exp(S_local - L_reshaped)
        P_local = tl.where(valid, P_local, 0.0)
        
        # Critical Fixes Applied Here:
        # 1. `dS_local` explicitly retains shape `[BLOCK_SIZE, BLOCK_SIZE]` mapping `[query, key]`.
        # 2. We multiply `dS` by the causal `valid` mask immediately to guarantee numerically 
        #    safe subsequent reductions regardless of compiler optimizations.
        dS_local = P_local * (dP_local - D_reshaped) * scale
        dS_local = tl.where(valid, dS_local, 0.0)
        
        # Fixed layout accumulation passing `dS_local` directly into the `tl.dot` without transpose.
        dQ_acc = tl.dot(dS_local, K_tile, dQ_acc)
        
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
    
    dK_acc = tl.zeros((BLOCK_SIZE, BLOCK_D), tl.float32)
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
        
        D_pre = O_tile * dO_tile
        D_pre = D_pre.sum(axis=1, keep_dims=True)
        
        l_base = L + batch_idx * l_s_B + head_idx * l_s_H + i * BLOCK_SIZE
        L_tile = tl.load(l_base + idx_q, mask=(i * BLOCK_SIZE + idx_q) < S_len, other=0.0)
        L_reshaped = L_tile[:, None]
        D_reshaped = D_pre
        
        S_local = tl.dot(Q_tile, K_tile.T, input_precision="ieee")
        dP_local = tl.dot(dO_tile, V_tile.T, input_precision="ieee")
        
        valid = ((i * BLOCK_SIZE + idx_q[:, None]) >= (j * BLOCK_SIZE + idx_k[None, :]))
        
        S_local = tl.where(valid, S_local * scale, float("-inf"))
        P_local = tl.exp(S_local - L_reshaped)
        P_local = tl.where(valid, P_local, 0.0)
        
        dS_local = P_local * (dP_local - D_reshaped) * scale
        dS_local = tl.where(valid, dS_local, 0.0)
        
        # Fixed layout accumulations matching standard GEMM expectations natively.
        dK_acc = tl.dot(dS_local.T, Q_tile, dK_acc)
        dV_acc = tl.dot(P_local.T, dO_tile, dV_acc)
        
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
        num_warps=4, num_stages=2
    )
    
    dK_dV_kernel[grid](
        Q, K, V, O, dO, L, dK, dV, S_len, scale, H,
        BLOCK_SIZE, BLOCK_D, s_B, s_H, s_S, l_s_B, l_s_H,
        num_warps=4, num_stages=2
    )