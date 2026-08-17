import torch
import triton
import triton.language as tl


@triton.jit
def precompute_D_kernel(
    O, dO, d_D,
    total_S,
    s_B: tl.constexpr,
    s_H: tl.constexpr,
    s_S: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Precompute D = rowsum(dO * O) to accelerate backward attention passes."""
    pid = tl.program_id(0)
    batch_head = pid // (total_S // BLOCK)
    seq_idx = pid % (total_S // BLOCK)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    idx = tl.arange(0, BLOCK)
    
    valid = (seq_idx * BLOCK + idx) < total_S
    
    o_ptr = O + batch_idx * s_B + head_idx * s_H + seq_idx * BLOCK * s_S
    do_ptr = dO + batch_idx * s_B + head_idx * s_H + seq_idx * BLOCK * s_S
    
    O_block = tl.load(o_ptr + idx[:, None] * s_S + tl.arange(0, 128)[None, :] * 1, 
                      mask=valid[:, None] & (tl.arange(0, 128)[None, :] < 128), other=0.0.to(tl.bfloat16))
    dO_block = tl.load(do_ptr + idx[:, None] * s_S + tl.arange(0, 128)[None, :] * 1, 
                       mask=valid[:, None] & (tl.arange(0, 128)[None, :] < 128), other=0.0.to(tl.bfloat16))
    
    D_block = (O_block * dO_block).sum(dim=1)
    
    d_ptr = d_D + batch_idx * (H * BLOCK) + head_idx * BLOCK + seq_idx * BLOCK
    tl.store(d_ptr + idx, D_block, mask=valid)


# -------------------------------------------------------------------
# dQ Kernel
# -------------------------------------------------------------------
@triton.jit
def dQ_kernel(
    Q, K, V, O, dO, L, d_Q, d_D,
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
        other=0.0.to(tl.bfloat16)
    )
    dO_tile = tl.load(
        do_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
        mask=((i * BLOCK_SIZE + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
        other=0.0.to(tl.bfloat16)
    )
    O_tile = tl.load(
        o_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
        mask=((i * BLOCK_SIZE + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
        other=0.0.to(tl.bfloat16)
    )
    
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
            other=0.0.to(tl.bfloat16)
        )
        V_tile = tl.load(
            v_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
            mask=((j * BLOCK_SIZE + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0.to(tl.bfloat16)
        )
        
        S_local = tl.dot(Q_tile, K_tile.T, input_precision="ieee")
        dP_local = tl.dot(dO_tile, V_tile.T, input_precision="ieee")
        
        valid = ((i * BLOCK_SIZE + idx_q[:, None]) >= (j * BLOCK_SIZE + idx_k[None, :]))
        
        S_local = tl.where(valid, S_local * scale, float("-inf"))
        P_local = tl.exp(S_local - L_reshaped)
        P_local = tl.where(valid, P_local, 0.0)
        
        dS_local = P_local * (dP_local - D_reshaped) * scale
        dS_local = tl.where(valid, dS_local, 0.0)
        
        dQ_acc = tl.dot(dS_local, K_tile, dQ_acc)
        
    dq_base = d_Q + batch_idx * s_B + head_idx * s_H + i * BLOCK_SIZE * s_S
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
    Q, K, V, O, dO, L, d_K, d_D,
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
        other=0.0.to(tl.bfloat16)
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
            other=0.0.to(tl.bfloat16)
        )
        dO_tile = tl.load(
            do_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
            mask=((i * BLOCK_SIZE + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0.to(tl.bfloat16)
        )
        O_tile = tl.load(
            o_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
            mask=((i * BLOCK_SIZE + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0.to(tl.bfloat16)
        )
        v_base = V + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
        V_tile = tl.load(
            v_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
            mask=((j * BLOCK_SIZE + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0.to(tl.bfloat16)
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
        
        dK_acc = tl.dot(dS_local.T, Q_tile, dK_acc)
        
    dk_base = d_K + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
    
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
    Q, K, V, O, dO, L, d_V, d_D,
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
        other=0.0.to(tl.bfloat16)
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
            other=0.0.to(tl.bfloat16)
        )
        dO_tile = tl.load(
            do_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
            mask=((i * BLOCK_SIZE + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0.to(tl.bfloat16)
        )
        O_tile = tl.load(
            o_base + idx_q[:, None] * s_S + idx_d[None, :] * 1,
            mask=((i * BLOCK_SIZE + idx_q) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0.to(tl.bfloat16)
        )
        k_base = K + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
        K_tile = tl.load(
            k_base + idx_k[:, None] * s_S + idx_d[None, :] * 1,
            mask=((j * BLOCK_SIZE + idx_k) < S_len)[:, None] & (idx_d < BLOCK_D)[None, :],
            other=0.0.to(tl.bfloat16)
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
        
        dV_acc = tl.dot(P_local.T, dO_tile, dV_acc)
        
    dv_base = d_V + batch_idx * s_B + head_idx * s_H + j * BLOCK_SIZE * s_S
    
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
    
    # Allocate space and seamlessly integrate into pipeline for initial scaling factor D_i calculations
    D_tensor = torch.empty((B, H, S_len), device=Q.device, dtype=torch.float32)
    
    total_elems = B * H * S_len
    grid_D = (triton.cdiv(total_elems, 128),)
    precompute_D_kernel[grid_D](
        O, dO, D_tensor, total_elems, s_B, s_H, s_S, H, BLOCK=128, num_warps=4
    )
    
    grid = (B * H, num_blocks)
    
    dQ_kernel[grid](
        Q, K, V, O, dO, L, dQ, D_tensor, S_len, scale, H,
        BLOCK_SIZE, BLOCK_D, s_B, s_H, s_S, l_s_B, l_s_H,
        num_warps=4, num_stages=2
    )
    
    dK_kernel[grid](
        Q, K, V, O, dO, L, dK, D_tensor, S_len, scale, H,
        BLOCK_SIZE, BLOCK_D, s_B, s_H, s_S, l_s_B, l_s_H,
        num_warps=4, num_stages=2
    )
    
    dV_kernel[grid](
        Q, K, V, O, dO, L, dV, D_tensor, S_len, scale, H,
        BLOCK_SIZE, BLOCK_D, s_B, s_H, s_S, l_s_B, l_s_H,
        num_warps=4, num_stages=2
    )