import torch
import triton
import triton.language as tl


@triton.jit
def _dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    B, H, S, d,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dK and dV for multi-head attention backward.

    Each program handles one (batch, head) group and one KV-sequence tile.
    Loops over all Q-sequence tiles, accumulating into local registers.
    All computation in FP32; output cast to BF16 on store.
    """
    pid_n = tl.program_id(0)    # KV tile index along sequence
    pid_bh = tl.program_id(1)   # (batch * head) index

    bh_stride = S * d
    bh_base = pid_bh * bh_stride

    # Column (KV-position) indices for this tile
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < S

    # Head-dimension indices
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < d

    # Combined mask for K/V loads: [BLOCK_N, BLOCK_D]
    nk_mask = mask_n[:, None] & mask_d[None, :]

    # Base pointer offset for this (B,H) group
    kv_base = bh_base + offs_n[:, None] * d + offs_d[None, :]

    # Load K and V once -- reused across all Q-tile iterations
    K_reg = tl.load(K_ptr + kv_base, mask=nk_mask, other=0.0).to(tl.float32)
    V_reg = tl.load(V_ptr + kv_base, mask=nk_mask, other=0.0).to(tl.float32)

    # Output accumulators in FP32
    acc_dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    num_m_tiles = tl.cdiv(S, BLOCK_M)

    for pm in range(num_m_tiles):
        offs_m = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < S

        # Masks for loads
        mr_mask = mask_m[:, None]          # [BLOCK_M, 1] for [BLOCK_M, BLOCK_D]
        mn_mask = mask_m[:, None] & mask_n[None, :]  # [BLOCK_M, BLOCK_N] for attention matrix

        # Pointer base for row data
        row_base = bh_base + offs_m[:, None] * d + offs_d[None, :]

        Q_reg  = tl.load(Q_ptr  + row_base, mask=mr_mask, other=0.0).to(tl.float32)
        dO_reg = tl.load(dO_ptr + row_base, mask=mr_mask, other=0.0).to(tl.float32)
        O_reg  = tl.load(O_ptr  + row_base, mask=mr_mask, other=0.0).to(tl.float32)

        # Load L for each query position (FP32 scalar per row)
        L_reg = tl.load(L_ptr + pid_bh * S + offs_m, mask=mask_m, other=0.0)

        # Forward quantities
        S_mat = tl.dot(Q_reg, K_reg.T) * sm_scale               # [BLOCK_M, BLOCK_N]
        P     = tl.exp(S_mat - L_reg[:, None])                   # [BLOCK_M, BLOCK_N]
        dP    = tl.dot(dO_reg, V_reg.T)                          # [BLOCK_M, BLOCK_N]

        # D[i] = sum_d(dO[i, d] * O[i, d])  -- scalar per query row
        D = tl.sum(dO_reg * O_reg, axis=1)                       # [BLOCK_M]

        # dS_i,j = P_i,j * (dP_i,j - D_i) * sm_scale
        dS = P * (dP - D[:, None]) * sm_scale                    # [BLOCK_M, BLOCK_N]

        # Accumulate: dK[n,d] += sum_m dS[m,n] * Q[m,d]
        acc_dk = tl.dot(dS.T, Q_reg, acc=acc_dk)                # [BLOCK_N, BLOCK_D]
        # Accumulate: dV[n,d] += sum_m P[m,n] * dO[m,d]
        acc_dv = tl.dot(P.T, dO_reg, acc=acc_dv)                # [BLOCK_N, BLOCK_D]

    # Store results
    tl.store(dK_ptr + kv_base, acc_dk.to(tl.bfloat16), mask=nk_mask)
    tl.store(dV_ptr + kv_base, acc_dv.to(tl.bfloat16), mask=nk_mask)


@triton.jit
def _dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    B, H, S, d,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ for multi-head attention backward.

    Each program handles one (batch, head) group and one Q-sequence tile.
    Loops over all KV-sequence tiles, accumulating into local registers.
    All computation in FP32; output cast to BF16 on store.
    """
    pid_m = tl.program_id(0)    # Q tile index along sequence
    pid_bh = tl.program_id(1)   # (batch * head) index

    bh_stride = S * d
    bh_base = pid_bh * bh_stride

    # Row (Q-position) indices for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S

    # Head-dimension indices
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < d

    # Combined mask for row loads: [BLOCK_M, BLOCK_D]
    mk_mask = mask_m[:, None] & mask_d[None, :]

    # Pointer base for row data
    row_base = bh_base + offs_m[:, None] * d + offs_d[None, :]

    # Load Q, dO, O once -- reused across all KV-tile iterations
    Q_reg  = tl.load(Q_ptr  + row_base, mask=mk_mask, other=0.0).to(tl.float32)
    dO_reg = tl.load(dO_ptr + row_base, mask=mk_mask, other=0.0).to(tl.float32)
    O_reg  = tl.load(O_ptr  + row_base, mask=mk_mask, other=0.0).to(tl.float32)

    # Load L for each query position
    L_reg = tl.load(L_ptr + pid_bh * S + offs_m, mask=mask_m, other=0.0)

    # D[i] = sum_d(dO[i, d] * O[i, d])
    D = tl.sum(dO_reg * O_reg, axis=1)                                    # [BLOCK_M]

    # Output accumulator in FP32
    acc_dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_n_tiles = tl.cdiv(S, BLOCK_N)

    for pn in range(num_n_tiles):
        offs_n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        # Mask for K/V loads: [BLOCK_N, BLOCK_D]
        nk_mask = mask_n[:, None] & mask_d[None, :]

        kv_base = bh_base + offs_n[:, None] * d + offs_d[None, :]

        K_reg = tl.load(K_ptr + kv_base, mask=nk_mask, other=0.0).to(tl.float32)
        V_reg = tl.load(V_ptr + kv_base, mask=nk_mask, other=0.0).to(tl.float32)

        # Forward quantities
        S_mat = tl.dot(Q_reg, K_reg.T) * sm_scale                        # [BLOCK_M, BLOCK_N]
        P     = tl.exp(S_mat - L_reg[:, None])                           # [BLOCK_M, BLOCK_N]
        dP    = tl.dot(dO_reg, V_reg.T)                                  # [BLOCK_M, BLOCK_N]

        # dS_i,j = P_i,j * (dP_i,j - D_i) * sm_scale
        dS = P * (dP - D[:, None]) * sm_scale                            # [BLOCK_M, BLOCK_N]

        # Accumulate: dQ[m,d] += sum_n dS[m,n] * K[n,d]
        acc_dq = tl.dot(dS, K_reg, acc=acc_dq)                           # [BLOCK_M, BLOCK_D]

    # Store result
    tl.store(dQ_ptr + row_base, acc_dq.to(tl.bfloat16), mask=mk_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV.

    Inputs (in definition order): Q, K, V, O, dO, L
    Outputs (in definition order): dQ, dK, dV

    Matches cuDNN SDPA backward with compute_data_type=FLOAT.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    sm_scale = 1.0 / (d ** 0.5)

    # Block sizes tuned for BF16/Hopper: fit registers, match d=128
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_D = d  # 128 -- entire head dim fits in one tile

    BH = B * H
    num_kv_tiles = triton.cdiv(S, BLOCK_N)
    num_q_tiles  = triton.cdiv(S, BLOCK_M)

    # Launch configuration: moderate warps, multi-stage pipelining
    num_warps = 4
    num_stages = 3

    # --- Phase 1: compute dK and dV ---
    # Grid: (num_kv_tiles, B*H)
    grid_dkdv = (num_kv_tiles, BH)
    _dkdv_kernel[grid_dkdv](
        Q, K, V, O, dO, L,
        dK, dV,
        B, H, S, d, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=num_stages,
    )

    # --- Phase 2: compute dQ ---
    # Grid: (num_q_tiles, B*H)
    grid_dq = (num_q_tiles, BH)
    _dq_kernel[grid_dq](
        Q, K, V, O, dO, L,
        dQ,
        B, H, S, d, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=num_stages,
    )