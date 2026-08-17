import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, l_ptr, dq_ptr,
    q_sb, q_sh, q_ss, q_sd,
    k_sb, k_sh, k_ss, k_sd,
    v_sb, v_sh, v_ss, v_sd,
    do_sb, do_sh, do_ss, do_sd,
    l_sb, l_sh, l_ss,
    dq_sb, dq_sh, dq_ss, dq_sd,
    num_bh, seq_len, scale, num_heads,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Compute dQ for causal SDPA backward via single-pass decomposition:
      dQ = scale * (sum_n (P*dP)*K  -  D * sum_n P*K)
    Single pass avoids materialising a separate D buffer.
    All intermediates kept in fp32; only the final store is bf16.
    """
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    if pid_bh >= num_bh:
        return

    bi = pid_bh // num_heads
    hi = pid_bh % num_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < seq_len
    offs_d = tl.arange(0, HEAD_DIM)

    # batch-head offsets
    bh_q  = bi * q_sb  + hi * q_sh
    bh_k  = bi * k_sb  + hi * k_sh
    bh_v  = bi * v_sb  + hi * v_sh
    bh_do = bi * do_sb + hi * do_sh
    bh_l  = bi * l_sb  + hi * l_sh
    bh_dq = bi * dq_sb + hi * dq_sh

    # ---- constant across KV loop ----
    q_ptrs  = q_ptr  + bh_q  + offs_m[:, None] * q_ss  + offs_d[None, :] * q_sd
    q_tile  = tl.load(q_ptrs,  mask=mask_m[:, None], other=0.0).to(tl.float32)

    do_ptrs = do_ptr + bh_do + offs_m[:, None] * do_ss + offs_d[None, :] * do_sd
    do_tile = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    l_ptrs  = l_ptr  + bh_l  + offs_m * l_ss
    l_vals  = tl.load(l_ptrs, mask=mask_m, other=0.0)  # already fp32

    # ---- fp32 accumulators ----
    acc_pdK = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    acc_pK  = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    acc_D   = tl.zeros((BLOCK_M,),         dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        causal = (offs_n[None, :] <= offs_m[:, None])
        valid  = causal & mask_m[:, None] & mask_n[None, :]

        k_ptrs = k_ptr + bh_k + offs_n[:, None] * k_ss + offs_d[None, :] * k_sd
        k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        v_ptrs = v_ptr + bh_v + offs_n[:, None] * v_ss + offs_d[None, :] * v_sd
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        s  = tl.dot(q_tile, k_tile.T) * scale          # [BM,BN] fp32
        p  = tl.where(valid, tl.exp(s - l_vals[:, None]), 0.0)   # fp32
        dp = tl.dot(do_tile, v_tile.T)                 # [BM,BN] fp32

        pdp = p * dp                                     # [BM,BN] fp32
        acc_D   += tl.sum(pdp, axis=1)                   # [BM] fp32
        acc_pdK += tl.dot(pdp, k_tile)                   # [BM,d] fp32
        acc_pK  += tl.dot(p, k_tile)                     # [BM,d] fp32

    dq_tile = scale * (acc_pdK - acc_D[:, None] * acc_pK)

    dq_ptrs = dq_ptr + bh_dq + offs_m[:, None] * dq_ss + offs_d[None, :] * dq_sd
    tl.store(dq_ptrs, dq_tile.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _attn_bwd_dkv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, l_ptr, dk_ptr, dv_ptr,
    q_sb, q_sh, q_ss, q_sd,
    k_sb, k_sh, k_ss, k_sd,
    v_sb, v_sh, v_ss, v_sd,
    do_sb, do_sh, do_ss, do_sd,
    o_sb, o_sh, o_ss, o_sd,
    l_sb, l_sh, l_ss,
    dk_sb, dk_sh, dk_ss, dk_sd,
    dv_sb, dv_sh, dv_ss, dv_sd,
    num_bh, seq_len, scale, num_heads,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Compute dK and dV for causal SDPA backward.
      dV = P^T @ dO
      dK = scale * dS^T @ Q   with dS = P * (dP - D)
    D[i] = sum_k dO[i,k] * O[i,k] is computed on-the-fly.
    All intermediates in fp32; final store in bf16.
    """
    pid_n  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    if pid_bh >= num_bh:
        return

    bi = pid_bh // num_heads
    hi = pid_bh % num_heads

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < seq_len
    offs_d = tl.arange(0, HEAD_DIM)

    bh_q  = bi * q_sb  + hi * q_sh
    bh_k  = bi * k_sb  + hi * k_sh
    bh_v  = bi * v_sb  + hi * v_sh
    bh_do = bi * do_sb + hi * do_sh
    bh_o  = bi * o_sb  + hi * o_sh
    bh_l  = bi * l_sb  + hi * l_sh
    bh_dk = bi * dk_sb + hi * dk_sh
    bh_dv = bi * dv_sb + hi * dv_sh

    # ---- constant across Q loop ----
    k_ptrs = k_ptr + bh_k + offs_n[:, None] * k_ss + offs_d[None, :] * k_sd
    k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    v_ptrs = v_ptr + bh_v + offs_n[:, None] * v_ss + offs_d[None, :] * v_sd
    v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    dk_acc = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

    for start_m in range(0, seq_len, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len

        causal = (offs_n[None, :] <= offs_m[:, None])
        valid  = causal & mask_m[:, None] & mask_n[None, :]

        q_ptrs  = q_ptr  + bh_q  + offs_m[:, None] * q_ss  + offs_d[None, :] * q_sd
        q_tile  = tl.load(q_ptrs,  mask=mask_m[:, None], other=0.0).to(tl.float32)

        do_ptrs = do_ptr + bh_do + offs_m[:, None] * do_ss + offs_d[None, :] * do_sd
        do_tile = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        o_ptrs  = o_ptr  + bh_o  + offs_m[:, None] * o_ss  + offs_d[None, :] * o_sd
        o_tile  = tl.load(o_ptrs,  mask=mask_m[:, None], other=0.0).to(tl.float32)

        l_ptrs  = l_ptr  + bh_l  + offs_m * l_ss
        l_vals  = tl.load(l_ptrs, mask=mask_m, other=0.0)  # fp32

        d_vals = tl.sum(do_tile * o_tile, axis=1)           # [BM] fp32

        s  = tl.dot(q_tile, k_tile.T) * scale               # [BM,BN] fp32
        p  = tl.where(valid, tl.exp(s - l_vals[:, None]), 0.0)   # fp32
        dp = tl.dot(do_tile, v_tile.T)                      # [BM,BN] fp32

        dS = p * (dp - d_vals[:, None]) * scale             # [BM,BN] fp32

        dk_acc += tl.dot(dS.T, q_tile)                      # [BN,d] fp32
        dv_acc += tl.dot(p.T, do_tile)                      # [BN,d] fp32

    dk_ptrs = dk_ptr + bh_dk + offs_n[:, None] * dk_ss + offs_d[None, :] * dk_sd
    tl.store(dk_ptrs, dk_acc.to(tl.bfloat16), mask=mask_n[:, None])

    dv_ptrs = dv_ptr + bh_dv + offs_n[:, None] * dv_ss + offs_d[None, :] * dv_sd
    tl.store(dv_ptrs, dv_acc.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Destination-passing entry point for causal MHA backward (bf16)."""
    B, H, S, d = Q.shape
    device = Q.device
    torch.cuda.set_device(device)

    scale = 1.0 / math.sqrt(d)
    num_bh = B * H

    BLOCK_M = 64
    BLOCK_N = 64

    qs  = Q.stride()
    ks  = K.stride()
    vs  = V.stride()
    os_ = O.stride()
    dos = dO.stride()
    ls  = L.stride()
    dqs = dQ.stride()
    dks = dK.stride()
    dvs = dV.stride()

    # ---------- dQ ----------
    dq_grid = (triton.cdiv(S, BLOCK_M), num_bh)
    _attn_bwd_dq_kernel[dq_grid](
        Q, K, V, dO, L, dQ,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        dqs[0], dqs[1], dqs[2], dqs[3],
        num_bh, S, scale, H,
        HEAD_DIM=d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=3,
    )

    # ---------- dK, dV ----------
    dkv_grid = (triton.cdiv(S, BLOCK_N), num_bh)
    _attn_bwd_dkv_kernel[dkv_grid](
        Q, K, V, dO, O, L, dK, dV,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3],
        dos[0], dos[1], dos[2], dos[3],
        os_[0], os_[1], os_[2], os_[3],
        ls[0], ls[1], ls[2],
        dks[0], dks[1], dks[2], dks[3],
        dvs[0], dvs[1], dvs[2], dvs[3],
        num_bh, S, scale, H,
        HEAD_DIM=d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=3,
    )