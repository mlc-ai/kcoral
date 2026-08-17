import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, l_ptr, dq_ptr,
    B, H, S, d, scale,
    q_lb, q_lh, q_ls, q_ld,
    k_lb, k_lh, k_ls, k_ld,
    v_lb, v_lh, v_ls, v_ld,
    do_lb, do_lh, do_ls, do_ld,
    l_lb, l_lh, l_ls,
    dq_lb, dq_lh, dq_ls, dq_ld,
    BM: tl.constexpr, BN: tl.constexpr,
):
    """Compute dQ for causal SDPA backward."""
    pid_bh = tl.program_id(0)
    if pid_bh >= B * H:
        return
    bi = pid_bh // H
    hi = pid_bh % H
    pid_m = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < S
    offs_d = tl.arange(0, d)

    # Batch + head base offsets for each tensor
    bh_q  = bi * q_lb + hi * q_lh
    bh_k  = bi * k_lb + hi * k_lh
    bh_v  = bi * v_lb + hi * v_lh
    bh_do = bi * do_lb + hi * do_lh
    bh_l  = bi * l_lb  + hi * l_lh
    bh_dq = bi * dq_lb + hi * dq_lh

    # Load Q and dO tiles (constant across K/V loop)
    q_ptrs  = bh_q + offs_m[:, None] * q_ls + offs_d[None, :] * q_ld
    q_tile  = tl.load(q_ptr  + q_ptrs,  mask=mask_m[:, None], other=0.0).to(tl.float32)

    do_ptrs = bh_do + offs_m[:, None] * do_ls + offs_d[None, :] * do_ld
    do_tile = tl.load(do_ptr + do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    l_ptrs = bh_l + offs_m * l_ls
    l_vals = tl.load(l_ptr + l_ptrs, mask=mask_m, other=0.0)

    # FP32 accumulators
    acc_dS_K = tl.zeros((BM, d), dtype=tl.float32)
    acc_D_PK = tl.zeros((BM,), dtype=tl.float32)

    for start_n in range(0, S, BN):
        offs_n = start_n + tl.arange(0, BN)
        mask_n = offs_n < S

        # Causal mask: kv_pos <= query_pos
        causal = offs_n[None, :] <= offs_m[:, None]
        valid  = causal & mask_m[:, None] & mask_n[None, :]

        k_ptrs = bh_k + offs_n[:, None] * k_ls + offs_d[None, :] * k_ld
        k_tile = tl.load(k_ptr + k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        v_ptrs = bh_v + offs_n[:, None] * v_ls + offs_d[None, :] * v_ld
        v_tile = tl.load(v_ptr + v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        s  = tl.dot(q_tile, k_tile.T) * scale      # [BM,BN]
        p  = tl.where(valid, tl.exp(s - l_vals[:, None]), 0.0)  # [BM,BN]
        dp = tl.dot(do_tile, v_tile.T)              # [BM,BN]

        # dQ contribution = scale * (P*dP)*K - scale*D*P*K
        pdp = p * dp  # [BM,BN]
        acc_dS_K += tl.dot(pdp, k_tile)             # [BM,d]
        acc_D_PK += tl.sum(pdp, axis=1)             # [BM]
        acc_D_PK_scaled_partial = tl.zeros((BM, d), dtype=tl.float32)
        # We need sum(P*dP) * sum(P*K) but we can accumulate sum(P*K) separately
        # Instead: track D_running and PK_running, compute D*PK at end
        pass

    # Recompute cleanly: track D and PK separately
    # Reset and redo accumulation properly
    pass


# Let me rewrite this more carefully
@triton.jit
def _attn_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, l_ptr, dq_ptr,
    B, H, S, d, scale,
    q_lb, q_lh, q_ls, q_ld,
    k_lb, k_lh, k_ls, k_ld,
    v_lb, v_lh, v_ls, v_ld,
    do_lb, do_lh, do_ls, do_ld,
    l_lb, l_lh, l_ls,
    dq_lb, dq_lh, dq_ls, dq_ld,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    if pid_bh >= B * H:
        return
    bi = pid_bh // H
    hi = pid_bh % H
    pid_m = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < S
    offs_d = tl.arange(0, d)

    bh_q  = bi * q_lb + hi * q_lh
    bh_k  = bi * k_lb + hi * k_lh
    bh_v  = bi * v_lb + hi * v_lh
    bh_do = bi * do_lb + hi * do_lh
    bh_l  = bi * l_lb  + hi * l_lh
    bh_dq = bi * dq_lb + hi * dq_lh

    q_ptrs  = bh_q + offs_m[:, None] * q_ls + offs_d[None, :] * q_ld
    q_tile  = tl.load(q_ptr  + q_ptrs,  mask=mask_m[:, None], other=0.0).to(tl.float32)

    do_ptrs = bh_do + offs_m[:, None] * do_ls + offs_d[None, :] * do_ld
    do_tile = tl.load(do_ptr + do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    l_ptrs = bh_l + offs_m * l_ls
    l_vals = tl.load(l_ptr + l_ptrs, mask=mask_m, other=0.0)

    # dQ = scale * (sum_n dS[n] @ K[n]) where dS = P*(dP - D)
    #     = scale * (sum_n (P*dP)@K  -  sum_n D[row]*P@K)
    #     = scale * sum_n (P*dP)@K   - scale * D_row * sum_n P@K
    acc_SKK = tl.zeros((BM, d), dtype=tl.float32)   # sum (P*dP)*K
    acc_PK  = tl.zeros((BM, d), dtype=tl.float32)   # sum P*K
    acc_D   = tl.zeros((BM,), dtype=tl.float32)     # sum P*dP (= D per row)

    for start_n in range(0, S, BN):
        offs_n = start_n + tl.arange(0, BN)
        mask_n = offs_n < S

        causal = offs_n[None, :] <= offs_m[:, None]
        valid  = causal & mask_m[:, None] & mask_n[None, :]

        k_ptrs = bh_k + offs_n[:, None] * k_ls + offs_d[None, :] * k_ld
        k_tile = tl.load(k_ptr + k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        v_ptrs = bh_v + offs_n[:, None] * v_ls + offs_d[None, :] * v_ld
        v_tile = tl.load(v_ptr + v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        s  = tl.dot(q_tile, k_tile.T) * scale
        p  = tl.where(valid, tl.exp(s - l_vals[:, None]), 0.0)
        dp = tl.dot(do_tile, v_tile.T)

        pdp = p * dp
        acc_D   += tl.sum(pdp, axis=1)
        acc_SKK += tl.dot(pdp, k_tile)
        acc_PK  += tl.dot(p, k_tile)

    dq_val = scale * (acc_SKK - acc_D[:, None] * acc_PK)

    dq_ptrs = bh_dq + offs_m[:, None] * dq_ls + offs_d[None, :] * dq_ld
    tl.store(dq_ptr + dq_ptrs, dq_val.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _attn_bwd_dkv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, l_ptr, dk_ptr, dv_ptr,
    B, H, S, d, scale,
    q_lb, q_lh, q_ls, q_ld,
    k_lb, k_lh, k_ls, k_ld,
    v_lb, v_lh, v_ls, v_ld,
    do_lb, do_lh, do_ls, do_ld,
    o_lb, o_lh, o_ls, o_ld,
    l_lb, l_lh, l_ls,
    dk_lb, dk_lh, dk_ls, dk_ld,
    dv_lb, dv_lh, dv_ls, dv_ld,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= tl.cdiv(S, BN):
        return
    pid_bh = tl.program_id(1)
    if pid_bh >= B * H:
        return

    bi = pid_bh // H
    hi = pid_bh % H

    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < S
    offs_d = tl.arange(0, d)

    bh_q  = bi * q_lb  + hi * q_lh
    bh_k  = bi * k_lb  + hi * k_lh
    bh_v  = bi * v_lb  + hi * v_lh
    bh_do = bi * do_lb + hi * do_lh
    bh_o  = bi * o_lb  + hi * o_lh
    bh_l  = bi * l_lb  + hi * l_lh
    bh_dk = bi * dk_lb + hi * dk_lh
    bh_dv = bi * dv_lb + hi * dv_lh

    k_ptrs = bh_k + offs_n[:, None] * k_ls + offs_d[None, :] * k_ld
    k_tile = tl.load(k_ptr + k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    v_ptrs = bh_v + offs_n[:, None] * v_ls + offs_d[None, :] * v_ld
    v_tile = tl.load(v_ptr + v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    dk_acc = tl.zeros((BN, d), dtype=tl.float32)
    dv_acc = tl.zeros((BN, d), dtype=tl.float32)

    for start_m in range(0, S, BM):
        offs_m = start_m + tl.arange(0, BM)
        mask_m = offs_m < S

        causal = offs_n[None, :] <= offs_m[:, None]
        valid  = causal & mask_m[:, None] & mask_n[None, :]

        q_ptrs  = bh_q + offs_m[:, None] * q_ls + offs_d[None, :] * q_ld
        q_tile  = tl.load(q_ptr  + q_ptrs,  mask=mask_m[:, None], other=0.0).to(tl.float32)

        do_ptrs = bh_do + offs_m[:, None] * do_ls + offs_d[None, :] * do_ld
        do_tile = tl.load(do_ptr + do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        o_ptrs = bh_o + offs_m[:, None] * o_ls + offs_d[None, :] * o_ld
        o_tile = tl.load(o_ptr  + o_ptrs,  mask=mask_m[:, None], other=0.0).to(tl.float32)

        l_ptrs = bh_l + offs_m * l_ls
        l_vals = tl.load(l_ptr + l_ptrs, mask=mask_m, other=0.0)

        D_row = tl.sum(do_tile * o_tile, axis=1)  # [BM] fp32

        s  = tl.dot(q_tile, k_tile.T) * scale
        p  = tl.where(valid, tl.exp(s - l_vals[:, None]), 0.0)
        dp = tl.dot(do_tile, v_tile.T)

        dS = p * (dp - D_row[:, None]) * scale  # [BM,BN] fp32

        dk_acc += tl.dot(dS.T, q_tile)
        dv_acc += tl.dot(p.T, do_tile)

    dk_ptrs = bh_dk + offs_n[:, None] * dk_ls + offs_d[None, :] * dk_ld
    tl.store(dk_ptr + dk_ptrs, dk_acc.to(tl.bfloat16), mask=mask_n[:, None])

    dv_ptrs = bh_dv + offs_n[:, None] * dv_ls + offs_d[None, :] * dv_ld
    tl.store(dv_ptr + dv_ptrs, dv_acc.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    B, H, S, d = Q.shape
    device = Q.device
    torch.cuda.set_device(device)

    scale = 1.0 / math.sqrt(d)

    BM = 32
    BN = 32

    qs = Q.stride()
    ks = K.stride()
    vs = V.stride()
    os = O.stride()
    dos = dO.stride()
    ls = L.stride()
    dqs = dQ.stride()
    dks = dK.stride()
    dvs = dV.stride()

    nh = B * H

    # dQ: tile over M (query positions), parallelize over B,H
    n_m = triton.cdiv(S, BM)
    dq_grid = (nh, n_m)
    _attn_bwd_dq_kernel[dq_grid](
        Q, K, V, dO, L, dQ,
        B, H, S, d, scale,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        dqs[0], dqs[1], dqs[2], dqs[3],
        BM=BM, BN=BN,
        num_warps=4, num_stages=3,
    )

    # dK,dV: tile over N (KV positions), parallelize over B,H
    n_n = triton.cdiv(S, BN)
    dkv_grid = (n_n, nh)
    _attn_bwd_dkv_kernel[dkv_grid](
        Q, K, V, dO, O, L, dK, dV,
        B, H, S, d, scale,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3],
        dos[0], dos[1], dos[2], dos[3],
        os[0], os[1], os[2], os[3],
        ls[0], ls[1], ls[2],
        dks[0], dks[1], dks[2], dks[3],
        dvs[0], dvs[1], dvs[2], dvs[3],
        BM=BM, BN=BN,
        num_warps=4, num_stages=3,
    )