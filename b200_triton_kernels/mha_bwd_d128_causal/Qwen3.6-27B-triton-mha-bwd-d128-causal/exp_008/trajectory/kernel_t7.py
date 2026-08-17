import math
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, l_ptr, dq_ptr,
    B, H, S, scale,
    q_lb, q_lh, q_ls, q_ld,
    k_lb, k_lh, k_ls, k_ld,
    v_lb, v_lh, v_ls, v_ld,
    do_lb, do_lh, do_ls, do_ld,
    l_lb, l_lh, l_ls,
    dq_lb, dq_lh, dq_ls, dq_ld,
    BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr,
):
    """
    Compute dQ for causal SDPA backward.
    
    Formulas:
      s[m,n] = Q[m,:]·K[n,:] / sqrt(d)
      P[m,n] = softmax(s, causal_mask) = exp(s[m,n] - L[m])
      dP = dO @ V^T
      D_row[m] = sum_n P[m,n] * dP[m,n]
      dS = scale * P * (dP - D_row)
      dQ = sum_n dS[: ,n_block] @ K[n_block,:]
    
    All computation in fp32, only conversion at load/store.
    """
    pid_bh = tl.program_id(0)
    if pid_bh >= B * H:
        return
    bi = pid_bh // H
    hi = pid_bh % H
    pid_m = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < S
    offs_d = tl.arange(0, HD)

    bq  = bi * q_lb  + hi * q_lh
    bk  = bi * k_lb  + hi * k_lh
    bv  = bi * v_lb  + hi * v_lh
    bdo = bi * do_lb + hi * do_lh
    bl  = bi * l_lb  + hi * l_lh
    bdq = bi * dq_lb + hi * dq_lh

    # Load query-side data (constant across KV loop)
    q_ptrs = bq + offs_m[:, None] * q_ls + offs_d[None, :] * q_ld
    q_tile = tl.load(q_ptr + q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    do_ptrs = bdo + offs_m[:, None] * do_ls + offs_d[None, :] * do_ld
    do_tile = tl.load(do_ptr + do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    l_ptrs = bl + offs_m * l_ls
    l_vals = tl.load(l_ptr + l_ptrs, mask=mask_m, other=0.0)

    acc = tl.zeros((BM, HD), dtype=tl.float32)

    for start_n in range(0, S, BN):
        offs_n = start_n + tl.arange(0, BN)
        mask_n = offs_n < S

        # Causal mask: key_pos <= query_pos
        causal = offs_n[None, :] <= offs_m[:, None]
        valid = causal & mask_m[:, None] & mask_n[None, :]

        k_ptrs = bk + offs_n[:, None] * k_ls + offs_d[None, :] * k_ld
        k_tile = tl.load(k_ptr + k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        v_ptrs = bv + offs_n[:, None] * v_ls + offs_d[None, :] * v_ld
        v_tile = tl.load(v_ptr + v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        s = tl.dot(q_tile, k_tile.T) * scale          # [BM,BN]
        p = tl.where(valid, tl.exp(s - l_vals[:, None]), 0.0)  # [BM,BN]
        dp = tl.dot(do_tile, v_tile.T)                # [BM,BN]

        pdp = p * dp                                    # [BM,BN]
        D_row = tl.sum(pdp, axis=1)                     # [BM]
        dS = scale * p * (dp - D_row[:, None])           # [BM,BN]

        acc += tl.dot(dS, k_tile)                        # [BM,HD]

    dq_ptrs = bdq + offs_m[:, None] * dq_ls + offs_d[None, :] * dq_ld
    tl.store(dq_ptr + dq_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])


@triton.jit
def _attn_bwd_dkv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr, o_ptr, l_ptr, dk_ptr, dv_ptr,
    B, H, S, scale,
    q_lb, q_lh, q_ls, q_ld,
    k_lb, k_lh, k_ls, k_ld,
    v_lb, v_lh, v_ls, v_ld,
    do_lb, do_lh, do_ls, do_ld,
    o_lb, o_lh, o_ls, o_ld,
    l_lb, l_lh, l_ls,
    dk_lb, dk_lh, dk_ls, dk_ld,
    dv_lb, dv_lh, dv_ls, dv_ld,
    BM: tl.constexpr, BN: tl.constexpr, HD: tl.constexpr,
):
    """
    Compute dK and dV for causal SDPA backward.
    
    dV = P^T @ dO
    dK = scale * dS^T @ Q
    
    Process one (batch, head, kv-tile) per program instance.
    """
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
    offs_d = tl.arange(0, HD)

    bq  = bi * q_lb  + hi * q_lh
    bk  = bi * k_lb  + hi * k_lh
    bv  = bi * v_lb  + hi * v_lh
    bdo = bi * do_lb + hi * do_lh
    bo  = bi * o_lb  + hi * o_lh
    bl  = bi * l_lb  + hi * l_lh
    bdk = bi * dk_lb + hi * dk_lh
    bdv = bi * dv_lb + hi * dv_lh

    k_ptrs = bk + offs_n[:, None] * k_ls + offs_d[None, :] * k_ld
    k_tile = tl.load(k_ptr + k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    v_ptrs = bv + offs_n[:, None] * v_ls + offs_d[None, :] * v_ld
    v_tile = tl.load(v_ptr + v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    dk_acc = tl.zeros((BN, HD), dtype=tl.float32)
    dv_acc = tl.zeros((BN, HD), dtype=tl.float32)

    for start_m in range(0, S, BM):
        offs_m = start_m + tl.arange(0, BM)
        mask_m = offs_m < S

        causal = offs_n[None, :] <= offs_m[:, None]
        valid = causal & mask_m[:, None] & mask_n[None, :]

        q_ptrs = bq + offs_m[:, None] * q_ls + offs_d[None, :] * q_ld
        q_tile = tl.load(q_ptr + q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        do_ptrs = bdo + offs_m[:, None] * do_ls + offs_d[None, :] * do_ld
        do_tile = tl.load(do_ptr + do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        o_ptrs = bo + offs_m[:, None] * o_ls + offs_d[None, :] * o_ld
        o_tile = tl.load(o_ptr + o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        l_ptrs = bl + offs_m * l_ls
        l_vals = tl.load(l_ptr + l_ptrs, mask=mask_m, other=0.0)

        D_row = tl.sum(do_tile * o_tile, axis=1)     # [BM]

        s = tl.dot(q_tile, k_tile.T) * scale           # [BM,BN]
        p = tl.where(valid, tl.exp(s - l_vals[:, None]), 0.0)  # [BM,BN]
        dp = tl.dot(do_tile, v_tile.T)                 # [BM,BN]

        dS = scale * p * (dp - D_row[:, None])         # [BM,BN]

        dk_acc += tl.dot(dS.T, q_tile)                 # [BN,HD]
        dv_acc += tl.dot(p.T, do_tile)                 # [BN,HD]

    dk_ptrs = bdk + offs_n[:, None] * dk_ls + offs_d[None, :] * dk_ld
    tl.store(dk_ptr + dk_ptrs, dk_acc.to(tl.bfloat16), mask=mask_n[:, None])

    dv_ptrs = bdv + offs_n[:, None] * dv_ls + offs_d[None, :] * dv_ld
    tl.store(dv_ptr + dv_ptrs, dv_acc.to(tl.bfloat16), mask=mask_n[:, None])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Destination-passing entry point for causal MHA backward."""
    B, H, S, d = Q.shape
    device = Q.device
    torch.cuda.set_device(device)

    scale = 1.0 / math.sqrt(d)

    BM = 16
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

    # Zero outputs before computation
    dQ.zero_()
    dK.zero_()
    dV.zero_()

    # dQ: parallelize over (batch,head) × query-tile
    n_m = triton.cdiv(S, BM)
    dq_grid = (nh, n_m)
    _attn_bwd_dq_kernel[dq_grid](
        Q, K, V, dO, L, dQ,
        B, H, S, scale,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        dqs[0], dqs[1], dqs[2], dqs[3],
        BM=BM, BN=BN, HD=d,
        num_warps=4, num_stages=2,
    )

    # dK,dV: parallelize over kv-tile × (batch,head)
    n_n = triton.cdiv(S, BN)
    dkv_grid = (n_n, nh)
    _attn_bwd_dkv_kernel[dkv_grid](
        Q, K, V, dO, O, L, dK, dV,
        B, H, S, scale,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3],
        dos[0], dos[1], dos[2], dos[3],
        os[0], os[1], os[2], os[3],
        ls[0], ls[1], ls[2],
        dks[0], dks[1], dks[2], dks[3],
        dvs[0], dvs[1], dvs[2], dvs[3],
        BM=BM, BN=BN, HD=d,
        num_warps=4, num_stages=2,
    )