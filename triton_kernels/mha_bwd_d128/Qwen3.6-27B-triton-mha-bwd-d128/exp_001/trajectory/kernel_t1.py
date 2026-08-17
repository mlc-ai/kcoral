import torch
import triton
import triton.language as tl


@triton.jit
def _fill_zeros_kernel(out_ptr, num_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    tl.store(out_ptr + offsets, 0.0, mask=mask)


@triton.jit
def _mha_bwd_dv_kernel(
    Q_ptr, K_ptr, dO_ptr, L_ptr, dV_ptr,
    B, H, S, d_head,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_dOb, stride_dOh, stride_dOs, stride_dOd,
    stride_Lb, stride_Lh, stride_Ls,
    stride_dVb, stride_dVh, stride_dVs, stride_dVd,
    scale,
    INF_VAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dV[b,h,:,d] = attention[b,h,:,:].T @ dO[b,h,:,d]."""
    pid_bh = tl.program_id(0)
    pid_k = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_bh = b * stride_qb + h * stride_qh
    k_bh = b * stride_kb + h * stride_kh
    dov_bh = b * stride_dOb + h * stride_dOh
    l_bh = b * stride_Lb + h * stride_Lh
    dv_bh = b * stride_dVb + h * stride_dVh

    start_k = pid_k * BLOCK_K
    offs_k = start_k + tl.arange(0, BLOCK_K)
    k_mask = offs_k < S
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < d_head

    k_ptrs = K_ptr + k_bh + offs_k[:, None] * stride_ks + offs_d[None, :] * stride_kd
    K_tile = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    dv_acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

    for start_q in range(0, S, BLOCK_Q):
        offs_q = start_q + tl.arange(0, BLOCK_Q)
        q_mask = offs_q < S

        q_ptrs = Q_ptr + q_bh + offs_q[:, None] * stride_qs + offs_d[None, :] * stride_qd
        Q_tile = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        logits = tl.dot(Q_tile, K_tile) * scale

        l_ptrs = L_ptr + l_bh + offs_q * stride_Ls
        L_tile = tl.load(l_ptrs, mask=q_mask, other=INF_VAL)

        valid_qk = q_mask[:, None] & k_mask[None, :]
        L_safe = tl.where(q_mask[:, None], L_tile[:, None], INF_VAL)
        attn = tl.exp(logits - L_safe)
        attn = tl.where(valid_qk, attn, 0.0)

        dov_ptrs = dO_ptr + dov_bh + offs_q[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
        dO_tile = tl.load(dov_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        dv_acc = tl.dot(attn.T, dO_tile, dv_acc)

    dv_ptrs = dV_ptr + dv_bh + offs_k[:, None] * stride_dVs + offs_d[None, :] * stride_dVd
    tl.store(dv_ptrs, dv_acc.to(tl.bfloat16), mask=k_mask[:, None] & d_mask[None, :])


@triton.jit
def _mha_bwd_dq_dk_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr, dK_ptr,
    B, H, S, d_head,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_dOb, stride_dOh, stride_dOs, stride_dOd,
    stride_Lb, stride_Lh, stride_Ls,
    stride_dQb, stride_dQh, stride_dQs, stride_dQd,
    stride_dKb, stride_dKh, stride_dKs, stride_dKd,
    scale,
    INF_VAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ (direct write) and dK (atomic add) for MHA backward.
    
    Softmax backward decomposition:
      dQ = scale * (C1 - r * C2)
      where C1 = sum_k(A*dA * K), C2 = sum_k(A * K), r = sum_k(A*dA)
    This allows computing dQ in a single pass over key tiles.
    dK requires a second pass with atomic accumulation.
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    q_bh = b * stride_qb + h * stride_qh
    k_bh = b * stride_kb + h * stride_kh
    v_bh = b * stride_vb + h * stride_vh
    dov_bh = b * stride_dOb + h * stride_dOh
    l_bh = b * stride_Lb + h * stride_Lh
    dq_bh = b * stride_dQb + h * stride_dQh
    dk_bh = b * stride_dKb + h * stride_dKh

    start_q = pid_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < S
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < d_head

    q_ptrs = Q_ptr + q_bh + offs_q[:, None] * stride_qs + offs_d[None, :] * stride_qd
    Q_tile = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    dov_ptrs = dO_ptr + dov_bh + offs_q[:, None] * stride_dOs + offs_d[None, :] * stride_dOd
    dO_tile = tl.load(dov_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    l_ptrs = L_ptr + l_bh + offs_q * stride_Ls
    L_tile = tl.load(l_ptrs, mask=q_mask, other=INF_VAL)

    # ---- Pass 1: accumulate C1, C2, r for softmax backward ----
    C1 = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    C2 = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    r = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    for start_k in range(0, S, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < S
        valid_qk = q_mask[:, None] & k_mask[None, :]

        k_ptrs = K_ptr + k_bh + offs_k[:, None] * stride_ks + offs_d[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_ptrs = V_ptr + v_bh + offs_k[:, None] * stride_vs + offs_d[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        logits = tl.dot(Q_tile, K_tile) * scale
        L_safe = tl.where(q_mask[:, None], L_tile[:, None], INF_VAL)
        attn = tl.exp(logits - L_safe)
        attn = tl.where(valid_qk, attn, 0.0)

        dA = tl.dot(dO_tile, V_tile.T)
        aw = attn * dA

        r = r + tl.sum(aw, axis=-1)
        C1 = tl.dot(aw, K_tile, C1)
        C2 = tl.dot(attn, K_tile, C2)

    # Store dQ = scale * (C1 - r * C2)
    dQ_out = (C1 - r[:, None] * C2) * scale
    dq_ptrs = dQ_ptr + dq_bh + offs_q[:, None] * stride_dQs + offs_d[None, :] * stride_dQd
    tl.store(dq_ptrs, dQ_out.to(tl.bfloat16), mask=q_mask[:, None] & d_mask[None, :])

    # ---- Pass 2: dK via atomics ----
    for start_k in range(0, S, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < S
        valid_qk = q_mask[:, None] & k_mask[None, :]

        k_ptrs = K_ptr + k_bh + offs_k[:, None] * stride_ks + offs_d[None, :] * stride_kd
        K_tile = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_ptrs = V_ptr + v_bh + offs_k[:, None] * stride_vs + offs_d[None, :] * stride_vd
        V_tile = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        logits = tl.dot(Q_tile, K_tile) * scale
        L_safe = tl.where(q_mask[:, None], L_tile[:, None], INF_VAL)
        attn = tl.exp(logits - L_safe)
        attn = tl.where(valid_qk, attn, 0.0)

        dA = tl.dot(dO_tile, V_tile.T)
        dS = attn * (dA - r[:, None])

        dk_contrib = tl.dot(dS.T, Q_tile) * scale
        dk_ptrs = dK_ptr + dk_bh + offs_k[:, None] * stride_dKs + offs_d[None, :] * stride_dKd
        tl.atomic_add(dk_ptrs, dk_contrib.to(tl.bfloat16),
                      mask=k_mask[:, None] & d_mask[None, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute gradients dQ, dK, dV for non-causal multi-head attention backward.

    Inputs are Q, K, V, forward output O, upstream gradient dO, and log-sum-exp L.
    Results are written into the preallocated output tensors dQ, dK, dV.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, d_head = Q.shape
    scale = 1.0 / (d_head ** 0.5)

    BLOCK_Q = 128
    BLOCK_K = 128
    BLOCK_D = 128
    INF_VAL = float("inf")

    num_bh = B * H
    num_qtiles = triton.cdiv(S, BLOCK_Q)
    num_ktiles = triton.cdiv(S, BLOCK_K)

    qs = Q.stride()
    ks = K.stride()
    vs = V.stride()
    dos = dO.stride()
    ls = L.stride()
    dqs = dQ.stride()
    dks = dK.stride()
    dvs = dV.stride()

    # Zero-initialize all outputs (dK needs it for atomics; dQ/dV for correctness when S=0)
    for out in (dQ, dK, dV):
        n = out.numel()
        if n > 0:
            z_grid = (triton.cdiv(n, 1024),)
            _fill_zeros_kernel[z_grid](out, n, BLOCK=1024)

    # Skip computation if there is no work
    if S == 0:
        return

    # Kernel 1: dV = attention^T @ dO
    _mha_bwd_dv_kernel[(num_bh, num_ktiles)](
        Q, K, dO, L, dV,
        B, H, S, d_head,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        dvs[0], dvs[1], dvs[2], dvs[3],
        scale,
        INF_VAL=INF_VAL,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # Kernel 2: dQ (direct) + dK (atomics)
    _mha_bwd_dq_dk_kernel[(num_bh, num_qtiles)](
        Q, K, V, dO, L, dQ, dK,
        B, H, S, d_head,
        qs[0], qs[1], qs[2], qs[3],
        ks[0], ks[1], ks[2], ks[3],
        vs[0], vs[1], vs[2], vs[3],
        dos[0], dos[1], dos[2], dos[3],
        ls[0], ls[1], ls[2],
        dqs[0], dqs[1], dqs[2], dqs[3],
        dks[0], dks[1], dks[2], dks[3],
        scale,
        INF_VAL=INF_VAL,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )