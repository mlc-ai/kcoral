import torch
import triton
import triton.language as tl


@triton.jit
def _bias_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    Bias_ptr,
    S,
    D,
    ATTN_SCALE,
    SQ_stride,
    SD_stride,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute Bias[b,h,:] = Sum_tgt P[b,h,:,:] * (dO[b,h,:,:] @ V[b,h,:,:]^T) per source position."""
    b = tl.program_id(0)
    h = tl.program_id(1)

    Q_base = Q_ptr + b * (H * S * D) + h * (S * D)
    K_base = K_ptr + b * (H * S * D) + h * (S * D)
    dO_base = dO_ptr + b * (H * S * D) + h * (S * D)
    V_base = V_ptr + b * (H * S * D) + h * (S * D)
    Bias_base = Bias_ptr + b * (H * S) + h * S

    qs = tl.arange(0, BLOCK_S)
    ds = tl.arange(0, BLOCK_D)
    acc_bias = tl.zeros((BLOCK_S,), dtype=tl.float32)

    for t_start in range(0, S, BLOCK_S):
        ts = t_start + tl.arange(0, BLOCK_S)

        Q_bs = tl.load(
            Q_base + qs[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(qs[:, None] < S),
            other=0.0,
        )
        K_bt = tl.load(
            K_base + ts[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(ts[:, None] < S),
            other=0.0,
        )
        dO_bs = tl.load(
            dO_base + qs[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(qs[:, None] < S),
            other=0.0,
        )
        V_bt = tl.load(
            V_base + ts[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(ts[:, None] < S),
            other=0.0,
        )

        sq = tl.dot(Q_bs, tl.trans(K_bt)) * ATTN_SCALE
        L_qs = tl.load(Bias_base + qs * SQ_stride, mask=qs < S, other=0.0)
        L_qs = tl.load(L_ptr + b * (H * S) + h * S + qs, mask=qs < S, other=0.0)
        causal_mask = qs[:, None] >= ts[None, :]
        P_sq = tl.exp(sq - L_qs[:, None])
        P_sq = tl.where(causal_mask, P_sq, 0.0)

        dP_sq = tl.dot(dO_bs, tl.trans(V_bt))
        w_sq = P_sq * dP_sq
        acc_bias += tl.sum(w_sq, axis=1)

    tl.store(Bias_base + qs, acc_bias, mask=qs < S)


@triton.jit
def _dqdk_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    O_ptr,
    Bias_ptr,
    dQ_ptr,
    dK_ptr,
    S,
    D,
    ATTN_SCALE,
    SQ_stride,
    SD_stride,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dQ and dK for causal attention backward."""
    b = tl.program_id(0)
    h = tl.program_id(1)

    Q_base = Q_ptr + b * (H * S * D) + h * (S * D)
    K_base = K_ptr + b * (H * S * D) + h * (S * D)
    dO_base = dO_ptr + b * (H * S * D) + h * (S * D)
    V_base = V_ptr + b * (H * S * D) + h * (S * D)
    O_base = O_ptr + b * (H * S * D) + h * (S * D)
    dQ_base = dQ_ptr + b * (H * S * D) + h * (S * D)
    dK_base = dK_ptr + b * (H * S * D) + h * (S * D)
    Bias_base = Bias_ptr + b * (H * S) + h * S

    qs = tl.arange(0, BLOCK_S)
    ds = tl.arange(0, BLOCK_D)

    acc_dq = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    acc_dk = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    acc_w_dot_K = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

    for t_start in range(0, S, BLOCK_S):
        ts = t_start + tl.arange(0, BLOCK_S)

        Q_bs = tl.load(
            Q_base + qs[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(qs[:, None] < S),
            other=0.0,
        )
        K_bt = tl.load(
            K_base + ts[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(ts[:, None] < S),
            other=0.0,
        )
        dO_bs = tl.load(
            dO_base + qs[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(qs[:, None] < S),
            other=0.0,
        )
        V_bt = tl.load(
            V_base + ts[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(ts[:, None] < S),
            other=0.0,
        )

        sq = tl.dot(Q_bs, tl.trans(K_bt)) * ATTN_SCALE
        L_qs = tl.load(L_ptr + b * (H * S) + h * S + qs, mask=qs < S, other=0.0)
        causal_mask = qs[:, None] >= ts[None, :]
        P_sq = tl.exp(sq - L_qs[:, None])
        P_sq = tl.where(causal_mask, P_sq, 0.0)

        dP_sq = tl.dot(dO_bs, tl.trans(V_bt))
        w_sq = P_sq * dP_sq
        acc_w_dot_K += tl.dot(w_sq, K_bt)

    Bias_qs = tl.load(Bias_base + qs, mask=qs < S, other=0.0)
    O_bs = tl.load(
        O_base + qs[:, None] * SQ_stride + ds[None, :] * SD_stride,
        mask=(qs[:, None] < S),
        other=0.0,
    )
    acc_dq = acc_w_dot_K - Bias_qs[:, None] * O_bs

    acc_dk_total = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    for t_start in range(0, S, BLOCK_S):
        ts = t_start + tl.arange(0, BLOCK_S)
        Q_bs = tl.load(
            Q_base + qs[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(qs[:, None] < S),
            other=0.0,
        )
        K_bt = tl.load(
            K_base + ts[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(ts[:, None] < S),
            other=0.0,
        )
        dO_bs = tl.load(
            dO_base + qs[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(qs[:, None] < S),
            other=0.0,
        )
        V_bt = tl.load(
            V_base + ts[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(ts[:, None] < S),
            other=0.0,
        )
        sq = tl.dot(Q_bs, tl.trans(K_bt)) * ATTN_SCALE
        L_qs = tl.load(L_ptr + b * (H * S) + h * S + qs, mask=qs < S, other=0.0)
        causal_mask = qs[:, None] >= ts[None, :]
        P_sq = tl.exp(sq - L_qs[:, None])
        P_sq = tl.where(causal_mask, P_sq, 0.0)
        dP_sq = tl.dot(dO_bs, tl.trans(V_bt))
        w_sq = P_sq * dP_sq
        acc_dk_total += tl.dot(tl.trans(w_sq), Q_bs)

    K_bs = tl.load(
        K_base + qs[:, None] * SQ_stride + ds[None, :] * SD_stride,
        mask=(qs[:, None] < S),
        other=0.0,
    )
    acc_dk = acc_dk_total - Bias_qs[:, None] * K_bs

    valid_qs = (qs < S)[:, None]
    valid_ds = (ds < D)[None, :]
    mask_sd = valid_qs & valid_ds
    tl.store(dQ_base + qs[:, None] * SQ_stride + ds[None, :] * SD_stride,
             acc_dq.to(dtype=tl.bfloat16), mask=mask_sd)
    tl.store(dK_base + qs[:, None] * SQ_stride + ds[None, :] * SD_stride,
             acc_dk.to(dtype=tl.bfloat16), mask=mask_sd)


@triton.jit
def _dv_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    L_ptr,
    dV_ptr,
    S,
    D,
    ATTN_SCALE,
    SQ_stride,
    SD_stride,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute dV for causal attention backward."""
    b = tl.program_id(0)
    h = tl.program_id(1)

    Q_base = Q_ptr + b * (H * S * D) + h * (S * D)
    K_base = K_ptr + b * (H * S * D) + h * (S * D)
    dO_base = dO_ptr + b * (H * S * D) + h * (S * D)
    dV_base = dV_ptr + b * (H * S * D) + h * (S * D)

    ts = tl.arange(0, BLOCK_S)
    ds = tl.arange(0, BLOCK_D)
    acc_dv = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        ss = s_start + tl.arange(0, BLOCK_S)

        Q_ss = tl.load(
            Q_base + ss[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(ss[:, None] < S),
            other=0.0,
        )
        K_ts = tl.load(
            K_base + ts[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(ts[:, None] < S),
            other=0.0,
        )
        dO_ss = tl.load(
            dO_base + ss[:, None] * SQ_stride + ds[None, :] * SD_stride,
            mask=(ss[:, None] < S),
            other=0.0,
        )

        sq = tl.dot(Q_ss, tl.trans(K_ts)) * ATTN_SCALE
        L_ss = tl.load(L_ptr + b * (H * S) + h * S + ss, mask=ss < S, other=0.0)
        causal_mask = ss[:, None] >= ts[None, :]
        P_st = tl.exp(sq - L_ss[:, None])
        P_st = tl.where(causal_mask, P_st, 0.0)

        acc_dv += tl.dot(tl.trans(P_st), dO_ss)

    valid_ts = (ts < S)[:, None]
    valid_ds = (ds < D)[None, :]
    mask_sd = valid_ts & valid_ds
    tl.store(dV_base + ts[:, None] * SQ_stride + ds[None, :] * SD_stride,
             acc_dv.to(dtype=tl.bfloat16), mask=mask_sd)


H = 48


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute causal multi-head attention backward: dQ, dK, dV.
    
    Args:
        Q: Query tensor [B, H, S, d]
        K: Key tensor [B, H, S, d]
        V: Value tensor [B, H, S, d]
        O: Forward output [B, H, S, d]
        dO: Upstream gradient [B, H, S, d]
        L: Logsumexp [B, H, S] (possibly with trailing dim)
        dQ: Output gradient for Q [B, H, S, d]
        dK: Output gradient for K [B, H, S, d]
        dV: Output gradient for V [B, H, S, d]
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape

    # Ensure contiguous layout in (S, d) dimension for stride arithmetic
    if not Q.is_contiguous():
        Q = Q.contiguous()
    if not K.is_contiguous():
        K = K.contiguous()
    if not V.is_contiguous():
        V = V.contiguous()
    if not O.is_contiguous():
        O = O.contiguous()
    if not dO.is_contiguous():
        dO = dO.contiguous()

    stride_seq = Q.stride(2)   # stride for sequence dimension
    stride_d = Q.stride(3)     # stride for head-dimension dimension

    ATTN_SCALE = 1.0 / (D ** 0.5)

    # Handle L: if 3D [B,H,S], unsqueeze to [B,H,S,1]; if 4D use as-is
    if L.dim() == 3:
        L = L.unsqueeze(-1)

    # Create bias buffer for intermediate computation
    device = Q.device
    Bias = torch.zeros((B, H, S), dtype=torch.float32, device=device)

    grid_bh = (B, H)
    BLOCK_S = 128
    BLOCK_D = 64

    # Pass 1: Compute bias per (b, h, src_position)
    _bias_kernel[grid_bh](
        Q, K, V, dO, L, Bias,
        S, D, ATTN_SCALE,
        stride_seq, stride_d,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
        H=H,
        num_warps=4, num_stages=3,
    )

    # Pass 2: Compute dQ and dK using bias
    _dqdk_kernel[grid_bh](
        Q, K, V, dO, L, O, Bias,
        dQ, dK,
        S, D, ATTN_SCALE,
        stride_seq, stride_d,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
        H=H,
        num_warps=4, num_stages=3,
    )

    # Pass 3: Compute dV
    _dv_kernel[grid_bh](
        Q, K, V, dO, L, dV,
        S, D, ATTN_SCALE,
        stride_seq, stride_d,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
        H=H,
        num_warps=4, num_stages=3,
    )