import torch
import triton
import triton.language as tl


@triton.jit
def _mha_bwd_dq_kernel(
    Q, K, V, dO, L, dQ,
    B, H, S, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute dQ: one program per (batch, head, query_tile)."""
    pid = tl.program_id(0)
    num_heads = H
    bh = pid // num_heads
    qi = pid % num_heads
    b = bh // num_heads
    h = bh % num_heads

    inv_sqrt_d = 1.0 / tl.sqrt(tl.float32(BLOCK_N))

    # Precompute base pointers for (b, h)
    stride_sq = S * D
    stride_sh = H * S * D
    Q_base = Q + b * stride_sh + h * stride_sq
    K_base = K + b * stride_sh + h * stride_sq
    V_base = V + b * stride_sq + h * stride_sq
    dO_base = dO + b * stride_sh + h * stride_sq
    L_base = L + b * H * S + h * S
    dQ_base = dQ + b * stride_sh + h * stride_sq

    offs_n = tl.arange(0, BLOCK_N)
    qs = qi * BLOCK_M + tl.arange(0, BLOCK_M)
    mq = qs < S

    Q_ptrs = Q_base + qs[:, None] * S + offs_n[None, :]
    dO_ptrs = dO_base + qs[:, None] * S + offs_n[None, :]
    dQ_ptrs = dQ_base + qs[:, None] * S + offs_n[None, :]

    Q_t = tl.load(Q_ptrs, mask=mq[:, None], other=0.0)
    dO_t = tl.load(dO_ptrs, mask=mq[:, None], other=0.0)
    L_t = tl.load(L_base + qs, mask=mq, other=0.0)

    num_kv_tiles = tl.cdiv(S, BLOCK_M)
    lse_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_WK = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_PK = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ki in range(num_kv_tiles):
        ks = ki * BLOCK_M + tl.arange(0, BLOCK_M)
        mk = ks < S

        K_ptrs = K_base + ks[:, None] * S + offs_n[None, :]
        V_ptrs = V_base + ks[:, None] * S + offs_n[None, :]

        K_t = tl.load(K_ptrs, mask=mk[:, None], other=0.0)
        V_t = tl.load(V_ptrs, mask=mk[:, None], other=0.0)

        scores = tl.dot(Q_t, K_t.T) * inv_sqrt_d
        attn = tl.exp(scores - L_t[:, None])
        vmask = mq[:, None] & mk[None, :]
        attn = tl.where(vmask, attn, 0.0)

        dattn = tl.dot(dO_t, V_t.T)
        W = attn * dattn

        lse_acc = lse_acc + tl.sum(W, axis=1)
        acc_WK = acc_WK + tl.dot(W.to(tl.bfloat16), K_t)
        acc_PK = acc_PK + tl.dot(attn.to(tl.bfloat16), K_t)

    dQ_val = ((acc_WK - lse_acc[:, None] * acc_PK) * inv_sqrt_d).to(tl.bfloat16)
    tl.store(dQ_ptrs, dQ_val, mask=mq[:, None])


@triton.jit
def _mha_bwd_dkv_kernel(
    Q, K, V, dO, L, dK, dV,
    B, H, S, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute dK and dV via atomics: one program per (batch, head, kv_tile)."""
    pid = tl.program_id(0)
    num_heads = H
    bh = pid // num_heads
    ki = pid % num_heads
    b = bh // num_heads
    h = bh % num_heads

    inv_sqrt_d = 1.0 / tl.sqrt(tl.float32(BLOCK_N))

    stride_sq = S * D
    stride_sh = H * S * D
    Q_base = Q + b * stride_sh + h * stride_sq
    K_base = K + b * stride_sh + h * stride_sq
    V_base = V + b * stride_sq + h * stride_sq
    dO_base = dO + b * stride_sh + h * stride_sq
    L_base = L + b * H * S + h * S
    dK_base = dK + b * stride_sh + h * stride_sq
    dV_base = dV + b * stride_sh + h * stride_sq

    offs_n = tl.arange(0, BLOCK_N)
    ks = ki * BLOCK_M + tl.arange(0, BLOCK_M)
    mk = ks < S

    K_ptrs = K_base + ks[:, None] * S + offs_n[None, :]
    V_ptrs = V_base + ks[:, None] * S + offs_n[None, :]
    dK_ptrs = dK_base + ks[:, None] * S + offs_n[None, :]
    dV_ptrs = dV_base + ks[:, None] * S + offs_n[None, :]

    K_t = tl.load(K_ptrs, mask=mk[:, None], other=0.0)
    V_t = tl.load(V_ptrs, mask=mk[:, None], other=0.0)

    num_q_tiles = tl.cdiv(S, BLOCK_M)
    lse_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for qi in range(num_q_tiles):
        qs = qi * BLOCK_M + tl.arange(0, BLOCK_M)
        mq = qs < S

        Q_ptrs = Q_base + qs[:, None] * S + offs_n[None, :]
        dO_ptrs = dO_base + qs[:, None] * S + offs_n[None, :]

        Q_t = tl.load(Q_ptrs, mask=mq[:, None], other=0.0)
        dO_t = tl.load(dO_ptrs, mask=mq[:, None], other=0.0)
        L_t = tl.load(L_base + qs, mask=mq, other=0.0)

        scores = tl.dot(Q_t, K_t.T) * inv_sqrt_d
        attn = tl.exp(scores - L_t[:, None])
        vmask = mq[:, None] & mk[None, :]
        attn = tl.where(vmask, attn, 0.0)

        dattn = tl.dot(dO_t, V_t.T)
        W = attn * dattn
        lse_acc = lse_acc + tl.sum(W, axis=1)

        adjusted = attn * lse_acc[:, None]

        tl.atomic_add(
            dK_ptrs,
            (tl.dot((W - adjusted).to(tl.bfloat16).T, Q_t) * inv_sqrt_d).to(tl.bfloat16),
            mask=mk[:, None],
        )
        tl.atomic_add(
            dV_ptrs,
            (tl.dot(attn.to(tl.bfloat16).T, dO_t)).to(tl.bfloat16),
            mask=mk[:, None],
        )


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Multi-head attention backward: compute dQ, dK, dV."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    if L.dim() == 4:
        L = L.squeeze(-1)

    # Ensure contiguous layout matching our stride assumptions
    # We assume row-major contiguous: stride(S)=D, stride(H)=S*D, stride(B)=H*S*D
    
    BLOCK_M = 64

    num_q_tiles = triton.cdiv(S, BLOCK_M)
    num_kv_tiles = triton.cdiv(S, BLOCK_M)

    extra_args_dq = {"num_warps": 4, "num_stages": 2}
    extra_args_dkv = {"num_warps": 4, "num_stages": 2}

    grid_dq = (B * H * num_q_tiles,)
    _mha_bwd_dq_kernel[grid_dq](
        Q, K, V, dO, L, dQ,
        B, H, S, D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=D,
        **extra_args_dq,
    )

    dK.zero_()
    dV.zero_()

    grid_dkv = (B * H * num_kv_tiles,)
    _mha_bwd_dkv_kernel[grid_dkv](
        Q, K, V, dO, L, dK, dV,
        B, H, S, D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=D,
        **extra_args_dkv,
    )