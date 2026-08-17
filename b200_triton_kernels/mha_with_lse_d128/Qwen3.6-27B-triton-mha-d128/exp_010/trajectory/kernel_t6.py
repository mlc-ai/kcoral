import torch
import triton
import triton.language as tl


@triton.jit
def _attn(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,
    stride_lm,
    S, SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n_base = tl.arange(0, BLOCK_N)

    mask_m = offs_m < S

    # View-based pointers: tensors are already [BH, S, D], so program_id(0) is BH group
    # But kernel receives raw ptrs — the view's underlying storage stride handles BH grouping
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    n_kv = tl.cdiv(S, BLOCK_N)

    for j in range(n_kv):
        offs_n = j * BLOCK_N + offs_n_base
        mask_n = offs_n < S

        k_ptrs = k_ptr + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        v_ptrs = v_ptr + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        attn = tl.dot(q, k.T, out_dtype=tl.float32) * SCALE
        m_cur = tl.max(attn, axis=1)

        alpha = tl.exp(m_i - m_cur)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        p = tl.exp(attn - m_cur[:, None])
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
        l_i = l_i + tl.sum(p, axis=1)
        m_i = m_cur

    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    o_val = (acc * inv_l[:, None]).to(tl.bfloat16)

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_val, mask=mask_m[:, None])

    lse_val = m_i + tl.log(tl.where(l_i > 0, l_i, 1.0))
    lse_ptrs = lse_ptr + offs_m * stride_lm
    tl.store(lse_ptrs, lse_val, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    n_bh = B * H
    scale = 1.0 / float(D ** 0.5)
    BLOCK_M, BLOCK_N = 64, 64

    # Flatten (B,H) -> contiguous [n_bh, S, D]; LSE -> [n_bh, S]
    Qf = Q.reshape(n_bh, S, D).contiguous()
    Kf = K.reshape(n_bh, S, D).contiguous()
    Vf = V.reshape(n_bh, S, D).contiguous()
    Of = O.reshape(n_bh, S, D)
    Lf = LSE.reshape(n_bh, S)

    grid = (n_bh, triton.cdiv(S, BLOCK_M))

    _attn[grid](
        Qf, Kf, Vf, Of, Lf,
        Qf.stride(0), Qf.stride(1),
        Kf.stride(0), Kf.stride(1),
        Vf.stride(0), Vf.stride(1),
        Of.stride(0), Of.stride(1),
        Lf.stride(0),
        S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
        num_warps=8,
        num_stages=3,
    )