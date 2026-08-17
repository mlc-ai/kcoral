import torch
import triton
import triton.language as tl


@triton.jit
def _attn(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    stride_qbh, stride_qm, stride_qd,
    stride_kbh, stride_km, stride_kd,
    stride_vbh, stride_vm, stride_vd,
    stride_obh, stride_om, stride_od,
    stride_lbh, stride_lm,
    S, SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bh_pid = tl.program_id(0)
    m_pid = tl.program_id(1)
    offs_m = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n_base = tl.arange(0, BLOCK_N)

    mask_m = offs_m < S

    # Compute base pointer offset for this (batch, head)
    bh_off = bh_pid * stride_qbh
    q_ptrs = q_ptr + bh_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + offs_n_base
        mask_n = offs_n < S

        k_ptrs = k_ptr + bh_pid * stride_kbh + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        v_ptrs = v_ptr + bh_pid * stride_vbh + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd
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

    o_ptrs = out_ptr + bh_pid * stride_obh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_val, mask=mask_m[:, None])

    lse_val = m_i + tl.log(tl.where(l_i > 0, l_i, 1.0))
    lse_ptrs = lse_ptr + bh_pid * stride_lbh + offs_m * stride_lm
    tl.store(lse_ptrs, lse_val, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    n_bh = B * H
    scale = 1.0 / float(D ** 0.5)
    BLOCK_M, BLOCK_N = 64, 64

    grid = (n_bh, triton.cdiv(S, BLOCK_M))

    _attn[grid](
        Q, K, V, O, LSE,
        Q.stride(0) * H, Q.stride(2), Q.stride(3),
        K.stride(0) * H, K.stride(2), K.stride(3),
        V.stride(0) * H, V.stride(2), V.stride(3),
        O.stride(0) * H, O.stride(2), O.stride(3),
        LSE.stride(0) * H, LSE.stride(2),
        S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
        num_warps=8,
        num_stages=3,
    )