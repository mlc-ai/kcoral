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
    pid = tl.program_id(0)
    off_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + off_m[:, None] * stride_qm + off_d[None, :] * stride_qd
    q_mask = (off_m < S)[:, None]
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -1e10, tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    off_n_base = tl.arange(0, BLOCK_N)
    n_tiles = tl.cdiv(S, BLOCK_N)

    for j in range(n_tiles):
        off_n = j * BLOCK_N + off_n_base
        n_mask = (off_n < S)

        k_ptrs = k_ptr + off_n[:, None] * stride_km + off_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        v_ptrs = v_ptr + off_n[:, None] * stride_vm + off_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        s = tl.dot(q, k.T) * SCALE
        m_new = tl.max(s, axis=1)

        e_old = tl.exp(m_i - m_new)
        acc_o = acc_o * e_old[:, None]
        l_i = l_i * e_old

        p = tl.exp(s - m_new[:, None])
        acc_o = tl.dot(p, v, acc=acc_o)
        l_i = l_i + tl.sum(p, axis=1)
        m_i = m_new

    inv_l = 1.0 / tl.where(l_i > 0, l_i, 1.0)
    o_val = acc_o * inv_l[:, None]

    o_ptrs = out_ptr + off_m[:, None] * stride_om + off_d[None, :] * stride_od
    tl.store(o_ptrs, o_val.to(tl.bfloat16), mask=q_mask)

    lse_val = m_i + tl.log(tl.where(l_i > 0, l_i, 1.0))
    tl.store(lse_ptr + off_m * stride_lm, lse_val, mask=off_m < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    Q3 = Q.view(B * H, S, D)
    K3 = K.view(B * H, S, D)
    V3 = V.view(B * H, S, D)
    O3 = O.view(B * H, S, D)
    L2 = LSE.view(B * H, S)

    n_bh = B * H
    scale = 1.0 / float(D ** 0.5)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (n_bh, triton.cdiv(S, BLOCK_M))

    _attn[grid](
        Q3, K3, V3, O3, L2,
        Q3.stride(0), Q3.stride(1),
        K3.stride(0), K3.stride(1),
        V3.stride(0), V3.stride(1),
        O3.stride(0), O3.stride(1),
        L2.stride(0),
        S, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
        num_warps=4,
        num_stages=3,
    )