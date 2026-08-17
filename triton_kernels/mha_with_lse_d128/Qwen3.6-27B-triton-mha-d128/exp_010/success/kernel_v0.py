import torch
import triton
import triton.language as tl


@triton.jit
def _attn(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    seq_len, scale,
    stride_qh, stride_qs, stride_qd,
    stride_kh, stride_ks, stride_kd,
    stride_vh, stride_vs, stride_vd,
    stride_oh, stride_os, stride_od,
    stride_lh, stride_ls,
    NUM_BH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles_m = tl.cdiv(seq_len, BLOCK_M)
    bh = pid // num_tiles_m
    tile_m = pid % num_tiles_m

    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n_base = tl.arange(0, BLOCK_N)

    mask_m = offs_m < seq_len
    bh_mask = bh < NUM_BH

    # Compute BH-stride offset
    q_off = bh * stride_qh
    k_off = bh * stride_kh
    v_off = bh * stride_vh
    o_off = bh * stride_oh
    l_off = bh * stride_lh

    q_ptrs = q_ptr + q_off + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(mask_m & bh_mask)[:, None], other=0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    n_kv_tiles = tl.cdiv(seq_len, BLOCK_N)
    for j in range(n_kv_tiles):
        offs_n = j * BLOCK_N + offs_n_base
        mask_n = offs_n < seq_len

        k_ptrs = k_ptr + k_off + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=(mask_n & bh_mask)[:, None], other=0.0)

        v_ptrs = v_ptr + v_off + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(mask_n & bh_mask)[:, None], other=0.0)

        attn = tl.dot(q, k.T, out_dtype=tl.float32) * scale
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

    o_ptrs = out_ptr + o_off + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_val, mask=(mask_m & bh_mask)[:, None])

    lse_val = m_i + tl.log(tl.where(l_i > 0, l_i, 1.0))
    lse_ptrs = lse_ptr + l_off + offs_m * stride_ls
    tl.store(lse_ptrs, lse_val, mask=mask_m & bh_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    n_bh = B * H
    scale = 1.0 / float(D ** 0.5)
    BLOCK_M, BLOCK_N = 64, 64

    num_tiles_m = triton.cdiv(S, BLOCK_M)
    grid = (n_bh * num_tiles_m,)

    _attn[grid](
        Q, K, V, O, LSE,
        S, scale,
        Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(1), K.stride(2), K.stride(3),
        V.stride(1), V.stride(2), V.stride(3),
        O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(1), LSE.stride(2),
        NUM_BH=n_bh,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D,
        num_warps=8,
        num_stages=3,
    )