import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lsb, stride_lsh, stride_lss,
    B, H, S, D,
    inv_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Decompose 1D program id into (batch, head, q_tile)
    num_tiles_per_bh = tl.cdiv(S, BLOCK_M)
    total_bh = tl.program_id(0)
    bid = total_bh // (H * num_tiles_per_bh)
    remainder = total_bh % (H * num_tiles_per_bh)
    hid = remainder // num_tiles_per_bh
    pid_m = remainder % num_tiles_per_bh

    off_m = tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)

    row_idx = pid_m * BLOCK_M + off_m
    mask_m = row_idx < S
    m_mask_2d = mask_m[:, None] & (off_d[None, :] < D)

    # Compute base offsets once
    q_base = Q_ptr + bid * stride_qb + hid * stride_qh
    k_base = K_ptr + bid * stride_kb + hid * stride_kh
    v_base = V_ptr + bid * stride_vb + hid * stride_vh
    o_base = O_ptr + bid * stride_ob + hid * stride_oh
    lse_base = LSE_ptr + bid * stride_lsb + hid * stride_lsh

    # Load Q once per program - [BLOCK_M, BLOCK_D]
    q_ptrs = (q_base
              + row_idx[:, None] * stride_qs
              + off_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=m_mask_2d, other=0.0)

    # Initialize online softmax accumulators (fp32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        n_off = start_n + tl.arange(0, BLOCK_N)
        mask_n = n_off < S
        n_mask_2d = mask_n[:, None] & (off_d[None, :] < D)

        # Load K [BLOCK_N, BLOCK_D]
        k_ptrs = (k_base
                  + n_off[:, None] * stride_ks
                  + off_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=n_mask_2d, other=0.0)

        # Load V [BLOCK_N, BLOCK_D]  
        v_ptrs = (v_base
                  + n_off[:, None] * stride_vs
                  + off_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=n_mask_2d, other=0.0)

        # s = q @ k.T  -> [BLOCK_M, BLOCK_N] in fp32
        s = tl.dot(q, k.T, out_dtype=tl.float32)
        s = s * inv_scale

        # Online softmax: update running max
        m_ij = tl.max(s, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        # Compute exponentiated attention weights (fp32)
        p = tl.exp(s - m_i_new[:, None])

        # Update accumulators
        old_scale = tl.exp(m_i - m_i_new)
        l_i_new = old_scale * l_i + tl.sum(p, axis=1)

        # p @ v: cast p to bf16 for dot, accumulate in fp32
        acc_o = old_scale[:, None] * acc_o + tl.dot(
            p.to(tl.bfloat16), v, acc=None, out_dtype=tl.float32)

        m_i = m_i_new
        l_i = l_i_new

    # Normalize output
    o_final = acc_o / l_i[:, None]
    o_stored = o_final.to(tl.bfloat16)

    # Store output O [BLOCK_M, BLOCK_D]
    o_ptrs = (o_base
              + row_idx[:, None] * stride_os
              + off_d[None, :] * stride_od)
    tl.store(o_ptrs, o_stored, mask=m_mask_2d)

    # Store LSE [BLOCK_M]
    lse_val = m_i + tl.log(l_i)
    lse_ptrs = lse_base + row_idx * stride_lss
    tl.store(lse_ptrs, lse_val, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D  # = 128

    num_tiles_per_bh = triton.cdiv(S, BLOCK_M)
    grid = (B * H * num_tiles_per_bh,)

    inv_scale = 1.0 / (D ** 0.5)

    _attention_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        inv_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=8,
        num_stages=4,
    )