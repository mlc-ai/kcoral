import torch
import triton
import triton.language as tl


def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)


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
    num_tiles_per_bh = tl.cdiv(S, BLOCK_M)
    total_bh = tl.program_id(0)
    bid = total_bh // (H * num_tiles_per_bh)
    remainder = total_bh % (H * num_tiles_per_bh)
    hid = remainder // num_tiles_per_bh
    pid_m = remainder % num_tiles_per_bh

    off_m = tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)

    mask_m = (pid_m * BLOCK_M + off_m) < S
    q_row_offset = pid_m * BLOCK_M + off_m

    q_base = Q_ptr + bid * stride_qb + hid * stride_qh
    k_base = K_ptr + bid * stride_kb + hid * stride_kh
    v_base = V_ptr + bid * stride_vb + hid * stride_vh
    o_base = O_ptr + bid * stride_ob + hid * stride_oh
    lse_base = LSE_ptr + bid * stride_lsb + hid * stride_lsh

    # Create per-head tensor descriptors
    Q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero")

    # Load Q once
    q = Q_desc.load([pid_m * BLOCK_M, 0])

    m_i = tl.full((BLOCK_M,), float("-inv"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_kv_iters = tl.cdiv(S, BLOCK_N)

    for start_n in range(num_kv_iters):
        offset_n = start_n * BLOCK_N

        k = K_desc.load([offset_n, 0])
        v = V_desc.load([offset_n, 0])

        s = tl.dot(q, k.T, out_dtype=tl.float32)
        s = s * inv_scale

        m_ij = tl.max(s, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)

        p = tl.exp(s - m_i_new[:, None])

        old_scale = tl.exp(m_i - m_i_new)
        l_i_new = old_scale * l_i + tl.sum(p, axis=1)
        acc_o = old_scale[:, None] * acc_o + tl.dot(
            p.to(tl.bfloat16), v, acc=None, out_dtype=tl.float32)

        m_i = m_i_new
        l_i = l_i_new

    # Store output O
    o_final = acc_o / l_i[:, None]
    o_stored = o_final.to(tl.bfloat16)

    o_ptrs = (o_base
              + q_row_offset[:, None] * stride_os
              + off_d[None, :] * stride_od)
    tl.store(o_ptrs, o_stored, mask=mask_m[:, None])

    # Store LSE
    lse_val = m_i + tl.log(l_i)
    lse_ptrs = lse_base + q_row_offset * stride_lss
    tl.store(lse_ptrs, lse_val, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

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