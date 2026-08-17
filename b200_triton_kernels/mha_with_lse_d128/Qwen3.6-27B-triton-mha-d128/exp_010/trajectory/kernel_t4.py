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
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    # Load Q tile — keep bf16 for dot, scale later
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    mask_m = offs_m < S
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Init accumulators in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < S

        k_ptrs = k_ptr + cur_n[:, None] * stride_km + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        v_ptrs = v_ptr + cur_n[:, None] * stride_vm + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # q@K^T in bf16->fp32 accumulator; result is fp32 [BM,BN]
        attn = tl.dot(q, k.T, out_dtype=tl.float32) * SCALE

        # Online softmax update
        m_cur = tl.max(attn, axis=1)
        alpha = tl.exp(m_i - m_cur)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        p = tl.exp(attn - m_cur[:, None])
        # P @ V: fp32 x bf16 -> fp32 (or bf16 x bf16 -> fp32 depending on lowering)
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
        l_i = l_i + tl.sum(p, axis=1)
        m_i = m_cur

    # Finalize output
    inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
    o_val = (acc * inv_l[:, None]).to(tl.bfloat16)

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_val, mask=mask_m[:, None])

    lse_val = m_i + tl.log(tl.where(l_i > 0, l_i, 1.0))
    tl.store(lse_ptr + offs_m * stride_lm, lse_val, mask=mask_m)


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
        num_warps=8,
        num_stages=3,
    )