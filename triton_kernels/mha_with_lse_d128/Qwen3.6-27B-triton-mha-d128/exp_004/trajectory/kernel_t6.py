import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    seq_len, num_heads,
    SCALE_INV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    # Offsets for indexing
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)     # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)                        # [BLOCK_N]  
    offs_d = tl.arange(0, BLOCK_D)                        # [BLOCK_D]

    # 2D masks for load/store boundary checks
    m_mask = offs_m[:, None] < seq_len                     # [BLOCK_M, 1]
    d_mask = offs_d[None, :] < BLOCK_D                     # [1, BLOCK_D], always true

    # Base pointers offset by batch and head
    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
    o_base = O_ptr + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE_ptr + pid_b * stride_lseb + pid_h * stride_lseh

    # Load Q tile once (before KV loop)
    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q_tile = tl.load(q_ptrs, mask=m_mask & d_mask, other=0.0)

    # Online softmax accumulators (fp32)
    acc_o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    mi = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)
    li = tl.full([BLOCK_M], value=1.0, dtype=tl.float32)

    # Iterate over sequence blocks for K and V
    num_kv_steps = tl.cdiv(seq_len, BLOCK_N)
    for step in range(num_kv_steps):
        start_n = step * BLOCK_N
        cur_offs_n = start_n + offs_n                      # [BLOCK_N]
        n_mask = cur_offs_n < seq_len                       # [BLOCK_N]
        
        # Load K tile [BLOCK_N, BLOCK_D]
        k_ptrs = k_base + cur_offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        # Attention scores: Q @ K^T / sqrt(D) -> [BLOCK_M, BLOCK_N]
        s = tl.dot(q_tile, k_tile) * SCALE_INV

        # Mask out invalid columns with large negative value
        col_neg_inf = tl.full((BLOCK_M, BLOCK_N), value=-5e4, dtype=tl.float32)
        s = tl.where(n_mask[None, :], s, col_neg_inf)

        # Online softmax update
        mi_old = mi                                           # [BLOCK_M]
        row_max = tl.max(s, axis=1)                            # [BLOCK_M]
        mi_new = tl.maximum(mi, row_max)                       # [BLOCK_M]

        alpha = tl.exp(mi_old - mi_new)                        # [BLOCK_M]
        p_block = tl.exp(s - mi_new[:, None])                  # [BLOCK_M, BLOCK_N]

        li_new = li * alpha + tl.sum(p_block, axis=1)          # [BLOCK_M]

        # Load V tile [BLOCK_N, BLOCK_D]
        v_ptrs = v_base + cur_offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        # Weighted accumulation: acc_o <- acc_o * alpha + p @ V
        acc_o = acc_o * alpha[:, None] + tl.dot(p_block.to(tl.bfloat16), v_tile)

        mi = mi_new
        li = li_new

    # Final normalization
    inv_li = tl.reciprocal(li)                               # [BLOCK_M]
    acc_o = acc_o * inv_li[:, None]

    # LSE = max + log(sum(exp(P - max)))
    lse_out = mi + tl.log(li)

    # Store output O [BLOCK_M, BLOCK_D]
    o_ptrs = o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=m_mask & d_mask)

    # Store LSE [BLOCK_M]
    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse_out, mask=(offs_m < seq_len))


def run(Q, K, V, O, LSE):
    """
    Multi-head attention forward pass:
      O = softmax(Q @ K^T / sqrt(D)) @ V
      LSE = logsumexp(Q @ K^T / sqrt(D), dim=-1)
    
    Destination-passing interface: writes into preallocated O and LSE tensors.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    scale_inv = 1.0 / float(D ** 0.5)

    grid = (B * H, triton.cdiv(S, BLOCK_M))

    _mha_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H,
        SCALE_INV=scale_inv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )