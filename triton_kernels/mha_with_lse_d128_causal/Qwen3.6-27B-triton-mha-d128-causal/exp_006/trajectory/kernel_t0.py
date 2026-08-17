import torch
import triton
import triton.language as tl


@triton.jit
def _mha_causal_fwd_kernel(
    Q,
    K,
    V,
    Out,
    LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lsb, stride_lsh, stride_lss,
    B, H, S, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Causal MHA forward kernel using FlashAttention-style online softmax.
    
    Each program instance handles one (batch, head) pair and one tile of query
    sequence positions. It iterates over all key tiles, computing attention
    scores with causal masking and accumulating the output.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query offsets (absolute sequence positions)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_boundary = offs_m < S

    # Key offset template (relative within a block)
    offs_n_init = tl.arange(0, BLOCK_N)

    # Head dimension offsets
    offs_d = tl.arange(0, D)

    # Base pointers for this batch/head combination
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = Out + pid_b * stride_ob + pid_h * stride_oh
    lse_base = LSE + pid_b * stride_lsb + pid_h * stride_lsh

    # ===== Load Q tile once: shape [BLOCK_M, D] =====
    q_row_off = offs_m[:, None] * stride_qs
    q_col_off = offs_d[None, :] * stride_qd
    q_ptrs = q_base + q_row_off + q_col_off
    Q_tile = tl.load(q_ptrs, mask=m_boundary[:, None], other=0.0)

    # ===== Initialize accumulators =====
    # Output accumulator: [BLOCK_M, D] in fp32
    acc_O = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    # Running max and log-sum-exp denominator per row: [BLOCK_M]
    acc_m = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    acc_l = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # ===== Main loop over key/value tiles =====
    loffs_n = offs_n_init  # mutable copy for advancing

    start_n = 0
    while start_n < S:
        # Absolute key positions for this tile
        loffs_n = start_n + offs_n_init
        n_boundary = loffs_n < S

        # === Load K tile [BLOCK_N, D] ===
        k_row_off = loffs_n[:, None] * stride_ks
        k_col_off = offs_d[None, :] * stride_kd
        k_ptrs = k_base + k_row_off + k_col_off
        K_tile = tl.load(k_ptrs, mask=n_boundary[:, None], other=0.0)

        # === Load V tile [BLOCK_N, D] ===
        v_row_off = loffs_n[:, None] * stride_vs
        v_col_off = offs_d[None, :] * stride_vd
        v_ptrs = v_base + v_row_off + v_col_off
        V_tile = tl.load(v_ptrs, mask=n_boundary[:, None], other=0.0)

        # === Compute attention scores [BLOCK_M, BLOCK_N] ===
        # Q_tile: [BLOCK_M, D], K_tile.T: [D, BLOCK_N] -> scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(Q_tile, K_tile.T, input_precision="tf32") * scale

        # === Apply causal mask ===
        # Causal: query position >= key position
        # offs_m: [BLOCK_M], loffs_n: [BLOCK_N]
        causal_mask = offs_m[:, None] >= loffs_n[None, :]

        # Combined mask: causal AND in-sequence boundaries
        full_mask = causal_mask & n_boundary[None, :] & m_boundary[:, None]

        # Invalid/upper-triangular positions get -inf
        scores = tl.where(full_mask, scores, float("-inf"))

        # === Online softmax (per-row numerical stability) ===
        # Current row-wise max
        cur_m = tl.max(scores, axis=1)  # [BLOCK_M]

        # New global max = max(old_acc_max, cur_tile_max)
        new_m = tl.maximum(acc_m, cur_m)  # [BLOCK_M]

        # Scaling factor: exp(old_max - new_max)
        # This handles the online softmax update correctly
        alpha = tl.exp(acc_m - new_m)  # [BLOCK_M]

        # Exp(scale * scores) shifted to the new max frame
        # exp(scores - new_m) = exp(scores - cur_m) * exp(cur_m - new_m)
        p = tl.exp(scores - new_m[:, None])  # [BLOCK_M, BLOCK_N]

        # === Accumulate output: O += P @ V_tile ===
        # P: [BLOCK_M, BLOCK_N], V_tile: [BLOCK_N, D] -> [BLOCK_M, D]
        acc_O = tl.dot(p, V_tile, acc=acc_O, input_precision="tf32")

        # === Update running sum of exponentials ===
        p_sum = tl.sum(p, axis=1)  # [BLOCK_M]
        acc_l = alpha * acc_l + p_sum

        # Move to next tile
        acc_m = new_m
        start_n += BLOCK_N

    # ===== Normalize output and compute LSE =====
    # Guard against div-by-zero for positions where no valid keys exist
    # (shouldn't happen in causal attn, but safe to handle)
    denom_inv = tl.where(acc_l > 0.0, 1.0 / acc_l, 0.0)
    Out_val = acc_O * denom_inv[:, None]

    # Store output [BLOCK_M, D]
    o_row_off = offs_m[:, None] * stride_os
    o_col_off = offs_d[None, :] * stride_od
    o_ptrs = o_base + o_row_off + o_col_off
    tl.store(o_ptrs, Out_val.to(tl.bfloat16), mask=m_boundary[:, None])

    # Store LSE [BLOCK_M]: max + log(sum_exp)
    lse_ptrs = lse_base + offs_m * stride_lss
    LSE_val = tl.where(
        acc_l > 0.0,
        acc_m + tl.log(acc_l),
        float("-inf"),
    )
    tl.store(lse_ptrs, LSE_val, mask=m_boundary)


def run(Q, K, V, O, LSE):
    """
    Causal multi-head attention forward pass.
    
    Computes:
      O = softmax(Q @ K^T / sqrt(D), causal_mask) @ V
      LSE = logsumexp(Q @ K^T / sqrt(D), causal_mask, dim=-1)
    
    All inputs are bf16. Output O is bf16, LSE is float32.
    Causal mask is lower-triangular over the shared sequence dimension.
    """
    torch.cuda.set_device(Q.device)

    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D), f"K shape {K.shape} != {(B, H, S, D)}"
    assert V.shape == (B, H, S, D), f"V shape {V.shape} != {(B, H, S, D)}"
    assert O.shape == (B, H, S, D), f"O shape {O.shape} != {(B, H, S, D)}"
    assert LSE.shape == (B, H, S), f"LSE shape {LSE.shape} != {(B, H, S)}"

    # Strides in elements
    stride_qb, stride_qh, stride_qs, stride_qd = Q.stride()
    stride_kb, stride_kh, stride_ks, stride_kd = K.stride()
    stride_vb, stride_vh, stride_vs, stride_vd = V.stride()
    stride_ob, stride_oh, stride_os, stride_od = O.stride()
    stride_lsb, stride_lsh, stride_lss = LSE.stride()

    # Scale factor for attention scores
    scale = 1.0 / (float(D) ** 0.5)

    # Tile sizes -- tuned for D=128 and Hopper tensor cores
    BLOCK_M = 64  # query sequence tile size
    BLOCK_N = 64  # key sequence tile size

    # Grid: axis 0 = batch*head, axis 1 = query sequence tiles
    num_pid_bh = B * H
    num_pid_m = triton.cdiv(S, BLOCK_M)
    grid = (num_pid_bh, num_pid_m)

    # Launch kernel with 4 warps for Hopper WGMMA compatibility
    _mha_causal_fwd_kernel[grid](
        Q, K, V,
        O, LSE,
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        stride_lsb, stride_lsh, stride_lss,
        B, H, S, D,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )