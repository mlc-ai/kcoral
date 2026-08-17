import torch
import triton
import triton.language as tl


@triton.jit
def _causal_mha_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    stride_qm, stride_qn,
    stride_km, stride_kn,
    stride_vm, stride_vn,
    stride_om, stride_on,
    stride_lm,
    S,
    D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Determine which (batch, head) and query-block this program owns
    bh_id = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Offsets for Q/V/K within the BHSD layout
    q_ptr_bh = q_ptr + bh_id * stride_qm
    k_ptr_bh = k_ptr + bh_id * stride_km
    v_ptr_bh = v_ptr + bh_id * stride_vm
    o_ptr_bh = o_ptr + bh_id * stride_om
    lse_ptr_bh = lse_ptr + bh_id * stride_lm

    # Query row offsets for this tile
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_in_bounds = off_m < S

    # Column offsets for iterating over K/V tiles
    offs_d = tl.arange(0, BLOCK_D)

    # Pointers for Q tile [BLOCK_M, BLOCK_D]
    q_ptrs = q_ptr_bh + off_m[:, None] * stride_qm + offs_d[None, :] * stride_qn
    q_tile = tl.load(q_ptrs, mask=q_in_bounds[:, None] & (offs_d[None, :] < D), other=0.0)

    # Initialize accumulators for online softmax
    # m_i: running max, l_i: running sum of exp(s-max), o_acc: weighted sum
    m_i = tl.full((BLOCK_M, 1), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    o_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    scale = 1.0 / tl.sqrt(D).to(tl.float32)

    # Iterate over K/V tiles along the sequence dimension
    num_k_tiles = tl.cdiv(S, BLOCK_N)
    for start_n in range(num_k_tiles):
        off_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
        n_in_bounds = off_n < S

        # Load K tile -> [BLOCK_D, BLOCK_N]
        k_ptrs = k_ptr_bh + off_n[None, :] * stride_km + offs_d[:, None] * stride_kn
        k_tile = tl.load(k_ptrs, mask=n_in_bounds[None, :] & (offs_d[:, None] < D), other=0.0)

        # Load V tile -> [BLOCK_N, BLOCK_D]
        v_ptrs = v_ptr_bh + off_n[:, None] * stride_vm + offs_d[None, :] * stride_vn
        v_tile = tl.load(v_ptrs, mask=n_in_bounds[:, None] & (offs_d[None, :] < D), other=0.0)

        # Compute attention scores S = Q @ K^T / sqrt(D) -> [BLOCK_M, BLOCK_N]
        s = tl.dot(q_tile, k_tile) * scale

        # Build causal + boundary mask
        # causal: query_pos >= key_pos (lower triangular)
        causal_mask = off_m[:, None] >= off_n[None, :]
        valid_mask = q_in_bounds[:, None] & n_in_bounds[None, :] & causal_mask

        # Masked scores: -inf for invalid positions
        masked_s = tl.where(valid_mask, s, float("-inf"))

        # Online softmax step
        rowmax = tl.max(masked_s, axis=1, keepdims=True)
        new_m = tl.maximum(m_i, rowmax)

        # Scale old accumulators
        alpha = tl.exp(m_i - new_m)

        # New probability contributions
        beta = tl.exp(masked_s - new_m)

        # Update LSE running stats
        new_l = alpha * l_i + tl.sum(beta, axis=1, keepdims=True)

        # Update output accumulator
        o_scaled = o_acc * alpha
        o_acc = o_scaled + tl.dot(beta, v_tile)

        m_i = new_m
        l_i = new_l

    # Final normalization: O = o_acc / l_i
    # Handle edge case where l_i might be 0 (no valid attention targets)
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    o_final = o_acc / l_safe

    # Write output O
    o_ptrs = o_ptr_bh + off_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, o_final.to(tl.bfloat16), mask=q_in_bounds[:, None])

    # Write LSE: m_i + log(l_i) with shape [S] per (batch, head)
    lse_val = (m_i + tl.log(l_i)).flatten()
    lse_ptrs = lse_ptr_bh + off_m * stride_lm
    tl.store(lse_ptrs, lse_val, mask=q_in_bounds)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward returning O and LSE.
    
    Q, K, V: (B, H, S, D) bfloat16
    O: (B, H, S, D) bfloat16 (preallocated output)
    LSE: (B, H, S) float32 (preallocated output)
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    assert Q.dtype == torch.bfloat16
    assert K.dtype == torch.bfloat16
    assert V.dtype == torch.bfloat16
    assert O.dtype == torch.bfloat16
    assert LSE.dtype == torch.float32
    
    BH = B * H
    
    # Block sizes tuned for Hopper with D=128
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D
    
    num_pid_m = triton.cdiv(S, BLOCK_M)
    
    grid = (BH, num_pid_m)
    
    _causal_mha_kernel[grid](
        Q,
        K,
        V,
        O,
        LSE,
        Q.stride(2), Q.stride(3),
        K.stride(2), K.stride(3),
        V.stride(2), V.stride(3),
        O.stride(2), O.stride(3),
        LSE.stride(2),
        S,
        D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=3,
    )