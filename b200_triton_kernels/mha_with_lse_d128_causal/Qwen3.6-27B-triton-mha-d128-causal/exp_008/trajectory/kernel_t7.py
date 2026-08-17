import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _flash_attn_fwd_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    LSE_desc,
    seqlen,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """FlashAttention-3 style causal kernel using tensor descriptors for TMA."""

    pid = tl.program_id(0)

    # Grouped scheduling: map linear pid to (batch, head, m_block)
    total_bh = NUM_SMS  # Will be overridden by grid
    # Count tiles per (b,h) pair
    num_pid_m = tl.cdiv(seqlen, BLOCK_M)
    
    # Persistent grid: assign batches and heads across SMs
    num_batches = LSE_desc.shape[0]
    num_heads = LSE_desc.shape[1]
    total_bh_pairs = num_batches * num_heads
    bh_idx = pid % total_bh_pairs
    restarts = pid // total_bh_pairs
    
    h_idx = bh_idx % num_heads
    b_idx = bh_idx // num_heads
    
    # Start from a different m-block if we've restarted
    start_pid_m = 0
    if restarts > 0:
        start_pid_m = restarts * tl.cdiv(num_pid_m, max(total_bh_pairs // NUM_SMS + 1, 1))

    # Iterate through assigned M-blocks for this (b,h)
    for pid_m in range(start_pid_m, num_pid_m):
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

        # Load Q tile via descriptor [BLOCK_M, D]
        q_tile = Q_desc.load([b_idx, h_idx, pid_m * BLOCK_M, 0])

        # Online softmax accumulators (FP32)
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        num_blocks_n = tl.cdiv(seqlen, BLOCK_N)

        for start_n in range(num_blocks_n):
            offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
            
            # Load K tile via descriptor [BLOCK_N, D]
            k_tile = K_desc.load([b_idx, h_idx, start_n * BLOCK_N, 0])

            # bf16 @ bf16.T -> fp32 accumulator
            s = tl.dot(q_tile, tl.trans(k_tile)) * scale

            # Causal mask: query_pos >= key_pos
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, -float("inf"))

            # Update running max
            m_i_old = m_i
            m_i_new = tl.maximum(m_i, tl.max(s, axis=1))

            # Normalized probabilities
            p = tl.exp(s - m_i_new[:, None])

            alpha = tl.exp(m_i_old - m_i_new)
            l_i = alpha * l_i + tl.sum(p, axis=1)

            # Load V tile via descriptor [BLOCK_N, D]
            v_tile = V_desc.load([b_idx, h_idx, start_n * BLOCK_N, 0]).to(tl.float32)

            # Accumulate weighted contribution
            acc = alpha[:, None] * acc + tl.dot(p, v_tile)

            m_i = m_i_new

        # Normalize output
        acc = acc / l_i[:, None]

        # Store output O [BLOCK_M, D]
        O_desc.store([b_idx, h_idx, pid_m * BLOCK_M, 0], acc.to(tl.bfloat16))

        # Store LSE [BLOCK_M]
        LSE_desc.store([b_idx, h_idx, pid_m * BLOCK_M], m_i + tl.log(l_i))


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward pass with LSE output."""
    torch.cuda.set_device(Q.device)

    bsz, num_heads, seqlen, head_dim = Q.shape
    scale = 1.0 / (head_dim ** 0.5)

    # Create tensor descriptors for TMA loads/stores
    Q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_M := 128, D := head_dim])
    K_desc = TensorDescriptor.from_tensor(K, [BLOCK_N := 64, D])
    V_desc = TensorDescriptor.from_tensor(V, [BLOCK_N, D])
    O_desc = TensorDescriptor.from_tensor(O, [BLOCK_M, D])
    LSE_desc_1d = TensorDescriptor.from_tensor(LSE, [BLOCK_M])

    # We need a 4D descriptor for LSE but it's 3D [B, H, S]
    # So we fall back to pointer-based for LSE and use descriptors for BHM*D tensors
    # Actually let's use stride-based approach with descriptors for spatial dims
    
    props = triton.runtime.driver.active.get_current_device_properties()
    num_sms = props["MULTIPROCESSOR_COUNT"]

    # Grid size capped at number of SMs for persistent scheduling
    grid_size = min(num_sms, bsz * num_heads * triton.cdiv(seqlen, BLOCK_M))
    grid = (grid_size,)

    # Descriptor approach needs 4D layout matching [B, H, S, D]
    # Let me use host-side descriptors with rank=4 for Q,K,V,O
    Q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K, [BLOCK_N, D])
    V_desc = TensorDescriptor.from_tensor(V, [BLOCK_N, D])
    O_desc = TensorDescriptor.from_tensor(O, [BLOCK_M, D])
    
    # For LSE which is [B, H, S], reshape descriptor
    LSE_view = LSE.reshape(-1, seqlen)
    LSE_desc = TensorDescriptor.from_tensor(LSE_view, [BLOCK_M])

    grid_size = min(num_sms, bsz * num_heads)
    grid = (grid_size,)

    # Simpler fallback: use standard pointer kernel but optimized
    _flash_attn_fwd_optimized[grid](
        Q, K, V, O, LSE,
        seqlen,
        scale,
        bsz, num_heads,
        Q.stride()[0], Q.stride()[1], Q.stride()[2], Q.stride()[3],
        K.stride()[0], K.stride()[1], K.stride()[2], K.stride()[3],
        V.stride()[0], V.stride()[1], V.stride()[2], V.stride()[3],
        O.stride()[0], O.stride()[1], O.stride()[2], O.stride()[3],
        LSE.stride()[0], LSE.stride()[1], LSE.stride()[2],
        D=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=4,
    )


@triton.jit
def _flash_attn_fwd_optimized(
    Q,
    K,
    V,
    O,
    LSE,
    seqlen,
    scale,
    num_batch,
    num_heads,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lse_b, stride_lse_h, stride_lse_s,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Optimized causal flash attention with grouped scheduling."""

    pid = tl.program_id(0)

    # Grouped/block-scheduled assignment
    num_pid_m = tl.cdiv(seqlen, BLOCK_M)
    total_tiles = num_batch * num_heads * num_pid_m

    # Persistent-like: cycle through tiles
    pid_in_round = pid % min(total_tiles, triton.runtime.program.num_sm())
    round_num = pid // min(total_tiles, triton.runtime.program.num_sm())
    
    tile_idx = pid_in_round
    for _ in range(round_num + 1):
        if tile_idx >= total_tiles:
            break

        # Decode tile index into (b, h, m)
        remaining = tile_idx
        pid_m = remaining % num_pid_m
        remaining = remaining // num_pid_m
        pid_h = remaining % num_heads
        pid_b = remaining // num_heads
        tile_idx += total_tiles

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)

        base_q = Q + pid_b * stride_qb + pid_h * stride_qh
        base_k = K + pid_b * stride_kb + pid_h * stride_kh
        base_v = V + pid_b * stride_vb + pid_h * stride_vh
        base_o = O + pid_b * stride_ob + pid_h * stride_oh
        base_lse = LSE + pid_b * stride_lse_b + pid_h * stride_lse_h

        # Load Q tile once
        q_ptrs = base_q + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
        q_tile = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)

        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        num_blocks_n = tl.cdiv(seqlen, BLOCK_N)

        for start_n in range(num_blocks_n):
            offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask = (offs_n < seqlen)[:, None]

            k_ptrs = base_k + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_tile = tl.load(k_ptrs, mask=n_mask, other=0.0)

            s = tl.dot(q_tile, tl.trans(k_tile)) * scale
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s = tl.where(causal_mask, s, -float("inf"))

            m_i_old = m_i
            m_i = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp(s - m_i[:, None])
            alpha = tl.exp(m_i_old - m_i)
            l_i = alpha * l_i + tl.sum(p, axis=1)

            v_ptrs = base_v + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
            v_tile = tl.load(v_ptrs, mask=n_mask, other=0.0).to(tl.float32)

            acc = alpha[:, None] * acc + tl.dot(p, v_tile)

        acc = acc / l_i[:, None]
        o_ptrs = base_o + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=offs_m[:, None] < seqlen)

        lse_ptrs = base_lse + offs_m * stride_lse_s
        tl.store(lse_ptrs, m_i + tl.log(l_i), mask=offs_m < seqlen)


def run(Q, K, V, O, LSE):
    """Causal multi-head attention forward pass with LSE output."""
    torch.cuda.set_device(Q.device)

    bsz, num_heads, seqlen, head_dim = Q.shape
    scale = 1.0 / (head_dim ** 0.5)

    sq, sk, sv, so, sl = Q.stride(), K.stride(), V.stride(), O.stride(), LSE.stride()

    BLOCK_M = 128
    BLOCK_N = 64

    props = triton.runtime.driver.active.get_current_device_properties()
    num_sms = props["MULTIPROCESSOR_COUNT"]
    num_tiles_m = triton.cdiv(seqlen, BLOCK_M)
    grid_size = min(num_sms, bsz * num_heads)
    grid = (grid_size,)

    _flash_attn_fwd_optimized[grid](
        Q, K, V, O, LSE,
        seqlen,
        scale,
        bsz, num_heads,
        sq[0], sq[1], sq[2], sq[3],
        sk[0], sk[1], sk[2], sk[3],
        sv[0], sv[1], sv[2], sv[3],
        so[0], so[1], so[2], so[3],
        sl[0], sl[1], sl[2],
        D=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=4,
    )