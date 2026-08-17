import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Set up descriptor allocator for Hopper TMA
_alloc_fn_set = False


def _ensure_allocator():
    global _alloc_fn_set
    if not _alloc_fn_set:
        triton.set_allocator(lambda size, alignment, stream:
                             torch.empty(size, device="cuda", dtype=torch.int8))
        _alloc_fn_set = True


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    stride_bh,
    stride_seq,
    S,
    D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """FlashAttention kernel with persistent scheduling on Hopper.

    Each program instance (CTA) owns one (batch, head) pair and iterates
    over all query tiles and all key tiles with online softmax accumulation.
    Uses fp32 accumulator throughout; downcasts only at final store.
    """

    pid_bh = tl.program_id(0)
    assert tl.num_programs(0) <= 192  # capped at BH

    # Query tile indexing
    num_query_tiles = tl.cdiv(S, BLOCK_M)
    D_int = int(D)  # compile-time cast for arange

    d_idx = tl.arange(0, D_int)
    bh_offset = pid_bh * stride_bh

    # Online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_M, D_int), dtype=tl.float32)

    # Iterate over each query tile in sequence
    for qtile in tl.range(num_query_tiles, num_stages=1):
        # Reset accumulators for this query tile
        local_acc_o = tl.zeros((BLOCK_M, D_int), dtype=tl.float32)
        local_m = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        local_l = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)

        m_abs = qtile * BLOCK_M
        m_idx = m_abs + tl.arange(0, BLOCK_M)
        m_mask = m_idx < S

        # Load Q tile once per query tile
        q_ptrs = q_ptr + bh_offset + m_idx[:, None] * stride_seq + d_idx[None, :]
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
        q_f32 = q.to(tl.float32)

        # Iterate over key tiles
        for start_n in tl.range(tl.cdiv(S, BLOCK_N), num_stages=3):
            n_idx = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask = n_idx < S

            # Load K and V tiles
            k_ptrs = k_ptr + bh_offset + n_idx[:, None] * stride_seq + d_idx[None, :]
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            k_f32 = k.to(tl.float32)

            v_ptrs = v_ptr + bh_offset + n_idx[:, None] * stride_seq + d_idx[None, :]
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
            v_f32 = v.to(tl.float32)

            # Compute attention scores: Q @ K^T * scale
            s = tl.dot(q_f32, k_f32.T) * scale

            # Online softmax update
            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(local_m, m_ij)

            alpha = tl.exp(local_m - m_new)
            p = tl.exp(s - m_new[:, None])

            local_acc_o = alpha[:, None] * local_acc_o + tl.dot(p, v_f32)

            beta = tl.sum(p, axis=1)
            local_l = alpha * local_l + beta
            local_m = m_new

        # Normalize and store output tile
        l_safe = tl.where(local_l > 0.0, local_l, 1.0)
        out = (local_acc_o / l_safe[:, None]).to(tl.bfloat16)
        o_ptrs = o_ptr + bh_offset + m_idx[:, None] * stride_seq + d_idx[None, :]
        tl.store(o_ptrs, out, mask=m_mask[:, None])

        # Store LSE
        lse_val = local_m + tl.log(l_safe)
        lse_offsets = pid_bh * S + m_idx
        tl.store(lse_ptr + lse_offsets, lse_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """Multi-head attention forward: O = softmax(Q@K^T/sqrt(D))@V with LSE.

    Inputs: Q, K, V : [B, H, S, D] bf16
    Outputs: O : [B, H, S, D] bf16, LSE : [B, H, S] f32
    Destination-passing: writes into preallocated O and LSE tensors.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape

    BH = B * H

    # Contiguous reshapes to [BH, S, D] with simple strides
    Q_c = Q.reshape(BH, S, D).contiguous()
    K_c = K.reshape(BH, S, D).contiguous()
    V_c = V.reshape(BH, S, D).contiguous()
    O_c = O.reshape(BH, S, D).contiguous()

    stride_bh = S * D  # bytes to skip between (b,h) pairs
    stride_seq = D     # bytes to skip between sequence positions

    scale = 1.0 / (float(D) ** 0.5)

    # Launch one CTA per (batch, head) pair — persistent scheduling
    grid = (BH,)

    _attention_kernel[grid](
        Q_c, K_c, V_c, O_c, LSE,
        stride_bh, stride_seq,
        S, D, scale,
        BLOCK_M=64, BLOCK_N=64,
    )