import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_desc,
    k_desc,
    v_desc,
    o_desc,
    lse_ptr,
    S,
    D,
    scale,
    BH_STRIDE_LSE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Hopper-optimized FlashAttention using TMA descriptors.
    
    One program per (batch*head, query_tile). Iterates over key tiles
    with online softmax normalization. Uses TMA for Q/K/V/O loads/stores.
    """
    pid_bh = tl.program_id(0)
    pid_qtile = tl.program_id(1)

    # Compute flat index for the (batch, head) pair
    q_row_base = pid_qtile * BLOCK_M

    # Load Q tile via TMA descriptor [BLOCK_M, D]
    q_off_m = q_row_base
    q_off_d = 0
    q = q_desc.load([pid_bh, q_off_m, 0])
    
    # Convert Q to fp32 for computation
    q_f32 = q.to(tl.float32)

    # Online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Build masks after loading
    m_idx = q_row_base + tl.arange(0, BLOCK_M)
    m_mask = m_idx < S

    # Iterate over key tiles
    num_ktiles = tl.cdiv(S, BLOCK_N)
    for ktile in range(num_ktiles):
        n_off = ktile * BLOCK_N

        # Load K tile [BLOCK_N, D] via TMA
        k = k_desc.load([pid_bh, n_off, 0])
        k_f32 = k.to(tl.float32)

        # Load V tile [BLOCK_N, D] via TMA  
        v = v_desc.load([pid_bh, n_off, 0])
        v_f32 = v.to(tl.float32)

        # N mask
        n_idx = n_off + tl.arange(0, BLOCK_N)
        n_mask = n_idx < S

        # Attention scores: Q @ K^T -> [BLOCK_M, BLOCK_N]
        s = tl.dot(q_f32, k_f32.T) * scale

        # Apply column mask: zero out invalid columns
        s = tl.where(n_mask[None, :], s, -float("inf"))

        # Online softmax update
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])

        acc_o = alpha[:, None] * acc_o + tl.dot(p, v_f32)

        beta = tl.sum(p, axis=1)
        l_i = alpha * l_i + beta
        m_i = m_new

    # Normalize and store output O tile via TMA
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = (acc_o / l_safe[:, None]).to(tl.bfloat16)

    if m_mask[0]:  # Only store if valid rows exist
        o_desc.store([pid_bh, q_row_base, 0], out)

    # Write LSE via regular store
    lse_val = m_i + tl.log(l_safe)
    lse_offsets = pid_bh * BH_STRIDE_LSE + m_idx
    tl.store(lse_ptr + lse_offsets, lse_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """Multi-head attention forward with LSE output.
    
    Inputs: Q, K, V : [B, H, S, D] bf16
    Outputs: O : [B, H, S, D] bf16, LSE : [B, H, S] f32
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    BH = B * H

    # Contiguous reshapes: [BH, S, D] for simple indexing
    Q_c = Q.reshape(BH, S, D).contiguous()
    K_c = K.reshape(BH, S, D).contiguous()
    V_c = V.reshape(BH, S, D).contiguous()
    O_c = O.reshape(BH, S, D).contiguous()

    # Create 3D tensor descriptors: [BH, S, D] with block [1, BLOCK_M, D]
    BLOCK_M = 64
    BLOCK_N = 64

    q_desc = TensorDescriptor.from_tensor(Q_c, block_shape=[1, BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K_c, block_shape=[1, BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V_c, block_shape=[1, BLOCK_N, D])
    o_desc = TensorDescriptor.from_tensor(O_c, block_shape=[1, BLOCK_M, D])

    scale = 1.0 / (float(D) ** 0.5)
    num_qtiles = triton.cdiv(S, BLOCK_M)

    # Grid: (batch*head, num_query_tiles)
    grid = (BH, num_qtiles)

    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc,
        LSE,
        S, D, scale,
        S,  # LSE stride along sequence
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=3,
    )