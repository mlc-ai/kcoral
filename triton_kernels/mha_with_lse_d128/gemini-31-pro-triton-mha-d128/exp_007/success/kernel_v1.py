import torch
import triton
import triton.language as tl

# Standard Triton descriptor allocator required for tl.make_tensor_descriptor on Hopper
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    offset_q = batch_id * stride_qb + head_id * stride_qh
    offset_k = batch_id * stride_kb + head_id * stride_kh
    offset_v = batch_id * stride_vb + head_id * stride_vh
    offset_o = batch_id * stride_ob + head_id * stride_oh

    # Device-created 2D tensor descriptors leveraging Hopper TMA
    q_desc = tl.make_tensor_descriptor(
        q_ptr + offset_q,
        shape=[S, BLOCK_D],
        strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr + offset_k,
        shape=[S, BLOCK_D],
        strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr + offset_v,
        shape=[S, BLOCK_D],
        strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr + offset_o,
        shape=[S, BLOCK_D],
        strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D]
    )

    offset_lse = batch_id * stride_lseb + head_id * stride_lseh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = lse_ptr + offset_lse + offs_m * stride_lses

    # Accumulators in FP32
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Initial async load of Q
    q = q_desc.load([start_m * BLOCK_M, 0])

    num_steps = tl.cdiv(S, BLOCK_N)
    for i in range(num_steps):
        # Async TMA loads of K and V
        k = k_desc.load([i * BLOCK_N, 0])
        v = v_desc.load([i * BLOCK_N, 0])

        # Hardware lowers this directly to WGMMA instructions
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale

        # Mask invalid K elements if S is not perfectly divisible by BLOCK_N
        offs_n_curr = i * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_mask = offs_n_curr[None, :] < S
        qk = tl.where(valid_mask, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(qk - m_i_new[:, None])
        beta = tl.where(valid_mask, beta, 0.0)

        l_i_new = alpha * l_i + tl.sum(beta, 1)

        acc = acc * alpha[:, None]
        # Hardware lowers this down to WGMMA with FP32 accumulator
        acc = tl.dot(beta.to(tl.bfloat16), v, acc, out_dtype=tl.float32)

        m_i = m_i_new
        l_i = l_i_new

    # Output normalization
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    # TMA store intrinsically handles boundary checking based on tensor shape descriptor
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Store standard logsumexp array
    out_mask = offs_m < S
    tl.store(lse_ptrs, lse, mask=out_mask)


def run(Q, K, V, O, LSE):
    """
    Compute non-causal multi-head attention.
    Q, K, V: [B, H, S, D] (bfloat16)
    O: [B, H, S, D] (bfloat16)
    LSE: [B, H, S] (float32)
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    if S == 0:
        return

    # Scaling factor 1 / sqrt(D)
    sm_scale = 1.0 / (D ** 0.5)

    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), H, B)

    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        sm_scale,
        BLOCK_D=128,
    )