import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=5),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, sm_scale_log2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    # Offsets to the start of the block
    q_offset = batch_id * stride_qb + head_id * stride_qh
    k_offset = batch_id * stride_kb + head_id * stride_kh
    v_offset = batch_id * stride_vb + head_id * stride_vh
    o_offset = batch_id * stride_ob + head_id * stride_oh
    lse_offset = batch_id * stride_lseb + head_id * stride_lseh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Initialize standard pointer block arrays
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + k_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses

    # Accumulators in FP32
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    q_mask = (offs_m[:, None] < S) & (offs_d[None, :] < BLOCK_D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Decouple the sequence length into fully complete blocks and remainder
    num_steps = S // BLOCK_N
    rem = S % BLOCK_N

    # Highly optimized inner loop operating without conditional bounds checking.
    # WGMMA execution natively pipelined by standard standard Triton.
    for i in range(num_steps):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        # Matmul down to Tensor Cores, out format FP32
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk = qk * sm_scale_log2

        # Fast max reductions
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)

        # Exponentiate directly using hardware base-2 instruction `tl.exp2`
        alpha = tl.exp2(m_i - m_i_new)
        beta = tl.exp2(qk - m_i_new[:, None])

        l_i_new = alpha * l_i + tl.sum(beta, 1)

        # Scale accumulator and perform V multiplication
        acc = acc * alpha[:, None]
        acc = tl.dot(beta.to(tl.bfloat16), v, acc, out_dtype=tl.float32)

        m_i = m_i_new
        l_i = l_i_new

        # Advance along the sequence dimension
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Handling the remaining uneven block if there is one
    if rem > 0:
        k_mask = (offs_n[:, None] < rem) & (offs_d[None, :] < BLOCK_D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        qk = qk * sm_scale_log2

        # Effectively mask out boundaries by introducing negative infinity before max
        qk = tl.where(offs_n[None, :] < rem, qk, float("-inf"))

        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp2(m_i - m_i_new)
        beta = tl.exp2(qk - m_i_new[:, None])

        l_i_new = alpha * l_i + tl.sum(beta, 1)

        acc = acc * alpha[:, None]
        acc = tl.dot(beta.to(tl.bfloat16), v, acc, out_dtype=tl.float32)

        m_i = m_i_new
        l_i = l_i_new

    # Outputs
    acc = acc / l_i[:, None]
    # Reconvert max from log2(e) back down to actual natural log values: `val * ln(2)`
    lse = m_i * 0.6931471805599453 + tl.log(l_i)

    out_mask_2d = (offs_m[:, None] < S) & (offs_d[None, :] < BLOCK_D)
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=out_mask_2d)
    
    out_mask_1d = offs_m < S
    tl.store(lse_ptrs, lse, mask=out_mask_1d)


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

    # Base scale configuration injected with constant log2(e) conversion setup natively
    sm_scale = 1.0 / (D ** 0.5)
    LOG2_E = 1.4426950408889634
    sm_scale_log2 = sm_scale * LOG2_E

    # Grid ordered `(S, H, B)` purposefully loops adjacent program IDs together naturally 
    # establishing outstanding L2 hit rates on the duplicated `K` and `V` matrix accesses between heads.
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), H, B)

    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, sm_scale_log2,
        BLOCK_D=128,
    )