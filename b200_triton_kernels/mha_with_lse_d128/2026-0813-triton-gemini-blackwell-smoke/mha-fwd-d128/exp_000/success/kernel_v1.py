import torch
import triton
import triton.language as tl


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offset_m = pid_m * BLOCK_M

    # Advance base pointers to the current batch and head
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    
    # Device-created descriptors for TMA loads
    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[S, BLOCK_D],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[S, BLOCK_D],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[S, BLOCK_D],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )

    q = q_desc.load([offset_m, 0])

    # Pre-scale Q in base-2 units to save operations in the KV loop
    RCP_LN2: tl.constexpr = 1.4426950408889634
    q = (q * softmax_scale * RCP_LN2).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    kv_blocks = tl.cdiv(S, BLOCK_N)
    full_blocks = S // BLOCK_N

    # Fast path: Loop over guaranteed full KV blocks
    for kv_idx in range(0, full_blocks):
        offset_n = kv_idx * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        scores = tl.dot(q, k.T)
        
        # Omit m_ij == -inf repair for proven active queries tracking valid full-block keys
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Slow path: Last KV block (partial sequence tracking)
    if full_blocks < kv_blocks:
        kv_idx = full_blocks
        offset_n = kv_idx * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        scores = tl.dot(q, k.T)
        
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        valid_score = offs_n < S
        scores = tl.where(valid_score[None, :], scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Epilogue
    output = acc / l_i[:, None]
    lse_log2 = m_i + tl.math.log2(l_i)
    
    LN2: tl.constexpr = 0.6931471805599453
    lse_ln = lse_log2 * LN2

    # TMA Store O (Hardware implicitly ignores out-of-bounds queries past shape)
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    o_desc = tl.make_tensor_descriptor(
        o_base,
        shape=[S, BLOCK_D],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )
    o_desc.store([offset_m, 0], output.to(tl.bfloat16))

    # Pointer Store LSE
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    lse_base = LSE + pid_b * stride_lseb + pid_h * stride_lseh
    lse_ptrs = lse_base + offs_m * stride_lses
    tl.store(lse_ptrs, lse_ln, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    softmax_scale = 1.0 / (D ** 0.5)

    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        softmax_scale,
        BLOCK_D=D,
    )