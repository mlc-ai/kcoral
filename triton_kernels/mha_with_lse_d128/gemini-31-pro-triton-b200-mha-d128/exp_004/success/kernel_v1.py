import torch
import triton
import triton.language as tl

# Set up the descriptor allocator for device-created TMA descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

def get_configs():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "STAGE_COUNT": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "STAGE_COUNT": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "STAGE_COUNT": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "STAGE_COUNT": 2}, num_warps=4, num_stages=2),
    ]

@triton.autotune(configs=get_configs(), key=["S"])
@triton.jit
def _attn_fwd_tma_kernel(
    Q, K, V, O, LSE,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE_COUNT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_m_start = pid_m * BLOCK_M

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh

    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[S, 128],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, 128],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[S, 128],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, 128],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[S, 128],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, 128],
        padding_option="zero"
    )

    # Load Q once
    q = q_desc.load([offs_m_start, 0])

    # Pre-scale Q to fold softmax scaling and natural-to-base-2 conversion
    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale: tl.constexpr = 0.08838834764831843  # 1.0 / sqrt(128)
    q = (q * scale * RCP_LN2).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    n_full_blocks = S // BLOCK_N

    # Pipelined loop for fully valid KV blocks
    for block_idx in tl.range(0, n_full_blocks, num_stages=STAGE_COUNT):
        start_n = block_idx * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        scores = tl.dot(q, k.T)
        
        # Because the keys are valid, max score is finite, m_ij is finite.
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Tail block that requires masking
    if S % BLOCK_N != 0:
        start_n = n_full_blocks * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        scores = tl.dot(q, k.T)
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        scores = tl.where(mask_n[None, :], scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # Epilogue: LSE and O computation
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    LN2: tl.constexpr = 0.6931471805599453
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_log2 * LN2

    offs_m = offs_m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, output.to(tl.bfloat16), mask=offs_m[:, None] < S)

    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=offs_m < S)


@triton.autotune(configs=get_configs(), key=["S"])
@triton.jit
def _attn_fwd_ptr_kernel(
    Q, K, V, O, LSE,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE_COUNT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 128)

    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)

    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale: tl.constexpr = 0.08838834764831843
    q = (q * scale * RCP_LN2).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    n_full_blocks = S // BLOCK_N

    for block_idx in tl.range(0, n_full_blocks, num_stages=STAGE_COUNT):
        start_n = block_idx * BLOCK_N
        k = tl.load(k_ptrs + start_n * stride_ks)
        v = tl.load(v_ptrs + start_n * stride_vs)
        
        scores = tl.dot(q, k.T)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    if S % BLOCK_N != 0:
        start_n = n_full_blocks * BLOCK_N
        curr_n = start_n + offs_n
        mask = curr_n < S
        k = tl.load(k_ptrs + start_n * stride_ks, mask=mask[:, None], other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vs, mask=mask[:, None], other=0.0)
        
        scores = tl.dot(q, k.T)
        scores = tl.where(mask[None, :], scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    LN2: tl.constexpr = 0.6931471805599453
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_log2 * LN2

    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, output.to(tl.bfloat16), mask=offs_m[:, None] < S)

    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """
    Computes non-causal multi-head attention forward returning O and LSE.
    """
    torch.cuda.set_device(Q.device)
    B = Q.size(0)
    H = Q.size(1)
    S = Q.size(2)
    
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    # Check if inputs are contiguous in the last dimension and leading strides are 16-byte aligned.
    if (Q.stride(-1) == 1 and K.stride(-1) == 1 and V.stride(-1) == 1 and
        (Q.stride(-2) * 2) % 16 == 0 and (K.stride(-2) * 2) % 16 == 0 and (V.stride(-2) * 2) % 16 == 0):
        # Dispatch to Blackwell TMA-backed kernel
        _attn_fwd_tma_kernel[grid](
            Q, K, V, O, LSE,
            S,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
        )
    else:
        # Dispatch to pointer fallback kernel
        _attn_fwd_ptr_kernel[grid](
            Q, K, V, O, LSE,
            S,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
        )