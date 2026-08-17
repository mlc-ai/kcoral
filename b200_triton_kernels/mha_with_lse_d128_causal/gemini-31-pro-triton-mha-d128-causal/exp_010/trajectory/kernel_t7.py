import torch
import triton
import triton.language as tl

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8, num_ctas=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8, num_ctas=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=8, num_ctas=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=8, num_ctas=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=4, num_warps=8, num_ctas=1),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_stages=2, num_warps=8, num_ctas=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4, num_ctas=1),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=5, num_warps=4, num_ctas=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4, num_ctas=1),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['S']
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale_log2,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    # Strict boundary guard preventing redundant execution
    if start_m >= S:
        return

    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    # Base pointers mapped for currently scheduled batch and head
    q_ptr = Q + off_b * stride_qb + off_h * stride_qh
    k_ptr = K + off_b * stride_kb + off_h * stride_kh
    v_ptr = V + off_b * stride_vb + off_h * stride_vh
    o_ptr = O + off_b * stride_ob + off_h * stride_oh

    # Hardware Tensor Memory Accelerator (TMA) fast layout descriptors 
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[S, BLOCK_D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[S, BLOCK_D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[S, BLOCK_D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[S, BLOCK_D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D]
    )

    # Issue single bulk TMA read for the fixed Query block
    q = q_desc.load([start_m, 0])

    # Online softmax execution contexts in FP32 fidelity
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Strict dynamic causal bounds formulation
    end_m = start_m + BLOCK_M
    seq_limit = tl.minimum(S, end_m)
    num_steps = (seq_limit + BLOCK_N - 1) // BLOCK_N
    
    # Identify loop boundaries cleanly separated from causal padding for maximal software pipelining
    num_steps_unmasked = start_m // BLOCK_N
    num_steps_unmasked = tl.minimum(num_steps_unmasked, num_steps)

    # Pipeline Phase 1: Fully Unmasked Loop leveraging pure WGMMA throughput natively
    for start_n_idx in range(0, num_steps_unmasked):
        start_n = start_n_idx * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        # Fuse scaling mathematically avoiding BF16 precision downgrade on Q
        qk = qk * sm_scale_log2
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Pipeline Phase 2: Tail Loop projecting dynamic causality and sequence boundaries safely
    for start_n_idx in range(num_steps_unmasked, num_steps):
        start_n = start_n_idx * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        # Enforce local causal triangulation rigorously
        causal_mask = (start_n + offs_n)[None, :] <= offs_m[:, None]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Shield trailing dimensions from generating false scores overriding descriptor padding
        if start_n + BLOCK_N > S:
            valid_mask = (start_n + offs_n)[None, :] < S
            qk = tl.where(valid_mask, qk, float("-inf"))
            
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Epilogue: Project execution back into Natural Log Scale strictly satisfying LSE format requirements
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    
    LN_2 = 0.6931471805599453
    lse = (m_i * LN_2) + tl.log(l_i)

    # Push bounded final vectors natively utilizing TMA bypassing sequence bounds handling logic
    o_desc.store([start_m, 0], acc.to(tl.bfloat16))

    # Safely anchor localized sequence stores protecting vector limits
    lse_offset = off_b * stride_lseb + off_h * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    q_valid = offs_m < S
    tl.store(lse_ptrs, lse, mask=q_valid)


def run(Q, K, V, O, LSE):
    # Anchor thread mapping context natively required for descriptors
    torch.cuda.set_device(Q.device)
    
    # Establish Triton global device descriptor storage
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    # Shift pre-computation natively onto CPU retaining kernel compactness
    LOG2_E = 1.4426950408889634
    sm_scale_log2 = sm_scale * LOG2_E

    # The grid topology isolates (M, B, H) enforcing concurrent CTAs sharing K,V blocks organically
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B,
        H
    )

    _fwd_kernel[grid](
        Q, K, V, sm_scale_log2,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        BLOCK_D=D,
    )