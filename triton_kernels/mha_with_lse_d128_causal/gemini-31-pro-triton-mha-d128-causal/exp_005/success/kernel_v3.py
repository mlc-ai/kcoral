import torch
import triton
import triton.language as tl

# Global pool to keep Triton TMA descriptors alive during asynchronous kernel execution, 
# preventing PyTorch caching allocator from overwriting them mid-flight.
_descriptor_pool = []

def alloc_fn(size: int, alignment: int, stream):
    tensor = torch.empty(size, device="cuda", dtype=torch.int8)
    _descriptor_pool.append(tensor)
    return tensor

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=5),
    ],
    key=['S_len']
)
@triton.jit
def _causal_mha_tma_kernel(
    q, k, v, o, lse,
    S_len, H,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch_idx = pid_bh // H
    head_idx = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    
    # Fast exit for cleanly out-of-bound blocks
    if start_m >= S_len:
        return
        
    # Offset bases mapped securely
    q_base = q + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v + batch_idx * stride_vb + head_idx * stride_vh
    o_base = o + batch_idx * stride_ob + head_idx * stride_oh
    
    # Hopper TMA descriptors creation
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S_len, BLOCK_D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    # WGMMA expects Column-Major for transpose math layout safely handling K's logical view seamlessly
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S_len, BLOCK_D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S_len, BLOCK_D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    # TMA implicitly discards out of bounds elements when storing; padding argument isn't applicable
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S_len, BLOCK_D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D]
    )
    
    # Start issuing Query loads (Asynchronous background move)
    q_val = q_desc.load([start_m, 0])
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    m_mask = offs_m < S_len
    
    # Stable online softmax tracked attributes (We explicitly initialize masked out values with `0.0`
    # for max offsets preventing `-inf - (-inf) => NaN` logic traps across boundaries)
    m_i = tl.where(m_mask, float("-inf"), 0.0).to(tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    unmasked_end = (start_m // BLOCK_N) * BLOCK_N
    
    # -------------------------------------------------------------
    # PHASE 1: Fully Unmasked Loop Sequence
    # (Processes all strictly upper boundary blocks relative to Q segment)
    # -------------------------------------------------------------
    for start_n in range(0, unmasked_end, BLOCK_N):
        k_val = k_desc.load([start_n, 0])
        v_val = v_desc.load([start_n, 0])
        
        # Scaling done dynamically in FP32 format post-dot avoiding early truncation errors
        qk = tl.dot(q_val, k_val.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        p_bf16 = p.to(v_val.dtype)
        acc = tl.dot(p_bf16, v_val, acc)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    # -------------------------------------------------------------
    # PHASE 2: Causal Masked and Padded Boundary Loop
    # -------------------------------------------------------------
    masked_end = start_m + BLOCK_M
    if S_len < masked_end:
        masked_end = S_len
    masked_end_aligned = ((masked_end + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    
    for start_n in range(unmasked_end, masked_end_aligned, BLOCK_N):
        k_val = k_desc.load([start_n, 0])
        v_val = v_desc.load([start_n, 0])
        
        qk = tl.dot(q_val, k_val.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        n_mask = offs_n[None, :] < S_len
        
        mask = causal_mask & n_mask
        qk = tl.where(mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        p_bf16 = p.to(v_val.dtype)
        acc = tl.dot(p_bf16, v_val, acc)
        
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
    # Scale finalized normalization
    l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc * (1.0 / l_i_safe[:, None])
    
    # Native Hopper TMA Store logic implicitly accounts for cropping limits safely avoiding padding overwrites
    o_desc.store([start_m, 0], acc.to(o.dtype.element_ty))
    
    # Store standard pointers based Log-Sum-Exp outputs
    lse_base = lse + batch_idx * stride_lseb + head_idx * stride_lseh
    lse_ptrs = lse_base + offs_m * stride_lses
    lse_val = m_i + tl.log(l_i_safe)
    
    tl.store(lse_ptrs, lse_val, mask=m_mask)


def run(Q, K, V, O, LSE):
    """
    Computes causal Multi-Head Attention forward pass and returns Output and Log-Sum-Exp.
    Writes outputs directly into preallocated `O` and `LSE` tensors using destination-passing convention.
    """
    # Free existing descriptor structure memory bounds strictly before dispatch to prevent memory leakage 
    # across multiple global module `run` transactions.
    _descriptor_pool.clear()
    
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    if S > 0:
        grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
        
        _causal_mha_tma_kernel[grid](
            Q, K, V, O, LSE,
            S, H,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            sm_scale,
            BLOCK_D=D
        )