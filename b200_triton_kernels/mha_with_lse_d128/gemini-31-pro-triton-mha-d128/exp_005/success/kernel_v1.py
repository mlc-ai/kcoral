import torch
import triton
import triton.language as tl

# Configure Triton's descriptor allocator to allow device-side tl.make_tensor_descriptor
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel_tma(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    S,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    b = off_hz // H
    h = off_hz % H

    # Offset to the start of the current head
    q_base = Q + b * stride_qb + h * stride_qh
    k_base = K + b * stride_kb + h * stride_kh
    v_base = V + b * stride_vb + h * stride_vh
    o_base = O + b * stride_ob + h * stride_oh

    # Create TMA descriptors for 2D slices [S, D]
    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[S, D],
        strides=[stride_qs, 1],
        block_shape=[BLOCK_M, D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[S, D],
        strides=[stride_ks, 1],
        block_shape=[BLOCK_N, D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[S, D],
        strides=[stride_vs, 1],
        block_shape=[BLOCK_N, D],
        padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base,
        shape=[S, D],
        strides=[stride_os, 1],
        block_shape=[BLOCK_M, D]
    )

    offset_m = start_m * BLOCK_M
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S

    # TMA load for Q
    q = q_desc.load([offset_m, 0])

    # Initialize statistics
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    # Mask out padded rows to prevent them from becoming NaN in softmax
    m_i = tl.where(mask_m, m_i, 0.0)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    sm_scale = 1.0 / (float(D) ** 0.5)
    num_steps = tl.cdiv(S, BLOCK_N)

    for step in range(0, num_steps):
        offset_n = step * BLOCK_N
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        
        # TMA load for K
        k = k_desc.load([offset_n, 0])
        
        # Compute Q @ K.T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, qk)
        qk = qk * sm_scale

        qk_mask = mask_m[:, None] & mask_n[None, :]
        qk = tl.where(qk_mask, qk, float("-inf"))

        # Update running max
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Keep padded M rows at 0.0 to prevent evaluating exp(-inf - -inf) = NaN
        m_i_new = tl.where(mask_m, m_i_new, 0.0)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # Update running sum and output accumulator
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        # TMA load for V
        v = v_desc.load([offset_n, 0])
        
        p_bf16 = p.to(tl.bfloat16)
        acc = acc * alpha[:, None]
        acc = tl.dot(p_bf16, v, acc)
        
        m_i = m_i_new
        l_i = l_i_new

    # Finalize outputs
    l_i_safe = tl.where(mask_m, l_i, 1.0)
    acc = acc / l_i_safe[:, None]
    out = acc.to(tl.bfloat16)
    
    # TMA store for O
    o_desc.store([offset_m, 0], out)
    
    # Store LSE
    lse = m_i + tl.log(l_i_safe)
    lse_offset = b * stride_lseb + h * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes Scaled Dot-Product Attention utilizing Hopper TMA and WGMMA.
    Writes outputs to preallocated O and LSE tensors.
    """
    torch.cuda.set_device(Q.device)
    
    B_sz, H_sz, S_sz, D_sz = Q.shape
    
    grid = lambda META: (
        triton.cdiv(S_sz, META["BLOCK_M"]),
        B_sz * H_sz
    )
    
    _fwd_kernel_tma[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S=S_sz, H=H_sz, D=D_sz,
    )