import torch
import triton
import triton.language as tl

# Set allocator for device-side tensor descriptors (required for Triton 3.x Hopper TMA)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # Optimized configurations for Hopper architecture (SM90/SM90a).
        # We use num_warps=8 (2 warp groups) to allow ping-pong interleaving 
        # of the QK^T and PV WGMMA instructions natively in hardware.
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _mha_fwd_kernel_tma_pipelined(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    sm_scale_log2,
    B, H, S,
    stride_qz, stride_qh, stride_qs, stride_qd,
    stride_kz, stride_kh, stride_ks, stride_kd,
    stride_vz, stride_vh, stride_vs, stride_vd,
    stride_oz, stride_oh, stride_os, stride_od,
    stride_lz, stride_lh, stride_ls,
    IS_EVEN_M: tl.constexpr,
    IS_EVEN_N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H

    # Offset to the start of this batch/head's [S, D] matrix
    q_offset = batch_idx * stride_qz + head_idx * stride_qh
    k_offset = batch_idx * stride_kz + head_idx * stride_kh
    v_offset = batch_idx * stride_vz + head_idx * stride_vh
    o_offset = batch_idx * stride_oz + head_idx * stride_oh
    lse_offset = batch_idx * stride_lz + head_idx * stride_lh

    # Device-side TMA descriptors 
    q_desc = tl.make_tensor_descriptor(
        Q_ptr + q_offset,
        shape=[S, D],
        strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K_ptr + k_offset,
        shape=[S, D],
        strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr + v_offset,
        shape=[S, D],
        strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D],
        padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O_ptr + o_offset,
        shape=[S, D],
        strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D]
    )

    offset_m = start_m * BLOCK_M
    
    # Prune CTAs completely outside sequence bounds
    if not IS_EVEN_M:
        if offset_m >= S:
            return

    # Load query block (Hopper TMA -> SMEM)
    q = q_desc.load([offset_m, 0])
    
    # Pre-scale Q with log2(e) scale.
    # This automatically moves Q from SMEM to registers, setting up an efficient RS-WGMMA 
    # for the inner loop and heavily reducing ALU operations inside the loop.
    q = (q.to(tl.float32) * sm_scale_log2).to(tl.bfloat16)

    offs_m = offset_m + tl.arange(0, BLOCK_M)
    if not IS_EVEN_M:
        mask_m = offs_m < S
        
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    # -------------------------------------------------------------------------
    # PROLOGUE: First Iteration (Manually unrolled for GEMM-Softmax pipelining)
    # -------------------------------------------------------------------------
    k = k_desc.load([0, 0])
    v = v_desc.load([0, 0])
    
    qk = tl.dot(q, k.T)
    if not IS_EVEN_N:
        offs_n_0 = tl.arange(0, BLOCK_N)
        qk = tl.where(offs_n_0[None, :] < S, qk, float("-inf"))
    if not IS_EVEN_M:
        qk = tl.where(mask_m[:, None], qk, float("-inf"))
        
    m_i_init = tl.max(qk, axis=1)
    
    # Initialize robust math states to avoid NaNs on potential out-of-bounds rows
    if not IS_EVEN_M:
        m_i = tl.where(mask_m, m_i_init, 0.0)
    else:
        m_i = m_i_init
        
    beta = tl.exp2(qk - m_i[:, None])
    
    if not IS_EVEN_M:
        l_i = tl.where(mask_m, tl.sum(beta, axis=1), 1.0)
    else:
        l_i = tl.sum(beta, axis=1)

    # -------------------------------------------------------------------------
    # MAIN PIPELINED LOOP
    # -------------------------------------------------------------------------
    for n_idx in range(1, num_n_blocks):
        offset_n = n_idx * BLOCK_N
        
        # Pre-issue TMA loads for the NEXT iteration.
        # Triton's software pipeliner will further multi-stage these underneath.
        k_next = k_desc.load([offset_n, 0])
        v_next = v_desc.load([offset_n, 0])
        
        # WGMMA RS: Compute Q @ K^T for the NEXT iteration.
        qk_next = tl.dot(q, k_next.T)
        
        if not IS_EVEN_N:
            offs_n_next = offset_n + tl.arange(0, BLOCK_N)
            qk_next = tl.where(offs_n_next[None, :] < S, qk_next, float("-inf"))
        if not IS_EVEN_M:
            qk_next = tl.where(mask_m[:, None], qk_next, float("-inf"))
            
        # -----------------------------------------------------------------
        # Ping-pong: Accumulate current V from the PREVIOUS iteration.
        # Because we haven't rescaled `acc` yet, this cleanly breaks the 
        # dependency chain, allowing WGMMA and MUFU to overlap in hardware!
        # -----------------------------------------------------------------
        acc = tl.dot(beta.to(tl.bfloat16), v, acc)
        
        # Update running softmax stats using the NEXT iteration's dots
        m_ij = tl.max(qk_next, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp2(m_i - m_new)
        beta_next = tl.exp2(qk_next - m_new[:, None])
        
        # Mathematically equivalent to standard scaling, applied AFTER the dot
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(beta_next, axis=1)
        
        # Step variables forward
        m_i = m_new
        beta = beta_next
        k = k_next
        v = v_next

    # -------------------------------------------------------------------------
    # EPILOGUE: Final Accumulation
    # -------------------------------------------------------------------------
    acc = tl.dot(beta.to(tl.bfloat16), v, acc)

    # Final scaling via fast reciprocal multiplication
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    
    # Store outputs via TMA (TMA naturally ignores out-of-bounds elements)
    o_desc.store([offset_m, 0], acc.to(tl.bfloat16))
    
    # Recover LSE in natural base-e: LSE_e = m_i_2 * ln(2) + ln(l_i_2)
    lse = (m_i * 0.6931471805599453) + tl.log(l_i)
    
    # Store LSE using standard pointers
    lse_ptrs = LSE_ptr + lse_offset + offs_m * stride_ls
    if IS_EVEN_M:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes Non-causal FlashAttention forward pass targeting NVIDIA Hopper (SM90/SM90a).
    Q, K, V, O: (B, H, S, D) in bfloat16
    LSE: (B, H, S) in float32
    """
    B, H, S, D = Q.shape

    if S == 0:
        return

    torch.cuda.set_device(Q.device)
    
    # Compute base scale and multiply by log2(e) to exploit hardware base-2 MUFU.EX2 limits
    sm_scale_log2 = (1.0 / (D ** 0.5)) * 1.4426950408889634

    # Group M blocks sequentially by batch_head for maximum L2 cache reuse of K and V
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B * H,
        1
    )
    
    # Capture perfect sequence divisibility for optimal kernel specializations without masks
    is_even_m = (S % 128 == 0) # 128 is a multiple of all BLOCK_M configs
    is_even_n = (S % 128 == 0) # 128 is a multiple of all BLOCK_N configs

    _mha_fwd_kernel_tma_pipelined[grid](
        Q, K, V, O, LSE,
        sm_scale_log2,
        B, H, S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        IS_EVEN_M=is_even_m,
        IS_EVEN_N=is_even_n,
        D=D
    )