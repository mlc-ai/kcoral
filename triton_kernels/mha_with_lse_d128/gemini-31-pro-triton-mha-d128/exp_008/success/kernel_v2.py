import torch
import triton
import triton.language as tl

# Set allocator for device-side tensor descriptors (required for Triton 3.x Hopper TMA)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # 128x128 limits to 3 stages max on SM90 due to 228KB SMEM
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        
        # 128x64 can fit up to 4 stages
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        
        # 64x128
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        
        # 64x64
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=5),
    ],
    key=["S"],
)
@triton.jit
def _mha_fwd_kernel_tma(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    sm_scale,
    B, H, S,
    stride_qz, stride_qh, stride_qs, stride_qd,
    stride_kz, stride_kh, stride_ks, stride_kd,
    stride_vz, stride_vh, stride_vs, stride_vd,
    stride_oz, stride_oh, stride_os, stride_od,
    stride_lz, stride_lh, stride_ls,
    IS_EVEN: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H

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
    
    # Prune CTAs completely outside bounds
    if not IS_EVEN:
        if offset_m >= S:
            return

    # Load query block (Hopper TMA)
    q = q_desc.load([offset_m, 0])

    offs_m = offset_m + tl.arange(0, BLOCK_M)
    
    # Initialize robust math states to avoid NaNs on potential out-of-bounds rows
    if IS_EVEN:
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    else:
        mask_m = offs_m < S
        m_i = tl.where(mask_m, float("-inf"), 0.0)
        l_i = tl.where(mask_m, 0.0, 1.0)
        
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    # WGMMA Pipeline Loop
    for n_idx in range(num_n_blocks):
        offset_n = n_idx * BLOCK_N
        
        # Load key and value blocks
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # Matrix multiply Q @ K.T (Lowers to Hopper WGMMA)
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale
        
        # Apply boundary masking ONLY if sequence length isn't evenly divisible by block size
        if not IS_EVEN:
            offs_n = offset_n + tl.arange(0, BLOCK_N)
            mask = (offs_m[:, None] < S) & (offs_n[None, :] < S)
            qk = tl.where(mask, qk, float("-inf"))
        
        # Online softmax stats updates
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        
        l_i = l_i * alpha + tl.sum(beta, axis=1)
        
        # Scale accumulated values and perform WGMMA for V
        acc = acc * alpha[:, None]
        acc = tl.dot(beta.to(tl.bfloat16), v, acc)
        
        m_i = m_new

    # Final scaling
    acc = acc / l_i[:, None]
    
    # Store outputs via TMA (TMA naturally ignores out-of-bounds elements)
    o_desc.store([offset_m, 0], acc.to(tl.bfloat16))
    
    # Store LSE using standard pointers
    lse_ptrs = LSE_ptr + lse_offset + offs_m * stride_ls
    if IS_EVEN:
        tl.store(lse_ptrs, m_i + tl.log(l_i))
    else:
        tl.store(lse_ptrs, m_i + tl.log(l_i), mask=mask_m)


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
    
    sm_scale = 1.0 / (D ** 0.5)

    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B * H,
        1
    )
    
    # Using 128 as it covers maximum BLOCK_M and BLOCK_N in our autotune config
    is_even = (S % 128 == 0)

    _mha_fwd_kernel_tma[grid](
        Q, K, V, O, LSE,
        sm_scale,
        B, H, S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        IS_EVEN=is_even,
        D=D
    )