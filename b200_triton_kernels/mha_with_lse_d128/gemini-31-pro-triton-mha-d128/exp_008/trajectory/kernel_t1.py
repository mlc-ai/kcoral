import torch
import triton
import triton.language as tl

# Set allocator for device-side tensor descriptors (required for Triton 3.x Hopper TMA)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
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

    # Make Hopper TMA descriptors for asynchronous block loading/storing
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
    
    # Prune CTAs that map completely outside the sequence length
    if offset_m >= S:
        return

    # Load query block
    q = q_desc.load([offset_m, 0])

    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    
    # Initialize variables for online Softmax (prevents NaNs on out-of-bounds rows)
    m_i = tl.where(mask_m, float("-inf"), 0.0)
    l_i = tl.where(mask_m, 0.0, 1.0)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    num_n_blocks = tl.cdiv(S, BLOCK_N)
    
    for n_idx in range(num_n_blocks):
        offset_n = n_idx * BLOCK_N
        
        # Asynchronously load key and value blocks via TMA
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # WGMMA dot product with implicit transpose on shared memory K
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale
        
        # Apply causal/sequence mask (TMA zero-padding doesn't apply to dot accumulators)
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        qk = tl.where(mask, qk, float("-inf"))
        
        # Update running max and scaling factors
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        
        # Update running sum
        l_i = l_i * alpha + tl.sum(beta, axis=1)
        
        # Scale previously accumulated values and perform values WGMMA
        acc = acc * alpha[:, None]
        acc = tl.dot(beta.to(tl.bfloat16), v, acc)
        
        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store outputs
    o_desc.store([offset_m, 0], acc.to(tl.bfloat16))
    
    lse_ptrs = LSE_ptr + lse_offset + offs_m * stride_ls
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

    _mha_fwd_kernel_tma[grid](
        Q, K, V, O, LSE,
        sm_scale,
        B, H, S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        D=D
    )