import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale,
    S, H,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_lz, stride_lh, stride_ls,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    # Program block coordinates
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H

    # Offset calculations for batch and head
    q_offset = batch_idx * stride_qz + head_idx * stride_qh
    k_offset = batch_idx * stride_kz + head_idx * stride_kh
    v_offset = batch_idx * stride_vz + head_idx * stride_vh
    o_offset = batch_idx * stride_oz + head_idx * stride_oh
    lse_offset = batch_idx * stride_lz + head_idx * stride_lh

    # Coordinate vectors
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    # Base pointers
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    mask_m = offs_m < S

    # Load Q
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize running stats to avoid NaNs on out-of-bounds queries
    # Using 0.0 and 1.0 for invalid rows guarantees mathematically clean fallback (without NaNs).
    m_i = tl.where(mask_m, float("-inf"), 0.0)
    l_i = tl.where(mask_m, 0.0, 1.0)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < S

        # Load K, V
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Compute dot
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale

        # Mask out-of-bounds queries and keys
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        # Update stats
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        
        l_i = l_i * alpha + tl.sum(beta, axis=1)
        
        # Accumulate
        acc = acc * alpha[:, None]
        acc = tl.dot(beta.to(tl.bfloat16), v, acc)
        
        m_i = m_new

        # Advance pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # Epilogue
    acc = acc / l_i[:, None]
    
    # Store outputs
    o_ptrs = O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None])

    lse_ptrs = LSE + lse_offset + offs_m * stride_ls
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes Non-causal FlashAttention forward pass.
    Q, K, V, O: (B, H, S, D) in bfloat16
    LSE: (B, H, S) in float32
    """
    B, H, S, D = Q.shape

    if S == 0:
        return

    # Ensure device is set
    torch.cuda.set_device(Q.device)
    
    # 1.0 / sqrt(D)
    sm_scale = 1.0 / (D ** 0.5)

    # Grid mapping
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B * H,
        1
    )

    # Launch Kernel
    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale,
        S, H,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        D=D
    )