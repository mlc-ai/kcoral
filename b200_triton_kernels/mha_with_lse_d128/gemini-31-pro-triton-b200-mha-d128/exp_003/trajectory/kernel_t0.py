import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qz, stride_qh, stride_qs, stride_qd,
    stride_kz, stride_kh, stride_ks, stride_kd,
    stride_vz, stride_vh, stride_vs, stride_vd,
    stride_oz, stride_oh, stride_os, stride_od,
    stride_lsez, stride_lseh, stride_lses,
    B, H, S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr
):
    start_m = tl.program_id(1)
    off_hz = tl.program_id(0)

    # Determine batch and head indices
    batch_idx = off_hz // H
    head_idx = off_hz % H

    # Base memory offsets for the current batch and head
    q_offset = batch_idx * stride_qz + head_idx * stride_qh
    k_offset = batch_idx * stride_kz + head_idx * stride_kh
    v_offset = batch_idx * stride_vz + head_idx * stride_vh
    o_offset = batch_idx * stride_oz + head_idx * stride_oh
    lse_offset = batch_idx * stride_lsez + head_idx * stride_lseh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)

    # Initialize pointers
    # Q is shaped [BLOCK_M, D_HEAD]
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    # K is loaded conceptually transposed: [D_HEAD, BLOCK_N]
    k_ptrs = K + k_offset + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks
    # V is shaped [BLOCK_N, D_HEAD]
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd

    # Initialize attention accumulator and LSE tracking variables
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # Load queries
    mask_q_1d = offs_m < S
    mask_q_2d = mask_q_1d[:, None]
    q = tl.load(q_ptrs, mask=mask_q_2d, other=0.0)

    # Iterate over key/value sequence
    for start_n in range(0, S, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # Load keys
        mask_k = (start_n + offs_n)[None, :] < S
        k = tl.load(k_ptrs, mask=mask_k, other=0.0)
        
        # Load values
        mask_v = (start_n + offs_n)[:, None] < S
        v = tl.load(v_ptrs, mask=mask_v, other=0.0)

        # Compute dot product (Q @ K^T)
        qk = tl.dot(q, k)
        qk = qk * sm_scale

        # Apply causal/padding masks
        mask_k_2d = (start_n + offs_n)[None, :] < S
        qk = tl.where(mask_k_2d & mask_q_2d, qk, float("-inf"))

        # Numerically stable softmax and attention weighting
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])

        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)

        m_i = m_ij
        
        # Advance pointers
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Finalize probabilities and LSE
    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    # Store output O
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=mask_q_2d)
    
    # Store output LSE
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=mask_q_1d)


def run(Q, K, V, O, LSE):
    """
    Computes Non-causal Multi-Head Attention.
    Inputs and outputs are provided directly via destination-passing.
    """
    torch.cuda.set_device(Q.device)
    
    # Q, K, V shapes: (B, H, S, D)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)

    grid = lambda META: (
        B * H,
        triton.cdiv(S, META["BLOCK_M"]),
    )
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        D_HEAD=D
    )