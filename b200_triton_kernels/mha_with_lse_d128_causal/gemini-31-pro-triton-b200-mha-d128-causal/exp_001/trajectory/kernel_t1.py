import torch
import triton
import triton.language as tl
import math

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

_configs = [
    # 128 x 64
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "WARP_SPECIALIZE": False}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "WARP_SPECIALIZE": True},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "WARP_SPECIALIZE": True},  num_warps=4, num_stages=4),
    
    # 128 x 128
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "WARP_SPECIALIZE": False}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "WARP_SPECIALIZE": True},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "WARP_SPECIALIZE": True},  num_warps=8, num_stages=3),
    
    # 256 x 64
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "WARP_SPECIALIZE": False}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "WARP_SPECIALIZE": True},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "WARP_SPECIALIZE": True},  num_warps=8, num_stages=4),
]

@triton.autotune(configs=_configs, key=["S"])
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE, sm_scale,
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr
):
    start_m = tl.program_id(1) * BLOCK_M
    if start_m >= S:
        return

    off_hz = tl.program_id(0)
    off_b = off_hz // H
    off_h = off_hz % H

    # Device descriptors for TMA memory paths
    q_desc = tl.make_tensor_descriptor(
        Q, shape=[B, H, S, D], strides=[stride_qb, stride_qh, stride_qs, stride_qd],
        block_shape=[1, 1, BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K, shape=[B, H, S, D], strides=[stride_kb, stride_kh, stride_ks, stride_kd],
        block_shape=[1, 1, BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V, shape=[B, H, S, D], strides=[stride_vb, stride_vh, stride_vs, stride_vd],
        block_shape=[1, 1, BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O, shape=[B, H, S, D], strides=[stride_ob, stride_oh, stride_os, stride_od],
        block_shape=[1, 1, BLOCK_M, BLOCK_D]
    )

    q_blk = q_desc.load([off_b, off_h, start_m, 0])
    q = tl.reshape(q_blk, (BLOCK_M, BLOCK_D))
    
    # Initialize running FlashAttention state
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    end_n = tl.minimum(S, start_m + BLOCK_M)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    
    for start_n in tl.range(0, end_n, BLOCK_N, warp_specialize=WARP_SPECIALIZE):
        k_blk = k_desc.load([off_b, off_h, start_n, 0])
        k = tl.reshape(k_blk, (BLOCK_N, BLOCK_D))
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Max update and exponentials
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Rescale existing accumulator
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v_blk = v_desc.load([off_b, off_h, start_n, 0])
        v = tl.reshape(v_blk, (BLOCK_N, BLOCK_D))
        
        # Cast probabilities to hardware bf16 and accumulate values
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Normalize final block result
    out = acc / l_i[:, None]
    out = out.to(tl.bfloat16)
    
    # Store out using TMA descriptor
    out_blk = tl.reshape(out, (1, 1, BLOCK_M, BLOCK_D))
    o_desc.store([off_b, off_h, start_m, 0], out_blk)
    
    # Write natural-log sum-exp slice safely through pointer
    lse = m_i + tl.log(l_i)
    lse_ptrs = LSE + off_b * stride_lseb + off_h * stride_lseh + offs_m * stride_lses
    mask_m = offs_m < S
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    # Required for handling memory for device-created TMA tensor descriptors
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    sm_scale = 1.0 / math.sqrt(D)
    grid = lambda META: (B * H, triton.cdiv(S, META['BLOCK_M']), 1)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE, sm_scale,
        B, H, S, D,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_D=D
    )