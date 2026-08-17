import torch
import triton
import triton.language as tl
import math

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

# Heavily optimized configs hitting high TMEM occupancy boundaries and perfect L2 sharing
_configs = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPE_STAGES": 3}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPE_STAGES": 4}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "PIPE_STAGES": 3}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "PIPE_STAGES": 4}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "PIPE_STAGES": 3}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "PIPE_STAGES": 3}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=_configs, key=["S"])
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE, sm_scale_log2,
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    PIPE_STAGES: tl.constexpr
):
    # Mapping X-axis to sequence offset blocks execution along a consistent Y-axis (batch*head)
    # maximizes SM overlapping where varying M bounds seamlessly re-use the exact same K and V 
    # matrices freshly stored in the ultra-fast L2 Cache.
    start_m = tl.program_id(0) * BLOCK_M
    if start_m >= S:
        return

    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H

    # 4D TMA block descriptors cleanly abstracting multi-dimensional strided out-of-bound protections natively
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
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Restrict execution explicitly to diagonal bounds ensuring causality
    end_n = tl.minimum(S, start_m + BLOCK_M)
    
    # SINGLE UNIFIED LOOP ensuring the `acc` tensors gracefully anchor in Tensor Memory mapping strictly to tcgen05
    for start_n in tl.range(0, end_n, BLOCK_N, num_stages=PIPE_STAGES):
        k_blk = k_desc.load([off_b, off_h, start_n, 0])
        k = tl.reshape(k_blk, (BLOCK_N, BLOCK_D))
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        # Simple logical causality checks omitting superfluous sequence bound checks because
        # mathematical bounds completely handle validity implicitly within causality.
        offs_n = start_n + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        
        # Single cycle EX2 (base-2 Exponentials)
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v_blk = v_desc.load([off_b, off_h, start_n, 0])
        v = tl.reshape(v_blk, (BLOCK_N, BLOCK_D))
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Re-normalize variables logically back onto outputs
    out = acc / l_i[:, None]
    out = out.to(tl.bfloat16)
    
    out_blk = tl.reshape(out, (1, 1, BLOCK_M, BLOCK_D))
    o_desc.store([off_b, off_h, start_m, 0], out_blk)
    
    # Scale mathematical representation back onto Ln base-e safely factoring in multiplier components
    lse = m_i + tl.log2(l_i)
    lse = lse * 0.6931471805599453
    lse_ptrs = LSE + off_b * stride_lseb + off_h * stride_lseh + offs_m * stride_lses
    mask_m = offs_m < S
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    # Scale applied beforehand factoring log2(e) for algorithmic mathematical evaluations avoiding scale downs
    sm_scale_log2 = (1.0 / math.sqrt(D)) * 1.4426950408889634
    
    # X Grid tracks local variables guaranteeing optimal L2 memory caching distributions
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H, 1)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE, sm_scale_log2,
        B, H, S, D,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_D=D
    )