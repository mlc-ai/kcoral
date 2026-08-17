import torch
import triton
import triton.language as tl
import math

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

# Configurations tailored for safe and high-occupancy 5th-Gen Tensor Core execution
_configs = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=_configs, key=["S"])
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE, sm_scale,
    S, H,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(1) * BLOCK_M
    if start_m >= S:
        return

    off_hz = tl.program_id(0)
    off_b = off_hz // H
    off_h = off_hz % H

    # Setup base pointers
    q_ptr = Q + off_b * stride_qb + off_h * stride_qh
    k_ptr = K + off_b * stride_kb + off_h * stride_kh
    v_ptr = V + off_b * stride_vb + off_h * stride_vh
    o_ptr = O + off_b * stride_ob + off_h * stride_oh

    # 2D device descriptors representing [S, D] slices (TMA constraints met by standard contiguous D)
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

    # Initial setup for online softmax
    q = q_desc.load([start_m, 0])
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    end_n = tl.minimum(S, start_m + BLOCK_M)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    
    # Range is converted into integer steps for robustness
    num_steps = tl.cdiv(end_n, BLOCK_N)
    for step in range(num_steps):
        start_n = step * BLOCK_N
        
        # Load logically non-transposed but feed .T directly to tl.dot
        k = k_desc.load([start_n, 0])
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Apply lower-triangular causal attention mask explicitly
        offs_n = start_n + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = causal_mask & (offs_n[None, :] < S)
        qk = tl.where(valid_mask, qk, float("-inf"))
        
        # FlashAttention rescaling sequence
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # Values integration
        v = v_desc.load([start_n, 0])
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Safe normalization scaling
    out = acc / l_i[:, None]
    out = out.to(tl.bfloat16)
    o_desc.store([start_m, 0], out)
    
    # Store LSE sequentially via direct pointers out-of-loop
    lse = m_i + tl.log(l_i)
    lse_ptrs = LSE + off_b * stride_lseb + off_h * stride_lseh + offs_m * stride_lses
    mask_m = offs_m < S
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    # Necessary runtime allocator intercept for descriptor creation infrastructure
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    sm_scale = 1.0 / math.sqrt(D)
    grid = lambda META: (B * H, triton.cdiv(S, META['BLOCK_M']), 1)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE, sm_scale,
        S, H,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_D=D
    )