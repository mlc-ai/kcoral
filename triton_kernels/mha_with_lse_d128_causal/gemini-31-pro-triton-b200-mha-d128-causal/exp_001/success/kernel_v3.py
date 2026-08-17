import torch
import triton
import triton.language as tl
import math

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

# Finely tuned constraints maximizing L2 re-use and optimal Shared Memory allocations on SM100
_configs = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPE_STAGES": 3}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPE_STAGES": 4}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "PIPE_STAGES": 3}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "PIPE_STAGES": 4}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "PIPE_STAGES": 3}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "PIPE_STAGES": 4}, num_warps=4, num_stages=4),
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
    start_m = tl.program_id(0) * BLOCK_M
    if start_m >= S:
        return

    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H

    # 4D TMA block descriptors for out-of-bounds safety and optimal hardware loads
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

    # Initial query load mapping 4D chunk logically into contiguous 2D chunk shape
    q_blk = q_desc.load([off_b, off_h, start_m, 0])
    q = tl.reshape(q_blk, (BLOCK_M, BLOCK_D))
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    
    # Init running state securely avoiding arbitrary NaN generation when scaling
    m_i = tl.where(offs_m < S, -float("inf"), 0.0)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # === UNMASKED BLOCKS ===
    # Fast path fully beneath diagonal intersection boundaries explicitly bypassing condition overhead
    for start_n in tl.range(0, start_m, BLOCK_N, num_stages=PIPE_STAGES):
        k_blk = k_desc.load([off_b, off_h, start_n, 0])
        k = tl.reshape(k_blk, (BLOCK_N, BLOCK_D))
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        # Optimal EX2 instructions executing identical semantics intrinsically bypassing FP32 scale-down logic step
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v_blk = v_desc.load([off_b, off_h, start_n, 0])
        v = tl.reshape(v_blk, (BLOCK_N, BLOCK_D))
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # === MASKED BLOCKS ===
    end_n = tl.minimum(S, start_m + BLOCK_M)
    for start_n in range(start_m, end_n, BLOCK_N):
        k_blk = k_desc.load([off_b, off_h, start_n, 0])
        k = tl.reshape(k_blk, (BLOCK_N, BLOCK_D))
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        valid_mask = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < S)
        qk = tl.where(valid_mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v_blk = v_desc.load([off_b, off_h, start_n, 0])
        v = tl.reshape(v_blk, (BLOCK_N, BLOCK_D))
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Normalization scaling and bfloat16 casting conversion directly before storage
    out = acc / l_i[:, None]
    out = out.to(tl.bfloat16)
    
    out_blk = tl.reshape(out, (1, 1, BLOCK_M, BLOCK_D))
    o_desc.store([off_b, off_h, start_m, 0], out_blk)
    
    # Write correctly offset safe Natural log-sum-exp variables utilizing generic standard LSE Pointers 
    lse = m_i + tl.log2(l_i)
    lse = lse * 0.6931471805599453 # element-wise conversion back onto base-e target representation via ln(2) multiplier scale
    lse_ptrs = LSE + off_b * stride_lseb + off_h * stride_lseh + offs_m * stride_lses
    mask_m = offs_m < S
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    # Scale applied beforehand factoring log2(e) for optimum mathematical inner processing pipeline evaluation
    sm_scale_log2 = (1.0 / math.sqrt(D)) * 1.4426950408889634
    
    # Ordering grid sequentially maps M Tiles representing independent query bounds per distinct SM core handling all elements
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