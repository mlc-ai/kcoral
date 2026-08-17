import torch
import triton
import triton.language as tl
import math

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

# Configs tuned for SM100 limits, maximizing L2 reuse while remaining well within the 228KB Shared Memory constraint
_configs = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "PIPELINE_STAGES": 3}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "PIPELINE_STAGES": 4}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPELINE_STAGES": 2}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "PIPELINE_STAGES": 2}, num_warps=8, num_stages=2),
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
    PIPELINE_STAGES: tl.constexpr
):
    # Mapping start_m to the X grid dimension so all tiles of the same head concurrently process
    # varying sequence queries, allowing L2 cache perfect sharing of the K & V loads.
    start_m = tl.program_id(0) * BLOCK_M
    if start_m >= S:
        return

    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H

    # 4D TMA block descriptors handling dynamic strided layouts & bounds protections intrinsically
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
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # === UNMASKED BLOCKS ===
    # For a causal mask, all K sequence steps prior to start_m are strictly valid. 
    # Skipping bounds computation entirely yields massive instruction savings here.
    # Software pipelining configured over these predictable bounds utilizing tl.range.
    for start_n in tl.range(0, start_m, BLOCK_N, num_stages=PIPELINE_STAGES):
        k_blk = k_desc.load([off_b, off_h, start_n, 0])
        k = tl.reshape(k_blk, (BLOCK_N, BLOCK_D))
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v_blk = v_desc.load([off_b, off_h, start_n, 0])
        v = tl.reshape(v_blk, (BLOCK_N, BLOCK_D))
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # === MASKED BLOCKS ===
    # Safely compute causal mask boundaries just around the diagonal intersections
    end_n = tl.minimum(S, start_m + BLOCK_M)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    
    for start_n in range(start_m, end_n, BLOCK_N):
        k_blk = k_desc.load([off_b, off_h, start_n, 0])
        k = tl.reshape(k_blk, (BLOCK_N, BLOCK_D))
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        valid_mask = causal_mask & (offs_n[None, :] < S)
        qk = tl.where(valid_mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v_blk = v_desc.load([off_b, off_h, start_n, 0])
        v = tl.reshape(v_blk, (BLOCK_N, BLOCK_D))
        
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Normalize accumulated values to proper domain constraints and bfloat16
    out = acc / l_i[:, None]
    out = out.to(tl.bfloat16)
    
    out_blk = tl.reshape(out, (1, 1, BLOCK_M, BLOCK_D))
    o_desc.store([off_b, off_h, start_m, 0], out_blk)
    
    # Store Natural Log-Sum-Exp accurately via standard global memory pointers
    lse = m_i + tl.log(l_i)
    lse_ptrs = LSE + off_b * stride_lseb + off_h * stride_lseh + offs_m * stride_lses
    mask_m = offs_m < S
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes Scaled Dot-Product Attention accurately in Triton.
    Utilizing the host-driven allocator guarantees Blackwell device TMA 
    descriptors construct smoothly.
    """
    torch.cuda.set_device(Q.device)
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    sm_scale = 1.0 / math.sqrt(D)
    
    # Grid swap strategy targeting maximal L2 Reuse of the V & K elements
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H, 1)
    
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