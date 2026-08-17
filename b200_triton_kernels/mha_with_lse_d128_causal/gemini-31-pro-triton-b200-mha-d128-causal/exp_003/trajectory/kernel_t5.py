import torch
import triton
import triton.language as tl
import math

def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "NUM_STAGES": 3, "WARP_SPEC": True}, num_warps=8, num_stages=3, num_ctas=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "NUM_STAGES": 4, "WARP_SPEC": True}, num_warps=8, num_stages=4, num_ctas=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "NUM_STAGES": 3, "WARP_SPEC": True}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "NUM_STAGES": 4, "WARP_SPEC": True}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "NUM_STAGES": 3, "WARP_SPEC": False}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "NUM_STAGES": 3, "WARP_SPEC": False}, num_warps=8, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    sm_scale,
    B, H, S,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    WARP_SPEC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    V_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    O_ptr = O + pid_b * stride_ob + pid_h * stride_oh

    q_desc = tl.make_tensor_descriptor(
        Q_ptr, shape=[S, D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K_ptr, shape=[S, D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr, shape=[S, D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O_ptr, shape=[S, D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )

    m_start = pid_m * BLOCK_M
    q = q_desc.load([m_start, 0])
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    limit = tl.minimum(S, m_start + BLOCK_M)
    loop_end = tl.cdiv(limit, BLOCK_N)
    
    offs_m = m_start + tl.arange(0, BLOCK_M)
    
    for k_idx in tl.range(0, loop_end, num_stages=NUM_STAGES, warp_specialize=WARP_SPEC):
        start_n = k_idx * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        offs_n = start_n + tl.arange(0, BLOCK_N)
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        
        acc = acc * alpha[:, None]
        
        beta_bf16 = beta.to(tl.bfloat16)
        acc += tl.dot(beta_bf16, v, out_dtype=tl.float32)
        
        l_i = l_i * alpha + tl.sum(beta, 1)
        m_i = m_new

    out = acc / l_i[:, None]
    o_desc.store([m_start, 0], out.to(tl.bfloat16))
    
    LSE_ptr = LSE + pid_b * stride_lseb + pid_h * stride_lseh
    LSE_ptrs = LSE_ptr + offs_m * stride_lses
    lse_val = m_i + tl.log(l_i)
    tl.store(LSE_ptrs, lse_val, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)

    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        sm_scale,
        B, H, S,
        D,
    )