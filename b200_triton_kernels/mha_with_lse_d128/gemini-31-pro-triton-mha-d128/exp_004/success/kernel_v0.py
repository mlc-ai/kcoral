import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def mha_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale,
    S,
    H,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offset_q = b * stride_qb + h * stride_qh
    offset_k = b * stride_kb + h * stride_kh
    offset_v = b * stride_vb + h * stride_vh
    offset_o = b * stride_ob + h * stride_oh
    offset_lse = b * stride_lseb + h * stride_lseh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + offset_q + offs_m[:, None] * stride_qs + offs_d[None, :]
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < S), other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    k_ptrs = K + offset_k + offs_n[:, None] * stride_ks + offs_d[None, :]
    v_ptrs = V + offset_v + offs_n[:, None] * stride_vs + offs_d[None, :]

    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n_block_idx in range(num_n_blocks):
        curr_n = n_block_idx * BLOCK_N + offs_n
        k = tl.load(k_ptrs, mask=(curr_n[:, None] < S), other=0.0)
        v = tl.load(v_ptrs, mask=(curr_n[:, None] < S), other=0.0)
        
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale
        
        mask = (offs_m[:, None] < S) & (curr_n[None, :] < S)
        qk = tl.where(mask, qk, float("-inf"))
        
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        
        l_ij = tl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        p_bf16 = tl.cast(p, tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc, out_dtype=tl.float32)
        
        m_i = m_new
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    o_ptrs = O + offset_o + offs_m[:, None] * stride_os + offs_d[None, :]
    tl.store(o_ptrs, tl.cast(acc, tl.bfloat16), mask=(offs_m[:, None] < S))

    lse_ptrs = LSE + offset_lse + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=(offs_m < S))

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B = Q.shape[0]
    H = Q.shape[1]
    S = Q.shape[2]
    D = Q.shape[3]
    
    sm_scale = 1.0 / (D ** 0.5)
    
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H, 1)
    
    mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale,
        S,
        H,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_D=D,
    )