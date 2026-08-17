import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['S']
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    b = off_hz // H
    h = off_hz % H
    
    # Base pointers for the current batch and head
    q_ptrs = Q + b * stride_qb + h * stride_qh
    k_ptrs = K + b * stride_kb + h * stride_kh
    v_ptrs = V + b * stride_vb + h * stride_vh
    o_ptrs = O + b * stride_ob + h * stride_oh
    lse_ptrs = LSE + b * stride_lseb + h * stride_lseh
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    # Q block is [BLOCK_M, BLOCK_D]
    q_offs = offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs + q_offs, mask=offs_m[:, None] < S, other=0.0)
    
    # Load K and V as [BLOCK_N, BLOCK_D] tiles physically to avoid transposed strides issue,
    # then use K.T for the dot product A @ B.T where natively supported
    k_ptrs = k_ptrs + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = v_ptrs + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Stable softmax tracking variables
    m_i = tl.where(offs_m < S, -float("inf"), 0.0)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    num_steps = (S + BLOCK_N - 1) // BLOCK_N
    for i in range(num_steps):
        start_n = i * BLOCK_N
        curr_n = start_n + offs_n
        
        # Load blocks
        k = tl.load(k_ptrs, mask=curr_n[:, None] < S, other=0.0)
        v = tl.load(v_ptrs, mask=curr_n[:, None] < S, other=0.0)
        
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # k is [BLOCK_N, BLOCK_D], k.T is [BLOCK_D, BLOCK_N]
        qk = tl.dot(q, k.T, qk)
        qk = qk * sm_scale
        
        # Mask out-of-bounds keys for valid and padded queries
        qk = tl.where((offs_m[:, None] < S) & (curr_n[None, :] < S), qk, float("-inf"))
        
        # Row-wise max over the current tile
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Update sum of exps and intermediate accumulator
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        # Commit running stats
        m_i = m_i_new
        l_i = l_i_new
        
        # Advance pointers along sequence length
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    # Epilogue: scale the final sum and calculate LogSumExp (LSE)
    l_i_safe = tl.where(offs_m < S, l_i, 1.0)
    acc = acc / l_i_safe[:, None]
    lse = m_i + tl.log(l_i)
    
    # Store O
    o_offs = offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs + o_offs, acc.to(tl.bfloat16), mask=offs_m[:, None] < S)
    
    # Store LSE
    lse_offs = offs_m * stride_lses
    tl.store(lse_ptrs + lse_offs, lse, mask=offs_m < S)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    sm_scale = 1.0 / (D ** 0.5)
    
    # Grid limits program ids cleanly aligned to blocks
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    
    _fwd_kernel[grid](
        Q, K, V, sm_scale,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        BLOCK_D=128
    )