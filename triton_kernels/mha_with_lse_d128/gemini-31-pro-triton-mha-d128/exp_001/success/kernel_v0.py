import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
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
    stride_lb, stride_lh, stride_ls,
    S,
    sm_scale,
    num_heads,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    
    b = bh // num_heads
    h = bh % num_heads

    m_offs = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_D)
    
    q_ptrs = Q + (b * stride_qb + h * stride_qh + m_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd)
    mask_m = m_offs < S
    
    # Load Q and scale it
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * sm_scale).to(tl.bfloat16)

    # Initialize running accumulator for max, sum, and output
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    n_offs = tl.arange(0, BLOCK_N)
    k_ptrs = K + (b * stride_kb + h * stride_kh + n_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd)
    v_ptrs = V + (b * stride_vb + h * stride_vh + n_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd)

    num_n_blocks = tl.cdiv(S, BLOCK_N)
    for n_idx in range(num_n_blocks):
        n_start = n_idx * BLOCK_N
        n_offs_cur = n_start + n_offs
        mask_n = n_offs_cur < S
        
        # Load K and V
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        # Q @ K.T
        qk = tl.dot(q, k.T)
        qk = tl.where(mask_n[None, :], qk, -float('inf'))
        
        # Standard FlashAttention logic
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(qk - m_i_new[:, None])
        
        l_ij = tl.sum(beta, 1)
        l_i_new = l_i * alpha + l_ij
        
        # Scale previously accumulated values
        acc = acc * alpha[:, None]
        p = beta.to(tl.bfloat16)
        
        # Update output
        acc = tl.dot(p, v, acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
        # Advance pointers
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # Finalize LSE and Output
    lse = m_i + tl.log(l_i)
    out = acc / l_i[:, None]
    
    # Store output
    o_ptrs = O + (b * stride_ob + h * stride_oh + m_offs[:, None] * stride_os + d_offs[None, :] * stride_od)
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=mask_m[:, None])
    
    # Store LSE
    lse_ptrs = LSE + (b * stride_lb + h * stride_lh + m_offs * stride_ls)
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    sm_scale = 1.0 / (D ** 0.5)
    
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B * H,
        1
    )
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        sm_scale,
        num_heads=H,
        BLOCK_D=128,
    )