import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    b = off_hz // H
    h = off_hz % H

    q_offset = b * stride_qb + h * stride_qh
    k_offset = b * stride_kb + h * stride_kh
    v_offset = b * stride_vb + h * stride_vh
    o_offset = b * stride_ob + h * stride_oh
    lse_offset = b * stride_lseb + h * stride_lseh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    
    q_mask = offs_m[:, None] < S
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize statistics
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    m_i = tl.where(offs_m < S, m_i, 0.0)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    sm_scale = 1.0 / (float(D) ** 0.5)

    num_steps = tl.cdiv(S, BLOCK_N)
    for step in range(0, num_steps):
        start_n = step * BLOCK_N
        offs_n_curr = start_n + offs_n
        kv_mask = offs_n_curr[:, None] < S
        
        # Load K in row-major block for fast transposed WGMMA
        k_ptrs = K + k_offset + offs_n_curr[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        
        # Compute Q @ K.T (FP32 Accumulation)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, qk)
        qk = qk * sm_scale
        
        qk_mask = (offs_m[:, None] < S) & (offs_n_curr[None, :] < S)
        qk = tl.where(qk_mask, qk, float("-inf"))
        
        # Safe log-sum-exp updates
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        m_i_new = tl.where(offs_m < S, m_i_new, 0.0)
        
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        p = tl.where(qk_mask, p, 0.0)
        
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        # Second dot product
        p_bf16 = p.to(V.dtype.element_ty)
        v_ptrs = V + v_offset + offs_n_curr[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p_bf16, v, acc)
        
        m_i = m_i_new
        l_i = l_i_new

    # Write output
    l_i_safe = tl.where(offs_m < S, l_i, 1.0)
    acc = acc / l_i_safe[:, None]
    out = acc.to(O.dtype.element_ty)
    
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out, mask=q_mask)
    
    # Write LSE
    lse = m_i + tl.log(l_i_safe)
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    """
    Computes Standard Scaled Dot-Product Attention (Forward Pass).
    Writes to the preallocated O and LSE output tensors.
    """
    torch.cuda.set_device(Q.device)
    
    B_sz, H_sz, S_sz, D_sz = Q.shape
    
    grid = lambda META: (
        triton.cdiv(S_sz, META["BLOCK_M"]),
        B_sz * H_sz
    )
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B=B_sz, H=H_sz, S=S_sz, D=D_sz,
    )