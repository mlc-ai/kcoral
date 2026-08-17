import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),  # High parallel WGMMA (~224 KiB)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=5),
    ],
    key=['S']
)
@triton.jit
def mha_fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    sm_scale_log2,
    EVEN_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    if EVEN_S:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < S, other=0.0)
        
    # Natively integrate mathematical scalar adjustments efficiently avoiding redundant loop executions
    q = (q * sm_scale_log2).to(q.dtype)
    
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Fully branchless hardware-pipelined iteration block
    num_steps = S // BLOCK_N
    for step in range(num_steps):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # Generates implicit SM90 WGMMA instructions tracking bf16 > fp32 accumulator layouts
        qk = tl.dot(q, k.T)
        
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        p_bf16 = p.to(q.dtype)
        acc = tl.dot(p_bf16, v, acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs
        
    # Dynamically peeled condition isolating unaligned tracking limits resolving branch-prediction losses entirely
    if not EVEN_S:
        k = tl.load(k_ptrs, mask=offs_n[:, None] < (S % BLOCK_N), other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < (S % BLOCK_N), other=0.0)
        
        qk = tl.dot(q, k.T)
        qk = tl.where(offs_n[None, :] < (S % BLOCK_N), qk, float('-inf'))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        
        p_bf16 = p.to(q.dtype)
        acc = tl.dot(p_bf16, v, acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
    acc = acc / l_i[:, None]
    
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    if EVEN_S:
        tl.store(o_ptrs, acc.to(q.dtype))
    else:
        tl.store(o_ptrs, acc.to(q.dtype), mask=offs_m[:, None] < S)
        
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    
    # Scale from Hopper native high-speed Base-2 natural layout expected by original PyTorch ATen formula 
    ln2 = 0.6931471805599453
    lse = (m_i + tl.log2(l_i)) * ln2
    
    if EVEN_S:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=offs_m < S)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    sm_scale = 1.0 / (D ** 0.5)
    log2_e = 1.4426950408889634
    sm_scale_log2 = sm_scale * log2_e
    
    EVEN_S = (S % 128 == 0)
    
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H
    )
    
    mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        sm_scale_log2,
        EVEN_S=EVEN_S,
        BLOCK_D=128
    )