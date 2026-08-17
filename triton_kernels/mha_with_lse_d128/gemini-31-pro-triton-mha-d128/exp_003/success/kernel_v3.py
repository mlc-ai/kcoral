import torch
import triton
import triton.language as tl

# Infrastructure storage allocator for device-created TMA descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
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
    B, H, S, D,
    sm_scale,
    EVEN_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh
    
    # Hopper TMA descriptors orchestrating direct bounds-safe memory transfers
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D]
    )
    
    offset_m = pid_m * BLOCK_M
    q = q_desc.load([offset_m, 0])
    
    # Pre-scale mathematics natively matching Hopper base-2 requirements outside iteration space 
    log2_e = 1.4426950408889634
    q = (q * (sm_scale * log2_e)).to(q.dtype)
    
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Core software-pipelined iteration strictly decoupled from conditional bounds checking 
    num_steps = S // BLOCK_N
    for step in range(num_steps):
        offset_n = step * BLOCK_N
        
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # Native WGMMA processing implicitly aligned without transposed physical constraints
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
        
    # Dynamically peel the singular looping edge tail enforcing bound masking overhead isolatedly
    if not EVEN_S:
        offset_n = num_steps * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        qk = tl.dot(q, k.T)
        
        offs_n_curr = offset_n + tl.arange(0, BLOCK_N)
        qk = tl.where(offs_n_curr[None, :] < S, qk, float('-inf'))
        
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
    
    # Memory writes gracefully drop any values trailing bounds seamlessly padded by TMA
    o_desc.store([offset_m, 0], acc.to(q.dtype))
    
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    
    ln2 = 0.6931471805599453
    lse = (m_i + tl.log2(l_i)) * ln2
    
    # Store cleanly dropping conditional writes if dimensions perfectly match
    if EVEN_S:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    """
    Executes mathematically precise Multi-Head Softmax Attention projecting seamlessly 
    atop Hoppers internal SM90 architectural structures efficiently.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    if S == 0:
        return
        
    sm_scale = 1.0 / (D ** 0.5)
    
    # Universal LCM coverage targeting dynamic lengths protecting our unrolled bounds optimizations
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
        B, H, S, D,
        sm_scale,
        EVEN_S=EVEN_S,
        BLOCK_D=128
    )