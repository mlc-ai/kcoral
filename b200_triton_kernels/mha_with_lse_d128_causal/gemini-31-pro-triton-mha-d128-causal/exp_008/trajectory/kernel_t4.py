import torch
import triton
import triton.language as tl

def get_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=5, num_warps=4),
    ]

@triton.autotune(
    configs=get_configs(),
    key=['N_CTX']
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, O, LSE,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_km, stride_kd,
    stride_vz, stride_vh, stride_vm, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_lsez, stride_lseh, stride_lsem,
    B, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + (off_z * stride_qz + off_h * stride_qh) + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = K + (off_z * stride_kz + off_h * stride_kh) + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
    v_ptrs = V + (off_z * stride_vz + off_h * stride_vh) + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd

    # Load query tile and pad with zeros if out of sequence length
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # Scale Q beforehand to save FP32 multiplies inside the critical inner loops
    q = (q * sm_scale).to(Q.dtype.element_ty)

    # Accumulators for Safe-Softmax online reduction
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    limit = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M)
    unmasked_steps = tl.minimum(N_CTX, start_m * BLOCK_M) // BLOCK_N

    # ====================================================================================
    # 1. Fully Unmasked Loop:
    # Processes the prefix keys where the causal mask is provably fully true.
    # No sequence or causal masking allows highly efficient block loads and software pipelining.
    # ====================================================================================
    for step in range(0, unmasked_steps):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        # Matrix multiply (Hopper Tensor Core WGMMA matching the col-major K.T optimization)
        qk = tl.dot(q, k.T)

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, 1)
        
        p_cast = p.to(V.dtype.element_ty)
        acc = acc * alpha[:, None]
        acc = tl.dot(p_cast, v, acc)
        
        m_i = m_i_new
        k_ptrs += BLOCK_N * stride_km
        v_ptrs += BLOCK_N * stride_vm
        offs_n += BLOCK_N

    # ====================================================================================
    # 2. Masked Loop:
    # Processes the diagonal boundary blocks that require causal masking.
    # ====================================================================================
    total_steps = tl.cdiv(limit, BLOCK_N)
    for step in range(unmasked_steps, total_steps):
        seq_mask = offs_n[:, None] < N_CTX
        k = tl.load(k_ptrs, mask=seq_mask, other=0.0)
        v = tl.load(v_ptrs, mask=seq_mask, other=0.0)
        
        qk = tl.dot(q, k.T)

        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, 1)
        
        p_cast = p.to(V.dtype.element_ty)
        acc = acc * alpha[:, None]
        acc = tl.dot(p_cast, v, acc)
        
        m_i = m_i_new
        k_ptrs += BLOCK_N * stride_km
        v_ptrs += BLOCK_N * stride_vm
        offs_n += BLOCK_N

    # Epilogue: LSE formulation & finalizing output
    l_i_log = tl.log(l_i)
    lse = m_i + l_i_log
    out = acc / l_i[:, None]

    o_ptrs = O + (off_z * stride_oz + off_h * stride_oh) + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)

    lse_ptrs = LSE + (off_z * stride_lsez + off_h * stride_lseh) + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=offs_m < N_CTX)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    if S == 0:
        return
        
    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_DMODEL = 128

    # The grid dimensions leverage META dictionary provided by `@triton.autotune`
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    
    _fwd_kernel[grid](
        Q, K, V, sm_scale, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        BLOCK_DMODEL=BLOCK_DMODEL
    )