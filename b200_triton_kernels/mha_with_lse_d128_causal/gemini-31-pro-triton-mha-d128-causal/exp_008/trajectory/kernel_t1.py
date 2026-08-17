import torch
import triton
import triton.language as tl

def get_autotune_configs():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
    ]

@triton.autotune(
    configs=get_autotune_configs(),
    key=['N_CTX'],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, O, LSE,
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_km,
    stride_vz, stride_vh, stride_vm,
    stride_oz, stride_oh, stride_om,
    stride_lsez, stride_lseh, stride_lsem,
    B, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_z * stride_qz + off_h * stride_qh
    k_base = K + off_z * stride_kz + off_h * stride_kh
    v_base = V + off_z * stride_vz + off_h * stride_vh
    o_base = O + off_z * stride_oz + off_h * stride_oh

    # Hardware TensorDescriptors for Hopper TMA
    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[N_CTX, BLOCK_DMODEL],
        strides=[stride_qm, 1],
        block_shape=[BLOCK_M, BLOCK_DMODEL],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[N_CTX, BLOCK_DMODEL],
        strides=[stride_km, 1],
        block_shape=[BLOCK_N, BLOCK_DMODEL],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[N_CTX, BLOCK_DMODEL],
        strides=[stride_vm, 1],
        block_shape=[BLOCK_N, BLOCK_DMODEL],
        padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base,
        shape=[N_CTX, BLOCK_DMODEL],
        strides=[stride_om, 1],
        block_shape=[BLOCK_M, BLOCK_DMODEL]
    )

    q = q_desc.load([start_m * BLOCK_M, 0])

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    seq_len = N_CTX
    max_n = tl.minimum(seq_len, (start_m + 1) * BLOCK_M)
    
    # Safe boundary where causal masking and seq_len checking are unnecessary
    unmasked_limit = tl.minimum(seq_len, start_m * BLOCK_M)
    unmasked_steps = unmasked_limit // BLOCK_N

    # Fully unmasked loop
    for step in range(0, unmasked_steps):
        start_n = step * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, qk)
        qk = qk * sm_scale
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, 1)
        
        p_cast = p.to(q.dtype.element_ty)
        acc = acc * alpha[:, None]
        acc = tl.dot(p_cast, v, acc)
        
        m_i = m_i_new

    # Masked loop for causal boundary and sequence tail
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    for step in range(unmasked_steps, tl.cdiv(max_n, BLOCK_N)):
        start_n = step * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, qk)
        qk = qk * sm_scale
        
        curr_n = start_n + offs_n
        causal_mask = offs_m[:, None] >= curr_n[None, :]
        seq_mask = curr_n[None, :] < seq_len
        valid_mask = causal_mask & seq_mask
        
        qk = tl.where(valid_mask, qk, float("-inf"))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, 1)
        
        p_cast = p.to(q.dtype.element_ty)
        acc = acc * alpha[:, None]
        acc = tl.dot(p_cast, v, acc)
        
        m_i = m_i_new

    # Finalize softmax scaling
    l_i_log = tl.log(l_i)
    lse = m_i + l_i_log
    out = acc / l_i[:, None]

    # TMA store automatically respects boundaries, dropping padded row writes
    o_desc.store([start_m * BLOCK_M, 0], out.to(q.dtype.element_ty))

    # Save LSE
    lse_base = LSE + off_z * stride_lsez + off_h * stride_lseh
    lse_ptrs = lse_base + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=offs_m < seq_len)

_allocator_set = False
def _tma_alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def run(Q, K, V, O, LSE):
    global _allocator_set
    if not _allocator_set:
        triton.set_allocator(_tma_alloc_fn)
        _allocator_set = True

    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    if S == 0:
        return
        
    sm_scale = 1.0 / (D ** 0.5)
    BLOCK_DMODEL = 128

    # grid resolves META dictionary provided by `@triton.autotune`
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    
    _fwd_kernel[grid](
        Q, K, V, sm_scale, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        BLOCK_DMODEL=BLOCK_DMODEL
    )