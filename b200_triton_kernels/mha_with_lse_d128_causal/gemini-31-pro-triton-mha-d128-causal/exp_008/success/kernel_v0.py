import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale, O, LSE,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_km, stride_kd,
    stride_vz, stride_vh, stride_vm, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    stride_lsez, stride_lseh, stride_lsem,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    lse_offset = off_z * stride_lsez + off_h * stride_lseh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    k_ptrs = K + k_offset + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd

    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Causal masking: limit maximum sequence index based on the query block
    max_n = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M)
    num_steps = tl.cdiv(max_n, BLOCK_N)
    
    for step in range(num_steps):
        start_n = step * BLOCK_N
        curr_n = start_n + offs_n

        k = tl.load(k_ptrs, mask=curr_n[:, None] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=curr_n[:, None] < N_CTX, other=0.0)
        
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, tl.trans(k), qk)
        qk = qk * sm_scale

        # Apply causal mask and sequence length bounds
        mask = (offs_m[:, None] >= curr_n[None, :]) & (curr_n[None, :] < N_CTX)
        qk = tl.where(mask, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        l_i = l_i * alpha + tl.sum(p, 1)

        # Cast p to matching precision of V for accurate dot operation accumulation
        p_cast = p.to(V.dtype.element_ty)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p_cast, v, acc)
        
        m_i = m_i_new
        
        k_ptrs += BLOCK_N * stride_km
        v_ptrs += BLOCK_N * stride_vm

    l_i_log = tl.log(l_i)
    lse = m_i + l_i_log
    out = acc / l_i[:, None]

    o_ptrs = O + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)

    lse_ptrs = LSE + lse_offset + offs_m * stride_lsem
    tl.store(lse_ptrs, lse, mask=offs_m < N_CTX)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DMODEL = 128

    if S == 0:
        return

    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    _fwd_kernel[grid](
        Q, K, V, sm_scale, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4, num_stages=3
    )