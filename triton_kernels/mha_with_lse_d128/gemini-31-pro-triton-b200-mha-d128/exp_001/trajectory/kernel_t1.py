import torch
import triton
import triton.language as tl

# Set allocator for device-created TMA descriptors
def _tma_alloc(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(_tma_alloc)

def get_autotune_configs():
    configs = []
    for ws in [False, True]:
        for m, n, w, s in [
            (128, 128, 8, 3),
            (128, 64, 4, 3),
            (128, 64, 8, 3),
            (64, 128, 4, 3),
            (64, 64, 4, 3),
            (64, 64, 4, 4),
        ]:
            configs.append(
                triton.Config(
                    {'BLOCK_M': m, 'BLOCK_N': n, 'WARP_SPECIALIZE': ws, 'LOOP_STAGES': s},
                    num_warps=w, num_stages=s
                )
            )
    return configs

@triton.autotune(configs=get_autotune_configs(), key=['S'])
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, D,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    start_m = pid_m * BLOCK_M

    # Base pointers for the current batch and head
    q_ptr = Q + b * stride_qb + h * stride_qh
    k_ptr = K + b * stride_kb + h * stride_kh
    v_ptr = V + b * stride_vb + h * stride_vh
    o_ptr = O + b * stride_ob + h * stride_oh

    # Device descriptors for TMA
    q_desc = tl.make_tensor_descriptor(
        q_ptr,
        shape=[S, D],
        strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr,
        shape=[S, D],
        strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr,
        shape=[S, D],
        strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr,
        shape=[S, D],
        strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )

    # Load Q tile
    q = q_desc.load([start_m, 0])

    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m_i = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S

    for start_n in tl.range(0, S, BLOCK_N, num_stages=LOOP_STAGES, warp_specialize=WARP_SPECIALIZE):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        # Load K and V tiles via TMA
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])

        # Compute Q @ K^T
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # Mask out-of-bounds keys and queries
        qk = tl.where(mask_n[None, :], qk, float('-inf'))
        qk = tl.where(mask_m[:, None], qk, float('-inf'))

        # Running max and exponential sum
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        # Scale accumulator and update running statistics
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        acc = acc * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)
        
        # Accumulate P @ V
        acc = tl.dot(p_bf16, v, acc, out_dtype=tl.float32)

    # Normalize the accumulator
    acc = acc / l_i[:, None]
    out = acc.to(tl.bfloat16)
    
    # Store O tile via TMA
    o_desc.store([start_m, 0], out)

    # Compute and store LSE
    lse = m_i + tl.log(l_i)
    lse_base = LSE + b * stride_lseb + h * stride_lseh
    lse_ptrs = lse_base + offs_m
    tl.store(lse_ptrs, lse, mask=mask_m)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    if S == 0:
        return

    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    sm_scale = 1.0 / (D ** 0.5)

    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, D,
        sm_scale,
        BLOCK_D=128
    )