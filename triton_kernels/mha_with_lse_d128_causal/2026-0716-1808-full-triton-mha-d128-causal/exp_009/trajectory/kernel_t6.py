import torch
import triton
import triton.language as tl
import math


@triton.jit
def make_desc(base_ptr, shape, strides, block_shape):
    return tl.make_tensor_descriptor(
        base_ptr,
        shape=shape,
        strides=strides,
        block_shape=block_shape,
        padding_option="zero"
    )


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, H, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    head = pid_bh % H
    batch = pid_bh // H
    
    row_start = pid_m * BLOCK_M
    
    block_idx_max = (row_start + BLOCK_M - 1) // BLOCK_N
    total_blocks = min(block_idx_max + 1, (S + BLOCK_N - 1) // BLOCK_N)
    
    if total_blocks == 0:
        return
    
    H_4d = H
    D_4d = 128
    stride_bh = S * D_4d
    
    q_base = Q_ptr + (batch * H_4d + head) * stride_bh
    k_base = K_ptr + (batch * H_4d + head) * stride_bh
    v_base = V_ptr + (batch * H_4d + head) * stride_bh
    o_base = O_ptr + (batch * H_4d + head) * stride_bh
    lse_base = LSE_ptr + (batch * H_4d + head) * S
    
    Q_desc = make_desc(q_base, [S, D_4d], [D_4d, 1], [BLOCK_M, 64])
    K_desc = make_desc(k_base, [S, D_4d], [D_4d, 1], [BLOCK_N, 64])
    V_desc = make_desc(v_base, [S, D_4d], [D_4d, 1], [BLOCK_N, 64])
    O_desc = make_desc(o_base, [S, D_4d], [D_4d, 1], [BLOCK_M, 64])
    LSE_desc = make_desc(lse_base, [S], [1], [BLOCK_M])
    
    q0 = Q_desc.load([row_start, 0])
    q1 = Q_desc.load([row_start, 64])
    
    o0 = tl.zeros((BLOCK_M, 64), tl.float32)
    o1 = tl.zeros((BLOCK_M, 64), tl.float32)
    m_prev = tl.full((BLOCK_M,), -1e20, tl.float32)
    l_prev = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    row_offsets = tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)
    
    mb = tl.malloc_shared(tl.int32, 1)
    tl.atomic_xchg(mb, 0)
    
    col_start = 0
    k0 = K_desc.load_async(col_start, 0, mb)
    k1 = K_desc.load_async(col_start, 64, mb)
    v0 = V_desc.load_async(col_start, 0, mb)
    v1 = V_desc.load_async(col_start, 64, mb)
    tl.expect(tl.atomic_add(mb, -4), 0)
    
    for block_idx in tl.range(0, total_blocks, 1, flatten=False, warp_specialize=WARP_SPECIALIZE):
        
        if block_idx + 1 < total_blocks:
            mb_next = tl.malloc_shared(tl.int32, 1)
            tl.atomic_xchg(mb_next, 0)
            
            next_col_start = (block_idx + 1) * BLOCK_N
            k0_next = K_desc.load_async(next_col_start, 0, mb_next)
            k1_next = K_desc.load_async(next_col_start, 64, mb_next)
            v0_next = V_desc.load_async(next_col_start, 0, mb_next)
            v1_next = V_desc.load_async(next_col_start, 64, mb_next)
            tl.expect(tl.atomic_add(mb_next, -4), 0)
        else:
            mb_next = mb
            
        tl.wait_for_mb(mb, 0)
        
        global_row = row_start + row_offsets[:, None]
        global_col = col_start + col_offsets[None, :]
        
        valid_q = (global_row < S)
        q0 = tl.where(valid_q, q0, 0.0)
        q1 = tl.where(valid_q, q1, 0.0)
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s += tl.dot(q0, k0.T)
        s += tl.dot(q1, k1.T)
        
        s *= scale
        
        mask = (global_col <= global_row)
        
        s_masked = s * mask
        masked_max = tl.maximum(tl.full((BLOCK_M,), -1e20, tl.float32), tl.max(s_masked, axis=-1))
        
        m_curr = tl.maximum(m_prev, masked_max)
        alpha = tl.exp(m_prev - m_curr)
        o0 *= alpha[:, None]
        o1 *= alpha[:, None]
        
        exp_s = tl.exp(s - m_curr)
        p = exp_s * mask
        
        l_prev = l_prev * alpha + tl.sum(p, axis=-1)
        m_prev = m_curr
        
        p_bf16 = p.to(tl.bfloat16)
        
        o0 += tl.dot(p_bf16, v0)
        o1 += tl.dot(p_bf16, v1)
        
        mb = mb_next
        k0 = k0_next
        k1 = k1_next
        v0 = v0_next
        v1 = v1_next
        col_start = (block_idx + 1) * BLOCK_N
        
    if m_prev[0] > -1e19:
        o0 /= l_prev[:, None]
        o1 /= l_prev[:, None]
    
    lse = m_prev + tl.log(l_prev)
    
    valid_q = (row_start + row_offsets) < S
    o0 = tl.where(valid_q[:, None], o0, 0.0)
    o1 = tl.where(valid_q[:, None], o1, 0.0)
    lse = tl.where(valid_q, lse, 0.0)
    
    O_desc.store([row_start, 0], o0.to(tl.bfloat16))
    O_desc.store([row_start, 64], o1.to(tl.bfloat16))
    LSE_desc.store([row_start], lse)


def run(Q, K, V, O, LSE):
    """Compute causal MHA and log-sum-exp into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    assert B == 4 and H == 48 and D == 128
    assert Q.dtype == torch.bfloat16 and K.dtype == torch.bfloat16 and V.dtype == torch.bfloat16
    assert O.shape == (B, H, S, D) and O.dtype == torch.bfloat16
    assert LSE.shape == (B, H, S) and LSE.dtype == torch.float32
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    num_programs_m = triton.cdiv(S, BLOCK_M)
    num_programs_bh = B * H
    grid = (num_programs_m, num_programs_bh)
    
    scale = 1.0 / math.sqrt(128)
    
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        S, H, scale,
        BLOCK_M, BLOCK_N,
        WARP_SPECIALIZE=True,
        num_warps=4,
        num_stages=4,
    )