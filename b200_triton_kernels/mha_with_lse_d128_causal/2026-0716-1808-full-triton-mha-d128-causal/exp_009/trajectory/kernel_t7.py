import torch
import triton
import triton.language as tl
import math


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, H, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
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
    
    Q_desc = tl.make_tensor_descriptor(q_base, shape=[S, D_4d], strides=[D_4d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(k_base, shape=[S, D_4d], strides=[D_4d, 1], block_shape=[64, 64], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(v_base, shape=[S, D_4d], strides=[D_4d, 1], block_shape=[64, 64], padding_option="zero")
    O_desc = tl.make_tensor_descriptor(o_base, shape=[S, D_4d], strides=[D_4d, 1], block_shape=[BLOCK_M, 64], padding_option="zero")
    LSE_desc = tl.make_tensor_descriptor(lse_base, shape=[S], strides=[1], block_shape=[BLOCK_M], padding_option="zero")
    
    q0 = Q_desc.load([row_start, 0])
    q1 = Q_desc.load([row_start, 64])
    
    o0 = tl.zeros((BLOCK_M, 64), tl.float32)
    o1 = tl.zeros((BLOCK_M, 64), tl.float32)
    m_prev = tl.full((BLOCK_M,), -1e20, tl.float32)
    l_prev = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    row_offsets = tl.arange(0, BLOCK_M)
    
    for block_idx in range(total_blocks):
        col_start = block_idx * BLOCK_N
        
        k0_0 = K_desc.load([col_start, 0])
        k0_1 = K_desc.load([col_start + 64, 0])
        k1_0 = K_desc.load([col_start, 64])
        k1_1 = K_desc.load([col_start + 64, 64])
        
        s0 = tl.dot(q0, k0_0.T)
        s0 += tl.dot(q1, k1_0.T)
        
        s1 = tl.dot(q0, k0_1.T)
        s1 += tl.dot(q1, k1_1.T)
        
        s0 *= scale
        s1 *= scale
        
        col_offsets_0 = tl.arange(0, 64)
        col_offsets_1 = 64 + tl.arange(0, 64)
        
        global_row = row_start + row_offsets[:, None]
        global_col_0 = col_start + col_offsets_0[None, :]
        global_col_1 = col_start + col_offsets_1[None, :]
        
        mask0 = (global_col_0 <= global_row)
        mask1 = (global_col_1 <= global_row)
        
        s0 = tl.where(mask0, s0, -1e20)
        s1 = tl.where(mask1, s1, -1e20)
        
        m_curr = tl.maximum(m_prev, tl.maximum(tl.max(s0, axis=-1), tl.max(s1, axis=-1)))
        alpha = tl.exp(m_prev - m_curr)
        o0 *= alpha[:, None]
        o1 *= alpha[:, None]
        
        p0 = tl.exp(s0 - m_curr[:, None]) * mask0
        p1 = tl.exp(s1 - m_curr[:, None]) * mask1
        
        l_prev = l_prev * alpha + tl.sum(p0, axis=-1) + tl.sum(p1, axis=-1)
        m_prev = m_curr
        
        v0_0 = V_desc.load([col_start, 0])
        v0_1 = V_desc.load([col_start + 64, 0])
        v1_0 = V_desc.load([col_start, 64])
        v1_1 = V_desc.load([col_start + 64, 64])
        
        o0 += tl.dot(p0, v0_0)
        o0 += tl.dot(p1, v0_1)
        
        o1 += tl.dot(p0, v1_0)
        o1 += tl.dot(p1, v1_1)
            
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
        num_warps=8,
        num_stages=2,
    )