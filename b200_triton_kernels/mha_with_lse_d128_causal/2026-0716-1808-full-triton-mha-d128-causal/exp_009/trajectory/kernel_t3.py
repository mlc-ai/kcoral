import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, H, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    head = pid_bh % H
    batch = pid_bh // H
    
    row_start = pid_m * BLOCK_M
    
    valid_rows = min(BLOCK_M, max(0, S - row_start))
    if valid_rows <= 0:
        return
    
    H_4d = H
    Q_ptr_4d = Q_ptr + (batch * H_4d + head) * S * D
    K_ptr_4d = K_ptr + (batch * H_4d + head) * S * D
    V_ptr_4d = V_ptr + (batch * H_4d + head) * S * D
    O_ptr_4d = O_ptr + (batch * H_4d + head) * S * D
    
    LSE_ptr_4d = LSE_ptr + (batch * H_4d + head) * S
    
    Q_desc = tl.make_tensor_descriptor(Q_ptr_4d, shape=[S, D], strides=[D, 1], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(K_ptr_4d, shape=[S, D], strides=[D, 1], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(V_ptr_4d, shape=[S, D], strides=[D, 1], block_shape=[BLOCK_N, BLOCK_K], padding_option="zero")
    O_desc = tl.make_tensor_descriptor(O_ptr_4d, shape=[S, D], strides=[D, 1], block_shape=[BLOCK_M, BLOCK_K], padding_option="zero")
    
    LSE_desc = tl.make_tensor_descriptor(LSE_ptr_4d, shape=[S], strides=[1], block_shape=[BLOCK_M], padding_option="zero")
    
    q_tile = Q_desc.load([row_start, 0])
    
    global_row = row_start + row_offsets[:, None]
    valid_q = (global_row < S)
    q_tile = tl.where(valid_q, q_tile, 0.0)
    
    o = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    m_prev = tl.full((BLOCK_M,), -1e20, tl.float32)
    l_prev = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    if S % BLOCK_N == 0:
        total_blocks = (row_start + BLOCK_M) // BLOCK_N
    else:
        total_blocks = (row_start + BLOCK_M) // BLOCK_N + 1
        
    row_offsets = tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)
    
    for block_idx in range(min(total_blocks, S // BLOCK_N)):
        col_start = block_idx * BLOCK_N
        
        global_col = col_start + col_offsets[None, :]
        valid_k = (global_col < S)
        
        k_tile = K_desc.load([col_start, 0])
        k_tile = tl.where(valid_k, k_tile, 0.0)
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        for i in range(0, BLOCK_N, BLOCK_K):
            q_chunk = q_tile[:, i : i + BLOCK_K]
            k_chunk = k_tile[:, i : i + BLOCK_K]
            s += tl.dot(q_chunk, k_chunk.T)
            
        s = s * scale
        
        mask = ((col_start + col_offsets[None, :]) <= (row_start + row_offsets[:, None]))
        
        s = tl.where(mask, s, -1e20)
        
        m_curr = tl.maximum(m_prev, tl.max(s, axis=1))
        alpha = tl.exp(m_prev - m_curr)
        o = o * alpha[:, None]
        
        exp_s = tl.exp(s - m_curr)
        p = tl.where(mask, exp_s, 0.0)
        
        l_prev = l_prev * alpha + tl.sum(p, axis=1)
        m_prev = m_curr
        
        v_tile = V_desc.load([col_start, 0])
        v_tile = tl.where(valid_k, v_tile, 0.0)
        
        for i in range(0, BLOCK_N, BLOCK_K):
            p_chunk = p[:, i : i + BLOCK_K]
            v_chunk = v_tile[:, i : i + BLOCK_K]
            o += tl.dot(p_chunk, v_chunk)
            
    o = o / l_prev[:, None]
    
    lse = m_prev + tl.log(l_prev)
    
    store_mask = row_offsets < valid_rows
    O_desc.store([row_start, 0], o.to(tl.bfloat16))
    LSE_desc.store([row_start], lse, mask=store_mask)


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
    BLOCK_K = 128
    
    num_programs_m = triton.cdiv(S, BLOCK_M)
    num_programs_bh = B * H
    grid = (num_programs_m, num_programs_bh)
    
    scale = 1.0 / math.sqrt(128)
    
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        S, H, scale,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=2,
    )