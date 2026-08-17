import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _mha_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_desc,
    S, H, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    head = pid_bh % H
    batch = pid_bh // H
    
    offset_row = (batch * H + head) * S
    row_start = pid_m * BLOCK_M
    
    valid_rows = min(BLOCK_M, S - row_start)
    if valid_rows <= 0:
        return
        
    q = Q_desc.load([offset_row + row_start, 0])
    
    o = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    m_prev = tl.full((BLOCK_M, 1), -1e20, tl.float32)
    l_prev = tl.full((BLOCK_M, 1), 0.0, tl.float32)
    
    block_idx_max = (row_start + BLOCK_M - 1) // BLOCK_N
    
    row_offsets = tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)
    
    for block_idx in range(block_idx_max + 1):
        col_start = block_idx * BLOCK_N
        
        k = K_desc.load([offset_row + col_start, 0])
        
        s = tl.dot(q, k.T)
        s *= scale
        
        mask = ((col_start + col_offsets[None, :]) <= (row_start + row_offsets[:, None]))
        s = s * mask
        
        m_prev_block = tl.maximum(m_prev, tl.max(s, axis=1)[:, None])
        
        p_scaled = tl.exp(s - m_prev_block)
        p_scaled = p_scaled * mask
        
        if m_prev_block[0, 0] != m_prev[0, 0]:
            alpha = tl.exp(m_prev - m_prev_block)
            o *= alpha[:, None]
        
        l_prev = l_prev * tl.exp(m_prev - m_prev_block) + tl.sum(p_scaled, axis=1, keep_dims=True)
        m_prev = m_prev_block
        
        for k_iter in range(0, BLOCK_N, BLOCK_K):
            k_start, k_end = k_iter, min(k_iter + BLOCK_K, BLOCK_N)
            v = V_desc.load([offset_row + col_start, k_iter])
            p = p_scaled[:, k_start:k_end]
            o = tl.dot(p, v[:, k_start:k_end], o)
            
    if m_prev[0, 0] > -1e19:
        o = o / l_prev
        
    lse = m_prev + tl.log(l_prev)
    
    store_mask = row_offsets < valid_rows
    O_desc.store([offset_row + row_start, 0], o, mask=store_mask[:, None])
    LSE_desc.store([offset_row + row_start, 0], lse, mask=store_mask[:, None])


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
    
    Q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_M, BLOCK_K])
    K_desc = TensorDescriptor.from_tensor(K, [BLOCK_N, BLOCK_K])
    V_desc = TensorDescriptor.from_tensor(V, [BLOCK_N, BLOCK_K])
    O_desc = TensorDescriptor.from_tensor(O, [BLOCK_M, BLOCK_K])
    LSE_desc = TensorDescriptor.from_tensor(LSE, [BLOCK_M, 1])
    
    num_programs_m = triton.cdiv(S, BLOCK_M)
    num_programs_bh = B * H
    grid = (num_programs_m, num_programs_bh)
    
    scale = 1.0 / math.sqrt(128)
    
    _mha_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE_desc,
        S, H, scale,
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4,
        num_stages=2,
    )