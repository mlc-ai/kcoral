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
    
    row_start = pid_m * BLOCK_M
    
    valid_rows = min(BLOCK_M, S - row_start)
    if valid_rows <= 0:
        return
    
    causal_block_max = (row_start + BLOCK_M - 1) // BLOCK_N + 1
    
    q = Q_desc.load([batch, head, row_start, 0])
    
    o = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    m_prev = tl.full((BLOCK_M,), -1e20, tl.float32)
    l_prev = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    row_offsets = tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)
    
    k = K_desc.load([batch, head, 0, 0])
    
    for block_idx in range(min(BLOCK_M, causal_block_max)):
        if block_idx + 1 < causal_block_max:
            next_col_start = (block_idx + 1) * BLOCK_N
            k_next = K_desc.load([batch, head, next_col_start, 0])
        
        col_start = block_idx * BLOCK_N
        
        s = tl.dot(q, k.T) * scale
        
        mask = ((col_start + col_offsets[None, :]) <= (row_start + row_offsets[:, None]))
        s = s * mask
        
        m_curr = tl.maximum(m_prev, tl.max(s, axis=1))
        alpha = tl.exp(m_prev - m_curr)
        o *= alpha[:, None]
        
        p = tl.exp(s - m_curr) * mask
        l_prev = l_prev * alpha + tl.sum(p, axis=1)
        m_prev = m_curr
        
        v = V_desc.load([batch, head, col_start, 0])
        o = tl.dot(p, v.T, o)
            
        if block_idx + 1 < causal_block_max:
            k = k_next
            
    if m_prev[0] > -1e19:
        o = o / l_prev[:, None]
    
    lse = m_prev + tl.log(l_prev)
    
    O_desc.store([batch, head, row_start, 0], o)
    LSE_desc.store([batch, head, row_start], lse)


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
    BLOCK_K = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, BLOCK_K], padding_option="zero")
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, BLOCK_K], padding_option="zero")
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, BLOCK_K], padding_option="zero")
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, BLOCK_K], padding_option="zero")
    LSE_desc = TensorDescriptor.from_tensor(LSE, [1, 1, BLOCK_M], padding_option="zero")
    
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