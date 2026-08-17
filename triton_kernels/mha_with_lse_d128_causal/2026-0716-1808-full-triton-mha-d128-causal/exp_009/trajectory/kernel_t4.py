import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _mha_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_desc,
    S, H, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    head = pid_bh % H
    batch = pid_bh // H
    
    row_start = pid_m * BLOCK_M
    
    valid_rows = min(BLOCK_M, max(0, S - row_start))
    if valid_rows <= 0:
        return
    
    block_idx_max = (row_start + BLOCK_M - 1) // BLOCK_N
    total_blocks = min(block_idx_max + 1, (S + BLOCK_N - 1) // BLOCK_N)
    
    q0 = Q_desc.load([batch, head, row_start, 0])
    q1 = Q_desc.load([batch, head, row_start, 64])
    
    o0 = tl.zeros((BLOCK_M, 64), tl.float32)
    o1 = tl.zeros((BLOCK_M, 64), tl.float32)
    m_prev = tl.full((BLOCK_M,), -1e20, tl.float32)
    l_prev = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    row_offsets = tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)
    
    for block_idx in range(total_blocks):
        col_start = block_idx * BLOCK_N
        
        k0 = K_desc.load([batch, head, col_start, 0])
        k1 = K_desc.load([batch, head, col_start, 64])
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        
        s += tl.dot(q0, k0.T)
        s += tl.dot(q1, k1.T)
        
        s = s * scale
        
        mask = ((col_start + col_offsets[None, :]) <= (row_start + row_offsets[:, None]))
        
        s = tl.where(mask, s, -1e20)
        
        m_curr = tl.maximum(m_prev, tl.max(s, axis=-1))
        alpha = tl.exp(m_prev - m_curr)
        o0 = o0 * alpha[:, None]
        o1 = o1 * alpha[:, None]
        
        exp_s = tl.exp(s - m_curr)
        p = tl.where(mask, exp_s, 0.0)
        
        l_prev = l_prev * alpha + tl.sum(p, axis=-1)
        m_prev = m_curr
        
        v_tile = V_desc.load([batch, head, col_start, 0])
        
        p0 = p[:, :64]
        p1 = p[:, 64:]
        
        v0 = v_tile[:, :64]
        v1 = v_tile[:, 64:]
        
        o0 += tl.dot(p0, v0)
        o1 += tl.dot(p1, v1)
            
    o0 = o0 / l_prev[:, None]
    o1 = o1 / l_prev[:, None]
    
    lse = m_prev + tl.log(l_prev)
    
    O_desc.store([batch, head, row_start, 0], o0.to(tl.bfloat16))
    O_desc.store([batch, head, row_start, 64], o1.to(tl.bfloat16))
    LSE_desc.store([batch, head, row_start], lse.to(tl.bfloat16))


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
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, 128, 64])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, 128, 64])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, 64, 128])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, 128, 64])
    LSE_desc = TensorDescriptor.from_tensor(LSE, [1, 1, 128])
    
    num_programs_m = triton.cdiv(S, BLOCK_M)
    num_programs_bh = B * H
    grid = (num_programs_m, num_programs_bh)
    
    scale = 1.0 / math.sqrt(128)
    
    _mha_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE_desc,
        S, H, scale,
        BLOCK_M, BLOCK_N,
        num_warps=4,
        num_stages=2,
    )