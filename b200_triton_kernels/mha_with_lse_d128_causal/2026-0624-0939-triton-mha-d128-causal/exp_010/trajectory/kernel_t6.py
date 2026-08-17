import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_desc,
    seq_len, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_head_idx = tl.program_id(0)
    i = tl.program_id(1)
    start_row = i * BLOCK_M
    
    if start_row >= seq_len:
        return
    
    rows_m = tl.arange(0, BLOCK_M)
    rows_n = tl.arange(0, BLOCK_N)
    
    q_offset = batch_head_idx * seq_len + start_row
    
    Q = Q_desc.load([q_offset, 0])
    Q = Q.to(tl.float32)
    
    O_acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for j in range(0, i + 1):
        k_offset = batch_head_idx * seq_len + j * BLOCK_N
        
        K_j = K_desc.load([k_offset, 0])
        K_j = K_j.to(tl.float32)
        
        V_j = V_desc.load([k_offset, 0])
        V_j = V_j.to(tl.float32)
        
        S_acc = tl.dot(Q, K_j.T)
        S_acc = S_acc * scale
        
        global_row = start_row + rows_m
        global_col = j * BLOCK_N + rows_n
        mask = (global_col[None, :] <= global_row[:, None]) & (global_col[None, :] < seq_len)
        S_acc = tl.where(mask, S_acc, -float('inf'))
        
        m_old = m
        rowmax = tl.max(S_acc, axis=1)
        m = tl.maximum(m, rowmax)
        exp_old = tl.exp(m_old - m)
        
        P = tl.exp(S_acc - m[:, None])
        l = l * exp_old + tl.sum(P, axis=1)
        
        O_acc = O_acc * exp_old[:, None] + tl.dot(P, V_j)
        
    valid = m > -float('inf')
    inv_l = 1.0 / l
    inv_l = tl.where(valid, inv_l, 0.0)
    O_acc *= inv_l[:, None]
    
    O_desc.store([q_offset, 0], O_acc.to(tl.bfloat16))
    
    lse_offset = batch_head_idx * seq_len + start_row
    lse_val = m + tl.log(l)
    lse_val = tl.where(valid, lse_val, 0.0)
    LSE_desc.store([lse_offset], lse_val)


def run(Q, K, V, O, LSE):
    """Compute Causal Multi-Head Attention forward pass and Log Sum Exp."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    seq_len = S
    
    Q_2d = Q.reshape(B * H * S, D)
    K_2d = K.reshape(B * H * S, D)
    V_2d = V.reshape(B * H * S, D)
    O_2d = O.reshape(B * H * S, D)
    LSE_1d = LSE.reshape(B * H * S)
    
    BLOCK_M = 128
    BLOCK_N = 64
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_N, D])
    V_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_N, D])
    O_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK_M, D])
    LSE_desc = TensorDescriptor.from_tensor(LSE_1d, [BLOCK_M])
    
    num_batches = B * H
    num_tiles = triton.cdiv(seq_len, BLOCK_M)
    
    grid = (num_batches, num_tiles)
    
    scale = 1.0 / (D ** 0.5)
    
    _mha_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE_desc,
        seq_len, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )