import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_desc,
    seq_len, scale,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(1)
    batch_idx = tl.program_id(0)
    start_row = i * BLOCK_SIZE
    
    if start_row >= seq_len:
        return
    
    rows = tl.arange(0, BLOCK_SIZE)
    
    q_offset = batch_idx * seq_len + start_row
    
    # Load Q: shape [BLOCK_SIZE, 128]
    Q = Q_desc.load([q_offset, 0])
    Q = Q.to(tl.float32)
    
    # Initialize stateful reductions
    O_acc = tl.zeros((BLOCK_SIZE, 128), dtype=tl.float32)
    m = tl.full((BLOCK_SIZE,), -float('inf'))
    l = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for j in range(0, i + 1):
        k_offset = batch_idx * seq_len + j * BLOCK_SIZE
        
        # Load K_j: shape [BLOCK_SIZE, 128]
        K_j = K_desc.load([k_offset, 0])
        K_j = K_j.to(tl.float32)
        
        # Load V_j: shape [BLOCK_SIZE, 128]
        V_j = V_desc.load([k_offset, 0])
        V_j = V_j.to(tl.float32)
        
        # Compute Attention Scores S = Q @ K_j^T / sqrt(D)
        S_acc = tl.dot(Q, K_j.T)
        S_acc = S_acc * scale
        
        # Causal Mask application
        global_row = (start_row + rows).to(tl.float32)
        global_col = (j * BLOCK_SIZE + rows).to(tl.float32)
        mask = (global_col[None, :] <= global_row[:, None]) & (global_col[None, :] < seq_len)
        S_acc = tl.where(mask, S_acc, -float('inf'))
        
        # Row Max Calculation
        m_old = m
        rowmax = tl.max(S_acc, axis=1)
        m = tl.maximum(m, rowmax)
        exp_old = tl.exp(m_old - m)
        
        # Exp Calculation
        P = tl.exp(S_acc - m[:, None])
        l = l * exp_old + tl.sum(P, axis=1)
        
        # Scale prior accumulated output
        O_acc *= exp_old[:, None]
        
        # Compute Output O_acc += P @ V_j
        O_acc = tl.dot(P, V_j, acc=O_acc)
        
    # Normalize Output
    inv_l = 1.0 / l
    O_acc *= inv_l[:, None]
    
    # Store O
    O_desc.store([q_offset, 0], O_acc)
    
    # Store LSE
    lse_offset = batch_idx * seq_len + start_row
    lse_val = m + tl.log(l)
    LSE_desc.store([lse_offset], lse_val)


def run(Q, K, V, O, LSE):
    """Compute Causal Multi-Head Attention forward pass and Log Sum Exp."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    seq_len = S
    
    # Flatten into 2D matrices to easily track batch/head offsets numerically.
    Q_2d = Q.reshape(B * H * S, D)
    K_2d = K.reshape(B * H * S, D)
    V_2d = V.reshape(B * H * S, D)
    O_2d = O.reshape(B * H * S, D)
    LSE_1d = LSE.reshape(B * H * S)
    
    BLOCK_M = 128
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_M, D])
    V_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_M, D])
    O_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK_M, D])
    LSE_desc = TensorDescriptor.from_tensor(LSE_1d, [BLOCK_M])
    
    num_batches = B * H
    num_tiles = triton.cdiv(seq_len, BLOCK_M)
    
    grid = (num_batches, num_tiles)
    
    scale = 1.0 / (128.0 ** 0.5)
    
    _mha_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE_desc,
        seq_len, scale,
        BLOCK_SIZE=BLOCK_M,
        num_warps=4,
        num_stages=3,
    )