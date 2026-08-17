import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    seq_len,
):
    i = tl.program_id(1)
    batch_idx = tl.program_id(0)
    start_row = i * 128
    
    if start_row >= seq_len:
        return
    
    rows = tl.arange(0, 128)
    cols = tl.arange(0, 128)
    
    # Load Q: shape [128, 128]
    q_ptrs = Q_ptr + (batch_idx * seq_len + start_row + rows[:, None]) * 128 + cols[None, :]
    q_mask = (start_row + rows[:, None] < seq_len)
    Q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    Q = Q.to_float32()
    
    # Initialize stateful reductions
    O_acc = tl.zeros((128, 128), dtype=tl.float32)
    m = tl.full((128,), -float('inf'))
    l = tl.zeros((128,), dtype=tl.float32)
    
    scale = 1.0 / tl.sqrt(tl.float32(128))
    
    for j in range(0, i + 1):
        # Load K_j: shape [128, 128]
        k_ptrs = K_ptr + (batch_idx * seq_len + j * 128 + rows[:, None]) * 128 + cols[None, :]
        k_mask = (j * 128 + rows[:, None] < seq_len)
        K_j = tl.load(k_ptrs, mask=k_mask, other=0.0)
        K_j = K_j.to_float32()
        
        # Load V_j: shape [128, 128]
        v_ptrs = V_ptr + (batch_idx * seq_len + j * 128 + rows[:, None]) * 128 + cols[None, :]
        v_mask = (j * 128 + rows[:, None] < seq_len)
        V_j = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # Compute Attention Scores S = Q @ K_j^T / sqrt(D)
        S_acc = tl.dot(Q, K_j.T)
        S_acc = S_acc * scale
        
        # Causal Mask application
        global_row = (i * 128 + rows[:, None]).to_float32()
        global_col = (j * 128 + cols[None, :]).to_float32()
        mask = (global_col <= global_row) & (global_col < seq_len)
        S_acc = tl.where(mask, S_acc, -float('inf'))
        
        # Row Max Calculation
        m_old = m
        rowmax = tl.maximum(tl.full((1,), -float('inf')), tl.max(S_acc, axis=1))
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
    o_ptrs = O_ptr + (batch_idx * seq_len + start_row + rows[:, None]) * 128 + cols[None, :]
    tl.store(o_ptrs, O_acc, mask=(start_row + rows[:, None] < seq_len))
    
    # Store LSE
    lse_ptrs = LSE_ptr + batch_idx * seq_len + start_row + rows
    tl.store(lse_ptrs, m + tl.log(l), mask=(start_row + rows < seq_len))


def run(Q, K, V, O, LSE):
    """Compute Causal Multi-Head Attention forward pass and Log Sum Exp."""
    torch.cuda.set_device(Q.device)
    seq_len = Q.shape[2]
    
    num_batches = Q.shape[0] * Q.shape[1]
    num_tiles = triton.cdiv(seq_len, 128)
    
    grid = (num_batches, num_tiles)
    
    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        seq_len,
    )