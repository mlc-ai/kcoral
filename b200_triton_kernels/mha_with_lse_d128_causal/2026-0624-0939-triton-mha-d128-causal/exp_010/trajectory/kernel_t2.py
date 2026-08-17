import torch
import triton
import triton.language as tl


@triton.jit
def _mha_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    seq_len, scale,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(1)
    batch_idx = tl.program_id(0)
    start_row = i * BLOCK_SIZE
    
    if start_row >= seq_len:
        return
    
    rows = tl.arange(0, BLOCK_SIZE)
    cols = tl.arange(0, BLOCK_SIZE)
    
    batch_base = batch_idx * 48 * seq_len * 128  # Hardcoded B=4, H=48 -> B*H=192 stride 
    
    # Load Q: shape [BLOCK_SIZE, 128]
    q_ptrs = Q_ptr + batch_base + (start_row + rows[:, None]) * 128 + cols[None, :]
    q_mask = (start_row + rows[:, None] < seq_len)
    Q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    Q = Q.to(tl.float32)
    
    # Initialize stateful reductions
    O_acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    m = tl.full((BLOCK_SIZE,), -float('inf'))
    l = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate exclusively across contributing KV blocks resolving diagonal conflicts natively via Step 3 masking logic.
    for j in range(start_row // BLOCK_SIZE, i + 1):
        # Load K_j: shape [BLOCK_SIZE, 128]
        k_ptrs = K_ptr + batch_base + (j * BLOCK_SIZE + rows[:, None]) * 128 + cols[None, :]
        k_mask = (j * BLOCK_SIZE + rows[:, None] < seq_len)
        K_j = tl.load(k_ptrs, mask=k_mask, other=0.0)
        K_j = K_j.to(tl.float32)
        
        # Load V_j: shape [BLOCK_SIZE, 128]
        v_ptrs = V_ptr + batch_base + (j * BLOCK_SIZE + rows[:, None]) * 128 + cols[None, :]
        v_mask = (j * BLOCK_SIZE + rows[:, None] < seq_len)
        V_j = tl.load(v_ptrs, mask=v_mask, other=0.0)
        V_j = V_j.to(tl.float32)
        
        # Compute Attention Scores S = Q @ K_j^T / sqrt(D)
        S_acc = tl.dot(Q, K_j.T)
        S_acc = S_acc * scale
        
        # Causal Mask application
        global_row = (start_row + rows[:, None]).to(tl.float32)
        global_col = (j * BLOCK_SIZE + cols[None, :]).to(tl.float32)
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
    o_ptrs = O_ptr + batch_base + (start_row + rows[:, None]) * 128 + cols[None, :]
    tl.store(o_ptrs, O_acc, mask=(start_row + rows[:, None] < seq_len))
    
    # Store LSE
    lse_ptrs = LSE_ptr + batch_idx * seq_len + start_row + rows
    lse_val = m + tl.log(l)
    tl.store(lse_ptrs, lse_val, mask=(start_row + rows < seq_len))


def run(Q, K, V, O, LSE):
    """Compute Causal Multi-Head Attention forward pass and Log Sum Exp."""
    torch.cuda.set_device(Q.device)
    seq_len = Q.shape[2]
    
    num_batches = Q.shape[0] * Q.shape[1]
    num_tiles = triton.cdiv(seq_len, 128)
    
    grid = (num_batches, num_tiles)
    
    scale = 1.0 / torch.sqrt(torch.tensor(128, dtype=torch.float32))
    
    _mha_fwd_kernel[grid](
        Q, K, V, O, LSE,
        seq_len, scale,
        BLOCK_SIZE=128,
    )