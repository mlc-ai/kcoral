import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    S_len,
    B,
    H,
    D,
    stride_b: tl.constexpr,
    stride_h: tl.constexpr,
    stride_s: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Causal Multi-Head Attention with LogSumExp output."""
    
    pid_n = tl.program_id(0)  
    idx_bh = tl.program_id(1)
    
    b = idx_bh // H
    h = idx_bh % H
    
    start_n = pid_n * BLOCK_N
    
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    cols_128 = tl.arange(0, 128)
    
    # --- 1. Load Q tiles ---
    q_row_idx = start_n + rows
    ptr_Q = Q_ptr + b * stride_b + h * stride_h + q_row_idx[:, None] * D + cols_128[None, :]
    
    Q = tl.load(ptr_Q, mask=(q_row_idx < S_len)[:, None], other=0.0)
    
    # --- 2. Initialize Sequential Accumulators ---
    O_acc = tl.zeros((BLOCK_N, 128), dtype=tl.float32)
    m = tl.full((BLOCK_N,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    sqrt_D = float(D ** 0.5)
    
    # --- 3. Main Causal Sequence Loop ---
    for j_block in range(0, pid_n + 1):
        
        k_row_idx = j_block * BLOCK_N + rows
        load_mask_K = (k_row_idx < S_len)[:, None]
        
        # Load K and V blocks
        ptr_K = K_ptr + b * stride_b + h * stride_h + k_row_idx[:, None] * D + cols_128[None, :]
        ptr_V = V_ptr + b * stride_b + h * stride_h + k_row_idx[:, None] * D + cols_128[None, :]
        
        K = tl.load(ptr_K, mask=load_mask_K, other=0.0)
        V = tl.load(ptr_V, mask=load_mask_K, other=0.0)
        
        # Compute attention scores 
        S_scores = tl.dot(Q, K.T) / sqrt_D
        
        # Apply lower-triangular causal mask natively inside the tile
        query_pos = start_n + rows[:, None]
        key_pos = j_block * BLOCK_N + cols[None, :]
        mask = (query_pos >= key_pos) & (query_pos < S_len) & (key_pos < S_len)
        S_scores = tl.where(mask, S_scores, -float('inf'))
        
        # Update sequential statistics (running max & scale factor)
        m_prev = m
        row_max_S = tl.max(S_scores, dim=1)
        m = tl.maximum(m_prev, row_max_S)
        scale = tl.exp(m_prev - m)
        
        # Compute local attention weights distribution P
        P = tl.exp(S_scores - m)
        l = l * scale + tl.sum(P, dim=1)
        
        # Update outputs using prior expectations scaling
        O_acc = O_acc * scale[:, None]
        
        # Accumulate weighted context representations
        P_bf16 = P.to(tl.bfloat16)
        O_acc = tl.dot(P_bf16, V, O_acc)

    # --- 4. Final Normalization ---
    inv_l = 1.0 / l
    O = (O_acc * inv_l[:, None]).to(tl.bfloat16)
    
    # --- 5. Persist Results ---
    ptr_O = O_ptr + b * stride_b + h * stride_h + q_row_idx[:, None] * D + cols_128[None, :]
    tl.store(ptr_O, O, mask=(q_row_idx < S_len)[:, None])
    
    # LogSumExp output mapping
    lse_ptr = LSE_ptr + b * H * S_len + h * S_len + start_n + rows
    lse_val = m + tl.log(l)
    store_mask_lse = (start_n + rows < S_len)
    tl.store(lse_ptr, lse_val, mask=store_mask_lse)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    BLOCK_N = 64
    
    grid = (triton.cdiv(S_len, BLOCK_N), B * H)
    _mha_kernel[grid](
        Q, K, V, O, LSE, S_len, B, H, D, 
        H * S_len * D, S_len * D, D, 
        BLOCK_N,
        num_warps=8
    )