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
    S,
    H,
    B,
    num_warps=4,
):
    """Causal Multi-Head Attention with LogSumExp output."""
    
    # Identify this program's position in the grid
    pid_n = tl.program_id(0)  # block along sequence
    h = tl.program_id(1)      # head index
    b = tl.program_id(2)      # batch index
    
    start_n = pid_n * 64
    
    # --- 1. Load Q tiles ---
    row_idx = b * H * S + h * S + start_n + tl.arange(0, 64)
    
    ptr_Q0 = Q_ptr + row_idx[:, None] * 128 + tl.arange(0, 64)[None, :]
    load_mask_Q = (start_n + tl.arange(0, 64) < S)[:, None]
    Q0 = tl.load(ptr_Q0, mask=load_mask_Q, other=0.0)
    
    ptr_Q1 = Q_ptr + row_idx[:, None] * 128 + (64 + tl.arange(0, 64))[None, :]
    Q1 = tl.load(ptr_Q1, mask=load_mask_Q, other=0.0)
    
    # --- 2. Initialize Sequential Accumulators ---
    O0_acc = tl.zeros((64, 64), dtype=tl.float32)
    O1_acc = tl.zeros((64, 64), dtype=tl.float32)
    m = tl.full((64,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((64,), dtype=tl.float32)
    
    # Precomputed constants
    sqrt_D = 11.31370849898476  # sqrt(128)
    
    # --- 3. Main Causal Sequence Loop ---
    for j_block in range(0, start_n // 64 + 1):
        
        # Load K block
        k_row_idx = b * H * S + h * S + j_block * 64 + tl.arange(0, 64)
        load_mask_K = (j_block * 64 + tl.arange(0, 64) < S)[:, None]
        
        ptr_K0 = K_ptr + k_row_idx[:, None] * 128 + tl.arange(0, 64)[None, :]
        K0 = tl.load(ptr_K0, mask=load_mask_K, other=0.0)
        
        ptr_K1 = K_ptr + k_row_idx[:, None] * 128 + (64 + tl.arange(0, 64))[None, :]
        K1 = tl.load(ptr_K1, mask=load_mask_K, other=0.0)
        
        # Compute attention scores (accumulate directly into 'S')
        S = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        S = S / sqrt_D
        
        # Apply lower-triangular causal mask natively inside the tile
        query_pos = start_n + tl.arange(0, 64)[:, None]
        key_pos = j_block * 64 + tl.arange(0, 64)[None, :]
        mask = (query_pos >= key_pos) & (query_pos < S) & (key_pos < S)
        S = tl.where(mask, S, -float('inf'))
        
        # Update sequential statistics (running max & scale factor)
        m_prev = m
        m = tl.max(tl.concat([m_prev[:, None], S], dim=1), dim=1).to(tl.float32)
        scale = tl.exp(m_prev - m)
        
        # Compute local attention weights distribution P
        P = tl.exp(S - m)
        l = l * scale + tl.sum(P, dim=1)
        
        # Update outputs using corrected prior expectations
        O0_acc = O0_acc * scale[:, None]
        O1_acc = O1_acc * scale[:, None]
        
        # Load corresponding V block
        load_mask_V = load_mask_K
        ptr_V0 = V_ptr + k_row_idx[:, None] * 128 + tl.arange(0, 64)[None, :]
        V0 = tl.load(ptr_V0, mask=load_mask_V, other=0.0)
        
        ptr_V1 = V_ptr + k_row_idx[:, None] * 128 + (64 + tl.arange(0, 64))[None, :]
        V1 = tl.load(ptr_V1, mask=load_mask_V, other=0.0)
        
        # Accumulate weighted context representations
        O0_acc = tl.dot(P.to(tl.bfloat16), V0, O0_acc)
        O1_acc = tl.dot(P.to(tl.bfloat16), V1, O1_acc)

    # --- 4. Final Normalization ---
    inv_l = 1.0 / l
    O0 = O0_acc * inv_l[:, None]
    O1 = O1_acc * inv_l[:, None]
    
    # Convert back to input format for memory commit
    O0 = O0.to(tl.bfloat16)
    O1 = O1.to(tl.bfloat16)
    
    # --- 5. Persist Results ---
    store_mask = (start_n + tl.arange(0, 64) < S)[:, None]
    
    out_ptr0 = O_ptr + row_idx[:, None] * 128 + tl.arange(0, 64)[None, :]
    tl.store(out_ptr0, O0, mask=store_mask)
    
    out_ptr1 = O_ptr + row_idx[:, None] * 128 + (64 + tl.arange(0, 64))[None, :]
    tl.store(out_ptr1, O1, mask=store_mask)
    
    # LogSumExp output mapping
    lse_ptr = LSE_ptr + b * H * S + h * S + start_n + tl.arange(0, 64)
    lse_val = m + tl.log(l)
    store_mask_lse = (start_n + tl.arange(0, 64) < S)
    tl.store(lse_ptr, lse_val, mask=store_mask_lse)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    grid = (S // 64, H, B)
    _mha_kernel[grid](Q, K, V, O, LSE, S, H, B)