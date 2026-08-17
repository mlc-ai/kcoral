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
    BLOCK_M: tl.constexpr,
):
    """Causal Multi-Head Attention with LogSumExp output."""
    
    pid_n = tl.program_id(0)  
    idx_bh = tl.program_id(1)
    
    B_val = int(B)
    H_val = int(H)
    b = idx_bh // H_val
    h = idx_bh % H_val
    
    start_n = pid_n * BLOCK_M
    
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_M)
    
    # --- 1. Load Q tiles ---
    q_row_idx = start_n + rows
    ptr_Q0 = Q_ptr + b * stride_b + h * stride_h + q_row_idx[:, None] * stride_s + cols[None, :]
    ptr_Q1 = Q_ptr + b * stride_b + h * stride_h + q_row_idx[:, None] * stride_s + (64 + cols)[None, :]
    
    Q0 = tl.load(ptr_Q0, mask=(q_row_idx < S_len)[:, None], other=0.0)
    Q1 = tl.load(ptr_Q1, mask=(q_row_idx < S_len)[:, None], other=0.0)
    
    # --- 2. Initialize Sequential Accumulators ---
    O0_acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
    O1_acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    sqrt_D = float(D ** 0.5)
    
    # --- 3. Main Causal Sequence Loop ---
    for j_block in range(0, pid_n + 1):
        
        k_row_idx = j_block * BLOCK_M + rows
        load_mask_K = (k_row_idx < S_len)[:, None]
        
        # Load K and V blocks
        ptr_K0 = K_ptr + b * stride_b + h * stride_h + k_row_idx[:, None] * stride_s + cols[None, :]
        ptr_K1 = K_ptr + b * stride_b + h * stride_h + k_row_idx[:, None] * stride_s + (64 + cols)[None, :]
        K0 = tl.load(ptr_K0, mask=load_mask_K, other=0.0)
        K1 = tl.load(ptr_K1, mask=load_mask_K, other=0.0)
        
        ptr_V0 = V_ptr + b * stride_b + h * stride_h + k_row_idx[:, None] * stride_s + cols[None, :]
        ptr_V1 = V_ptr + b * stride_b + h * stride_h + k_row_idx[:, None] * stride_s + (64 + cols)[None, :]
        V0 = tl.load(ptr_V0, mask=load_mask_K, other=0.0)
        V1 = tl.load(ptr_V1, mask=load_mask_K, other=0.0)
        
        # Compute attention scores 
        S_scores = (tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)) / sqrt_D
        
        # Apply lower-triangular causal mask natively inside the tile
        query_pos = start_n + rows[:, None]
        key_pos = j_block * BLOCK_M + cols[None, :]
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
        O0_acc = O0_acc * scale[:, None]
        O1_acc = O1_acc * scale[:, None]
        
        # Accumulate weighted context representations
        P_bf16 = P.to(tl.bfloat16)
        O0_acc = tl.dot(P_bf16, V0, O0_acc)
        O1_acc = tl.dot(P_bf16, V1, O1_acc)

    # --- 4. Final Normalization ---
    inv_l = 1.0 / l
    O0 = (O0_acc * inv_l[:, None]).to(tl.bfloat16)
    O1 = (O1_acc * inv_l[:, None]).to(tl.bfloat16)
    
    # --- 5. Persist Results ---
    ptr_O0 = O_ptr + b * stride_b + h * stride_h + q_row_idx[:, None] * stride_s + cols[None, :]
    ptr_O1 = O_ptr + b * stride_b + h * stride_h + q_row_idx[:, None] * stride_s + (64 + cols)[None, :]
    
    tl.store(ptr_O0, O0, mask=(q_row_idx < S_len)[:, None])
    tl.store(ptr_O1, O1, mask=(q_row_idx < S_len)[:, None])
    
    # LogSumExp output mapping
    lse_ptr = LSE_ptr + b * H_val * S_len + h * S_len + start_n + rows
    lse_val = m + tl.log(l)
    store_mask_lse = (start_n + rows < S_len)
    tl.store(lse_ptr, lse_val, mask=store_mask_lse)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    BLOCK_M = 64
    
    grid = (triton.cdiv(S_len, BLOCK_M), B * H)
    _mha_kernel[grid](
        Q, K, V, O, LSE, S_len, B, H, D, 
        H * S_len * D, S_len * D, D, 
        BLOCK_M,
        num_warps=8
    )