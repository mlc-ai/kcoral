import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    LSE_ptr,
    S_len,
    B,
    H,
    D,
    STRIDE_LSE_BH: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Causal Multi-Head Attention with LogSumExp output."""
    
    seq_id = tl.program_id(0)  
    idx_bh = tl.program_id(1)  
    
    start_n = seq_id * BLOCK_M
    
    rows = tl.arange(0, BLOCK_M)
    
    # --- 1. Load Q tiles ---
    Q = Q_desc.load([idx_bh, start_n, 0])
    Q = tl.reshape(Q, (BLOCK_M, D))
    
    # --- 2. Initialize Sequential Accumulators ---
    O_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    sqrt_D = float(D ** 0.5)
    
    num_k_tiles = tl.cdiv(S_len, BLOCK_M)
    max_j = min(seq_id, num_k_tiles - 1)
    
    # --- 3. Main Causal Sequence Loop ---
    for j_block in range(0, max_j + 1):
        
        # Load K and V blocks
        K = K_desc.load([idx_bh, j_block * BLOCK_M, 0])
        K = tl.reshape(K, (BLOCK_M, D))
        
        V = V_desc.load([idx_bh, j_block * BLOCK_M, 0])
        V = tl.reshape(V, (BLOCK_M, D))
        
        # Compute attention scores 
        S_scores = tl.dot(Q, K.T) / sqrt_D
        
        # Apply lower-triangular causal mask natively inside the tile
        query_pos = start_n + rows[:, None]
        key_pos = j_block * BLOCK_M + rows[None, :]
        mask = (query_pos >= key_pos) & (query_pos < S_len) & (key_pos < S_len)
        S_scores = tl.where(mask, S_scores, -float('inf'))
        
        # Update sequential statistics (running max & scale factor)
        m_prev = m
        row_max_S = tl.max(S_scores, dim=1)
        m_new = tl.maximum(m_prev, row_max_S)
        
        diff = m_prev - m_new
        scale = tl.exp(diff)
        
        # Compute local attention weights distribution P
        P = tl.exp(S_scores - m_new)
        
        # Update reduction coefficients correctly tracking denominator magnitude
        l = l * scale + tl.sum(P, dim=1)
        m = m_new
        
        # Update outputs using prior expectations scaling
        O_acc = O_acc * scale[:, None]
        
        # Accumulate weighted context representations
        P_bf16 = P.to(tl.bfloat16)
        O_acc = tl.dot(P_bf16, V, O_acc)

    # --- 4. Final Normalization ---
    inv_l = 1.0 / l
    inv_l = tl.where(l > 0.0, inv_l, 0.0)
    
    O = (O_acc * inv_l[:, None]).to(tl.bfloat16)
    
    # --- 5. Persist Results ---
    O_reshaped = tl.reshape(O, (1, BLOCK_M, D))
    O_desc.store([idx_bh, start_n, 0], O_reshaped)
    
    # LogSumExp output mapping
    lse_ptr = LSE_ptr + idx_bh * STRIDE_LSE_BH + start_n + rows
    lse_val = m + tl.log(l)
    lse_val = tl.where(l > 0.0, lse_val, 0.0)
    store_mask_lse = (start_n + rows < S_len)
    tl.store(lse_ptr, lse_val, mask=store_mask_lse)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    BLOCK_M = 64
    
    Q_3d = Q.view(B * H, S_len, D)
    K_3d = K.view(B * H, S_len, D)
    V_3d = V.view(B * H, S_len, D)
    O_3d = O.view(B * H, S_len, D)
    
    Q_desc = TensorDescriptor.from_tensor(Q_3d, [1, BLOCK_M, D])
    K_desc = TensorDescriptor.from_tensor(K_3d, [1, BLOCK_M, D])
    V_desc = TensorDescriptor.from_tensor(V_3d, [1, BLOCK_M, D])
    O_desc = TensorDescriptor.from_tensor(O_3d, [1, BLOCK_M, D])
    
    grid = (triton.cdiv(S_len, BLOCK_M), B * H)
    
    STRIDE_LSE_BH = H * S_len
    
    _mha_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE, S_len, B, H, D, 
        STRIDE_LSE_BH, BLOCK_M,
        num_warps=8
    )