import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S_len, H, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Configuration for Head Dimension 128
    D = 128
    HALF_D = 64
    
    # Determine work coordinates from program IDs
    pid_m = tl.program_id(0)
    bh_idx = tl.program_id(1)
    b = bh_idx // H
    h = bh_idx % H
    
    start_m = pid_m * BLOCK_M
    base_offset = (b * H + h) * S_len * D
    
    # Pre-generate sequential index tensors for block construction
    q_idx = tl.arange(0, BLOCK_M)
    d_idx_0 = tl.arange(0, HALF_D)
    d_idx_1 = tl.arange(0, HALF_D)
    k_idx = tl.arange(0, BLOCK_N)
    
    # --- Pass 1 & 2 combined: FlashAttention ---
    # Load Q tile [BLOCK_M, 128] (split into two 64-wide feature blocks)
    q_off_0 = base_offset + (start_m + q_idx[:, None]) * D + d_idx_0[None, :]
    q_off_1 = base_offset + (start_m + q_idx[:, None]) * D + d_idx_1[None, :] + HALF_D
    mask_m = (start_m + q_idx[:, None]) < S_len
    
    Q_0 = tl.load(Q_ptr + q_off_0, mask=mask_m, other=0.0)
    Q_1 = tl.load(Q_ptr + q_off_1, mask=mask_m, other=0.0)
    
    Q_0_fp32 = Q_0.to(tl.float32)
    Q_1_fp32 = Q_1.to(tl.float32)
    
    m_local = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    s_local = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    acc_0 = tl.zeros((BLOCK_M, HALF_D), dtype=tl.float32)
    acc_1 = tl.zeros((BLOCK_M, HALF_D), dtype=tl.float32)
    
    num_k_tiles = tl.cdiv(S_len, BLOCK_N)
    
    for k_tile in range(num_k_tiles):
        start_k = k_tile * BLOCK_N
        
        # Load K and V tiles
        k_off_0 = base_offset + (start_k + k_idx[:, None]) * D + d_idx_0[None, :]
        k_off_1 = base_offset + (start_k + k_idx[:, None]) * D + d_idx_1[None, :] + HALF_D
        v_off_0 = base_offset + (start_k + k_idx[:, None]) * D + d_idx_0[None, :]
        v_off_1 = base_offset + (start_k + k_idx[:, None]) * D + d_idx_1[None, :] + HALF_D
        
        mask_k = (start_k + k_idx[:, None]) < S_len
        
        K_0 = tl.load(K_ptr + k_off_0, mask=mask_k, other=0.0)
        K_1 = tl.load(K_ptr + k_off_1, mask=mask_k, other=0.0)
        V_0 = tl.load(V_ptr + v_off_0, mask=mask_k, other=0.0)
        V_1 = tl.load(V_ptr + v_off_1, mask=mask_k, other=0.0)
        
        K_0_fp32 = K_0.to(tl.float32)
        K_1_fp32 = K_1.to(tl.float32)
        V_0_fp32 = V_0.to(tl.float32)
        V_1_fp32 = V_1.to(tl.float32)
        
        # Compute scaled attention scores
        S_qk = tl.dot(Q_0_fp32, K_0_fp32.T) * scale + \
               tl.dot(Q_1_fp32, K_1_fp32.T) * scale
        
        # Apply sequence boundary mask safely without introducing NaNs to reductions
        valid_k = (start_k + k_idx) < S_len
        S_qk = tl.where(valid_k[None, :], S_qk, -float('inf'))
        
        # Update local maximum
        m_curr = tl.max(S_qk, axis=1)
        m_new = tl.maximum(m_local, m_curr)
        
        # Compute attention probabilities and update sum
        p = tl.exp(S_qk - m_new[:, None])
        s_curr = tl.sum(p, axis=1)
        
        # Update normalization constant and scale existing accumulators
        exp_val = tl.exp(m_local - m_new)
        s_local = s_local * exp_val + s_curr
        
        acc_0 = acc_0 * exp_val[:, None] + tl.dot(p, V_0_fp32)
        acc_1 = acc_1 * exp_val[:, None] + tl.dot(p, V_1_fp32)
        
        m_local = m_new
    
    # --- Epilogue ---
    d = 1.0 / s_local
    O_0 = (acc_0 * d[:, None]).to(tl.bfloat16)
    O_1 = (acc_1 * d[:, None]).to(tl.bfloat16)
    
    o_off_0 = base_offset + (start_m + q_idx[:, None]) * D + d_idx_0[None, :]
    o_off_1 = base_offset + (start_m + q_idx[:, None]) * D + d_idx_1[None, :] + HALF_D
    
    valid_m = (start_m + q_idx) < S_len
    tl.store(O_ptr + o_off_0, O_0, mask=valid_m[:, None])
    tl.store(O_ptr + o_off_1, O_1, mask=valid_m[:, None])
    
    # Compute and store Log Sum Exp
    lse = m_local + tl.log(s_local)
    lse_off = b * H * S_len + h * S_len + start_m + q_idx
    tl.store(LSE_ptr + lse_off, lse, mask=valid_m)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    grid = (triton.cdiv(S_len, BLOCK_M), B * H)
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        S_len, H, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3,
    )