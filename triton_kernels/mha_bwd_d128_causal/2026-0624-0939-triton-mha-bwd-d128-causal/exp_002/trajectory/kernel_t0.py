import torch
import triton
import triton.language as tl
import math


@triton.jit
def dv_dk_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dV_ptr, dK_ptr,
    S, H, B,
    stride_b, stride_h, stride_s,
    scale,
    BLOCK_S: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    j = tl.program_id(0)
    
    d = 128
    
    # Pre-calculate base offsets to avoid repeated arithmetic
    b_h_offset_Q = b_idx * stride_b + h_idx * stride_h
    b_h_offset_K = b_idx * stride_b + h_idx * stride_h
    b_h_offset_V = b_idx * stride_b + h_idx * stride_h
    b_h_offset_O = b_idx * stride_b + h_idx * stride_h
    b_h_offset_dO = b_idx * stride_b + h_idx * stride_h
    b_h_offset_L = b_idx * (H * S) + h_idx * S
    b_h_offset_dV = b_idx * stride_b + h_idx * stride_h
    b_h_offset_dK = b_idx * stride_b + h_idx * stride_h
    
    # Coordinate tiles
    rows = tl.arange(0, BLOCK_S)[:, None]  # [BLOCK_S, 1]
    cols = tl.arange(0, d)[None, :]         # [1, d]
    
    # Load K and V for the current block j
    k_idx_base = j * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_k = (k_idx_base[:, None] < S)
    
    k_base = K_ptr + b_h_offset_K + j * BLOCK_S * stride_s
    K = tl.load(k_base + rows * stride_s + cols, mask=mask_k, other=0.0)
    
    v_base = V_ptr + b_h_offset_V + j * BLOCK_S * stride_s
    V = tl.load(v_base + rows * stride_s + cols, mask=mask_k, other=0.0)
    
    dV_acc = tl.zeros((BLOCK_S, d), tl.float32)
    dK_acc = tl.zeros((BLOCK_S, d), tl.float32)
    
    num_blocks = triton.cdiv(S, BLOCK_S)
    
    # Iterate over all query blocks causally following or overlapping with key block j
    for i in range(j, num_blocks):
        q_idx_base = i * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_q = (q_idx_base[:, None] < S)
        
        q_base = Q_ptr + b_h_offset_Q + i * BLOCK_S * stride_s
        Q_i = tl.load(q_base + rows * stride_s + cols, mask=mask_q, other=0.0)
        
        do_base = dO_ptr + b_h_offset_dO + i * BLOCK_S * stride_s
        dO_i = tl.load(do_base + rows * stride_s + cols, mask=mask_q, other=0.0)
        
        o_base = O_ptr + b_h_offset_O + i * BLOCK_S * stride_s
        O_i = tl.load(o_base + rows * stride_s + cols, mask=mask_q, other=0.0)
        
        l_base = L_ptr + b_h_offset_L + i * BLOCK_S
        L_i = tl.load(l_base + tl.arange(0, BLOCK_S), mask=q_idx_base < S, other=0.0)
        
        # Forward logic inside backward loop to get P
        s = tl.dot(Q_i, K.T)
        
        q_idx_2d = q_idx_base[:, None]
        k_idx_2d = k_idx_base[None, :]
        causal_mask = (q_idx_2d >= k_idx_2d) & (q_idx_2d < S) & (k_idx_2d < S)
        
        p = tl.exp(s * scale - L_i[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        # Backward logic components
        D = tl.sum(O_i.to(tl.float32) * dO_i.to(tl.float32), axis=1)
        
        dp = tl.dot(dO_i, V.T)
        
        ds = p * (dp - D[:, None]) * scale
        
        dV_acc = tl.dot(p.T, dO_i.to(tl.float32), dV_acc)
        dK_acc = tl.dot(ds.T, Q_i.to(tl.float32), dK_acc)
    
    # Flush accumulated gradients
    dv_base = dV_ptr + b_h_offset_dV + j * BLOCK_S * stride_s
    tl.store(dv_base + rows * stride_s + cols, dV_acc.to(tl.bfloat16), mask=mask_k)
    
    dk_base = dK_ptr + b_h_offset_dK + j * BLOCK_S * stride_s
    tl.store(dk_base + rows * stride_s + cols, dK_acc.to(tl.bfloat16), mask=mask_k)


@triton.jit
def dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S, H, B,
    stride_b, stride_h, stride_s,
    scale,
    BLOCK_S: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    i = tl.program_id(0)
    
    d = 128
    
    # Pre-calculate base offsets
    b_h_offset_Q = b_idx * stride_b + h_idx * stride_h
    b_h_offset_K = b_idx * stride_b + h_idx * stride_h
    b_h_offset_V = b_idx * stride_b + h_idx * stride_h
    b_h_offset_O = b_idx * stride_b + h_idx * stride_h
    b_h_offset_dO = b_idx * stride_b + h_idx * stride_h
    b_h_offset_L = b_idx * (H * S) + h_idx * S
    b_h_offset_dQ = b_idx * stride_b + h_idx * stride_h
    
    rows = tl.arange(0, BLOCK_S)[:, None]  # [BLOCK_S, 1]
    cols = tl.arange(0, d)[None, :]         # [1, d]
    
    q_idx_base = i * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_q = (q_idx_base[:, None] < S)
    
    # Load invariant Query information
    q_base = Q_ptr + b_h_offset_Q + i * BLOCK_S * stride_s
    Q_i = tl.load(q_base + rows * stride_s + cols, mask=mask_q, other=0.0)
    
    do_base = dO_ptr + b_h_offset_dO + i * BLOCK_S * stride_s
    dO_i = tl.load(do_base + rows * stride_s + cols, mask=mask_q, other=0.0)
    
    o_base = O_ptr + b_h_offset_O + i * BLOCK_S * stride_s
    O_i = tl.load(o_base + rows * stride_s + cols, mask=mask_q, other=0.0)
    
    l_base = L_ptr + b_h_offset_L + i * BLOCK_S
    L_i = tl.load(l_base + tl.arange(0, BLOCK_S), mask=q_idx_base < S, other=0.0)
    
    D = tl.sum(O_i.to(tl.float32) * dO_i.to(tl.float32), axis=1)
    
    dQ_acc = tl.zeros((BLOCK_S, d), tl.float32)
    
    # Iterate over key blocks causally preceding or overlapping with query block i
    for j in range(0, i + 1):
        k_idx_base = j * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_k = (k_idx_base[:, None] < S)
        
        k_base = K_ptr + b_h_offset_K + j * BLOCK_S * stride_s
        K_j = tl.load(k_base + rows * stride_s + cols, mask=mask_k, other=0.0)
        
        v_base = V_ptr + b_h_offset_V + j * BLOCK_S * stride_s
        V_j = tl.load(v_base + rows * stride_s + cols, mask=mask_k, other=0.0)
        
        s = tl.dot(Q_i, K_j.T)
        
        q_idx_2d = q_idx_base[:, None]
        k_idx_2d = k_idx_base[None, :]
        causal_mask = (q_idx_2d >= k_idx_2d) & (q_idx_2d < S) & (k_idx_2d < S)
        
        p = tl.exp(s * scale - L_i[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        dp = tl.dot(dO_i, V_j.T)
        
        ds = p * (dp - D[:, None]) * scale
        
        dQ_acc = tl.dot(ds, K_j.to(tl.float32), dQ_acc)
    
    dq_base = dQ_ptr + b_h_offset_dQ + i * BLOCK_S * stride_s
    tl.store(dq_base + rows * stride_s + cols, dQ_acc.to(tl.bfloat16), mask=mask_q)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute causal multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(float(d))
    
    stride_b = H * S * d
    stride_h = S * d
    stride_s = d
    
    BLOCK_S = 64
    
    grid = (triton.cdiv(S, BLOCK_S), H, B)
    
    dv_dk_kernel[grid](
        Q, K, V, O, dO, L, dV, dK,
        S, H, B,
        stride_b, stride_h, stride_s,
        scale,
        BLOCK_S=BLOCK_S,
        num_warps=4,
    )
    
    dq_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        S, H, B,
        stride_b, stride_h, stride_s,
        scale,
        BLOCK_S=BLOCK_S,
        num_warps=4,
    )