import torch
import triton
import triton.language as tl
import math


@triton.jit
def dv_dk_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dV_ptr, dK_ptr,
    S, H, B,
    stride_b, stride_h, stride_s, stride_l_b, stride_l_h,
    scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    j = tl.program_id(0)
    
    b_h_offset = b_idx * stride_b + h_idx * stride_h
    b_h_offset_L = b_idx * stride_l_b + h_idx * stride_l_h
    
    rows = tl.arange(0, BLOCK_S)
    cols_d0 = tl.arange(0, BLOCK_D)
    cols_d1 = tl.arange(0, BLOCK_D)
    
    k_idx_base = j * BLOCK_S + rows
    mask_k = k_idx_base[:, None] < S
    
    base_j = K_ptr + b_h_offset + j * BLOCK_S * stride_s
    ptrs_k0 = base_j + rows[:, None] * stride_s + cols_d0[None, :]
    ptrs_k1 = base_j + rows[:, None] * stride_s + cols_d1[None, :] + BLOCK_D
    
    K_j0 = tl.load(ptrs_k0, mask=mask_k, other=0.0)
    K_j1 = tl.load(ptrs_k1, mask=mask_k, other=0.0)
    
    base_v = V_ptr + b_h_offset + j * BLOCK_S * stride_s
    V_j0 = tl.load(base_v + rows[:, None] * stride_s + cols_d0[None, :], mask=mask_k, other=0.0)
    V_j1 = tl.load(base_v + rows[:, None] * stride_s + cols_d1[None, :] + BLOCK_D, mask=mask_k, other=0.0)
    
    dV_acc0 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    dV_acc1 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    dK_acc0 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    dK_acc1 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    
    num_blocks = tl.cdiv(S, BLOCK_S)
    
    for i in range(j, num_blocks):
        q_idx_base = i * BLOCK_S + rows
        mask_q = q_idx_base[:, None] < S
        
        base_i = Q_ptr + b_h_offset + i * BLOCK_S * stride_s
        ptrs_q0 = base_i + rows[:, None] * stride_s + cols_d0[None, :]
        ptrs_q1 = base_i + rows[:, None] * stride_s + cols_d1[None, :] + BLOCK_D
        
        Q_i0 = tl.load(ptrs_q0, mask=mask_q, other=0.0)
        Q_i1 = tl.load(ptrs_q1, mask=mask_q, other=0.0)
        
        base_o = O_ptr + b_h_offset + i * BLOCK_S * stride_s
        O_i0 = tl.load(base_o + rows[:, None] * stride_s + cols_d0[None, :], mask=mask_q, other=0.0)
        O_i1 = tl.load(base_o + rows[:, None] * stride_s + cols_d1[None, :] + BLOCK_D, mask=mask_q, other=0.0)
        
        base_do = dO_ptr + b_h_offset + i * BLOCK_S * stride_s
        dO_i0 = tl.load(base_do + rows[:, None] * stride_s + cols_d0[None, :], mask=mask_q, other=0.0)
        dO_i1 = tl.load(base_do + rows[:, None] * stride_s + cols_d1[None, :] + BLOCK_D, mask=mask_q, other=0.0)
        
        l_base = L_ptr + b_h_offset_L + i * BLOCK_S
        L_i = tl.load(l_base + rows, mask=q_idx_base < S, other=0.0)
        
        s = tl.dot(Q_i0, K_j0.T) + tl.dot(Q_i1, K_j1.T)
        
        q_idx_2d = q_idx_base[:, None]
        k_idx_2d = k_idx_base[None, :]
        causal_mask = (q_idx_2d >= k_idx_2d) & (q_idx_2d < S) & (k_idx_2d < S)
        
        p = tl.exp(s * scale - L_i[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        D = tl.sum(O_i0.to(tl.float32) * dO_i0.to(tl.float32), axis=1) + \
            tl.sum(O_i1.to(tl.float32) * dO_i1.to(tl.float32), axis=1)
        
        dp = tl.dot(dO_i0.to(tl.float32), V_j0.T.to(tl.float32)) + \
             tl.dot(dO_i1.to(tl.float32), V_j1.T.to(tl.float32))
        
        ds = p * (dp - D[:, None]) * scale
        
        dV_acc0 = tl.dot(p.T, dO_i0.to(tl.float32), dV_acc0)
        dV_acc1 = tl.dot(p.T, dO_i1.to(tl.float32), dV_acc1)
        dK_acc0 = tl.dot(ds.T, Q_i0.to(tl.float32), dK_acc0)
        dK_acc1 = tl.dot(ds.T, Q_i1.to(tl.float32), dK_acc1)
    
    dv_base0 = dV_ptr + b_h_offset + j * BLOCK_S * stride_s
    dv_base1 = dV_ptr + b_h_offset + j * BLOCK_S * stride_s + BLOCK_D
    dk_base0 = dK_ptr + b_h_offset + j * BLOCK_S * stride_s
    dk_base1 = dK_ptr + b_h_offset + j * BLOCK_S * stride_s + BLOCK_D
    
    mask_store = (k_idx_base[:, None] < S) & (cols_d0[None, :] < 128)
    
    tl.store(dv_base0 + rows[:, None] * stride_s + cols_d0[None, :], dV_acc0.to(tl.bfloat16), mask=mask_store)
    tl.store(dv_base1 + rows[:, None] * stride_s + cols_d1[None, :], dV_acc1.to(tl.bfloat16), mask=mask_store)
    tl.store(dk_base0 + rows[:, None] * stride_s + cols_d0[None, :], dK_acc0.to(tl.bfloat16), mask=mask_store)
    tl.store(dk_base1 + rows[:, None] * stride_s + cols_d1[None, :], dK_acc1.to(tl.bfloat16), mask=mask_store)


@triton.jit
def dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S, H, B,
    stride_b, stride_h, stride_s, stride_l_b, stride_l_h,
    scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    i = tl.program_id(0)
    
    b_h_offset = b_idx * stride_b + h_idx * stride_h
    b_h_offset_L = b_idx * stride_l_b + h_idx * stride_l_h
    
    rows = tl.arange(0, BLOCK_S)
    cols_d0 = tl.arange(0, BLOCK_D)
    cols_d1 = tl.arange(0, BLOCK_D)
    
    q_idx_base = i * BLOCK_S + rows
    mask_q = q_idx_base[:, None] < S
    
    base_i = Q_ptr + b_h_offset + i * BLOCK_S * stride_s
    ptrs_q0 = base_i + rows[:, None] * stride_s + cols_d0[None, :]
    ptrs_q1 = base_i + rows[:, None] * stride_s + cols_d1[None, :] + BLOCK_D
    
    Q_i0 = tl.load(ptrs_q0, mask=mask_q, other=0.0)
    Q_i1 = tl.load(ptrs_q1, mask=mask_q, other=0.0)
    
    base_o = O_ptr + b_h_offset + i * BLOCK_S * stride_s
    O_i0 = tl.load(base_o + rows[:, None] * stride_s + cols_d0[None, :], mask=mask_q, other=0.0)
    O_i1 = tl.load(base_o + rows[:, None] * stride_s + cols_d1[None, :] + BLOCK_D, mask=mask_q, other=0.0)
    
    base_do = dO_ptr + b_h_offset + i * BLOCK_S * stride_s
    dO_i0 = tl.load(base_do + rows[:, None] * stride_s + cols_d0[None, :], mask=mask_q, other=0.0)
    dO_i1 = tl.load(base_do + rows[:, None] * stride_s + cols_d1[None, :] + BLOCK_D, mask=mask_q, other=0.0)
    
    l_base = L_ptr + b_h_offset_L + i * BLOCK_S
    L_i = tl.load(l_base + rows, mask=q_idx_base < S, other=0.0)
    
    D = tl.sum(O_i0.to(tl.float32) * dO_i0.to(tl.float32), axis=1) + \
        tl.sum(O_i1.to(tl.float32) * dO_i1.to(tl.float32), axis=1)
    
    dQ_acc0 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    dQ_acc1 = tl.zeros((BLOCK_S, BLOCK_D), tl.float32)
    
    for j in range(0, i + 1):
        k_idx_base = j * BLOCK_S + rows
        mask_k = k_idx_base[:, None] < S
        
        base_j = K_ptr + b_h_offset + j * BLOCK_S * stride_s
        ptrs_k0 = base_j + rows[:, None] * stride_s + cols_d0[None, :]
        ptrs_k1 = base_j + rows[:, None] * stride_s + cols_d1[None, :] + BLOCK_D
        
        K_j0 = tl.load(ptrs_k0, mask=mask_k, other=0.0)
        K_j1 = tl.load(ptrs_k1, mask=mask_k, other=0.0)
        
        base_v = V_ptr + b_h_offset + j * BLOCK_S * stride_s
        ptrs_v0 = base_v + rows[:, None] * stride_s + cols_d0[None, :]
        ptrs_v1 = base_v + rows[:, None] * stride_s + cols_d1[None, :] + BLOCK_D
        
        V_j0 = tl.load(ptrs_v0, mask=mask_k, other=0.0)
        V_j1 = tl.load(ptrs_v1, mask=mask_k, other=0.0)
        
        s = tl.dot(Q_i0, K_j0.T) + tl.dot(Q_i1, K_j1.T)
        
        q_idx_2d = q_idx_base[:, None]
        k_idx_2d = k_idx_base[None, :]
        causal_mask = (q_idx_2d >= k_idx_2d) & (q_idx_2d < S) & (k_idx_2d < S)
        
        p = tl.exp(s * scale - L_i[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        dp = tl.dot(dO_i0.to(tl.float32), V_j0.T.to(tl.float32)) + \
             tl.dot(dO_i1.to(tl.float32), V_j1.T.to(tl.float32))
        
        ds = p * (dp - D[:, None]) * scale
        
        dQ_acc0 = tl.dot(ds, K_j0.to(tl.float32), dQ_acc0)
        dQ_acc1 = tl.dot(ds, K_j1.to(tl.float32), dQ_acc1)
    
    dq_base0 = dQ_ptr + b_h_offset + i * BLOCK_S * stride_s
    dq_base1 = dQ_ptr + b_h_offset + i * BLOCK_S * stride_s + BLOCK_D
    
    mask_store = (q_idx_base[:, None] < S) & (cols_d0[None, :] < 128)
    
    tl.store(dq_base0 + rows[:, None] * stride_s + cols_d0[None, :], dQ_acc0.to(tl.bfloat16), mask=mask_store)
    tl.store(dq_base1 + rows[:, None] * stride_s + cols_d1[None, :], dQ_acc1.to(tl.bfloat16), mask=mask_store)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute causal multi-head attention backward pass."""
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    scale = 1.0 / math.sqrt(float(d))
    
    stride_b = H * S * d
    stride_h = S * d
    stride_s = d
    stride_l_b = H * S
    stride_l_h = S
    
    BLOCK_S = 64
    BLOCK_D = 64
    
    num_blocks = triton.cdiv(S, BLOCK_S)
    grid = (num_blocks, H, B)
    
    dv_dk_kernel[grid](
        Q, K, V, O, dO, L, dV, dK,
        S, H, B,
        stride_b, stride_h, stride_s, stride_l_b, stride_l_h,
        scale,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=8,
        num_stages=3,
        maxnumreg=128,
    )
    
    dq_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        S, H, B,
        stride_b, stride_h, stride_s, stride_l_b, stride_l_h,
        scale,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=8,
        num_stages=3,
        maxnumreg=128,
    )