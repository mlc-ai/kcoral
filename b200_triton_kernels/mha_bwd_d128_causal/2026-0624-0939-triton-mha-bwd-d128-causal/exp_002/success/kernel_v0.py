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
    d_step: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    j = tl.program_id(0)
    
    b_h_offset = b_idx * stride_b + h_idx * stride_h
    b_h_offset_L = b_idx * stride_l_b + h_idx * stride_l_h
    
    rows = tl.arange(0, BLOCK_S)
    
    cols0 = tl.arange(0, d_step)
    cols1 = tl.arange(0, d_step)
    cols2 = tl.arange(0, d_step)
    cols3 = tl.arange(0, d_step)
    
    base_k0 = K_ptr + b_h_offset + j * BLOCK_S * stride_s + 0
    base_k1 = K_ptr + b_h_offset + j * BLOCK_S * stride_s + d_step
    base_k2 = K_ptr + b_h_offset + j * BLOCK_S * stride_s + 2 * d_step
    base_k3 = K_ptr + b_h_offset + j * BLOCK_S * stride_s + 3 * d_step
    
    base_v0 = V_ptr + b_h_offset + j * BLOCK_S * stride_s + 0
    base_v1 = V_ptr + b_h_offset + j * BLOCK_S * stride_s + d_step
    base_v2 = V_ptr + b_h_offset + j * BLOCK_S * stride_s + 2 * d_step
    base_v3 = V_ptr + b_h_offset + j * BLOCK_S * stride_s + 3 * d_step
    
    mask_k = (j * BLOCK_S + rows[:, None]) < S
    
    K_j0 = tl.load(base_k0 + rows[:, None] * stride_s + cols0[None, :], mask=mask_k, other=0.0)
    K_j1 = tl.load(base_k1 + rows[:, None] * stride_s + cols1[None, :], mask=mask_k, other=0.0)
    K_j2 = tl.load(base_k2 + rows[:, None] * stride_s + cols2[None, :], mask=mask_k, other=0.0)
    K_j3 = tl.load(base_k3 + rows[:, None] * stride_s + cols3[None, :], mask=mask_k, other=0.0)
    
    V_j0 = tl.load(base_v0 + rows[:, None] * stride_s + cols0[None, :], mask=mask_k, other=0.0)
    V_j1 = tl.load(base_v1 + rows[:, None] * stride_s + cols1[None, :], mask=mask_k, other=0.0)
    V_j2 = tl.load(base_v2 + rows[:, None] * stride_s + cols2[None, :], mask=mask_k, other=0.0)
    V_j3 = tl.load(base_v3 + rows[:, None] * stride_s + cols3[None, :], mask=mask_k, other=0.0)
    
    dV_acc0 = tl.zeros((BLOCK_S, d_step), tl.float32)
    dV_acc1 = tl.zeros((BLOCK_S, d_step), tl.float32)
    dV_acc2 = tl.zeros((BLOCK_S, d_step), tl.float32)
    dV_acc3 = tl.zeros((BLOCK_S, d_step), tl.float32)
    
    dK_acc0 = tl.zeros((BLOCK_S, d_step), tl.float32)
    dK_acc1 = tl.zeros((BLOCK_S, d_step), tl.float32)
    dK_acc2 = tl.zeros((BLOCK_S, d_step), tl.float32)
    dK_acc3 = tl.zeros((BLOCK_S, d_step), tl.float32)
    
    num_blocks = tl.cdiv(S, BLOCK_S)
    
    for i in range(j, num_blocks):
        base_q0 = Q_ptr + b_h_offset + i * BLOCK_S * stride_s + 0
        base_q1 = Q_ptr + b_h_offset + i * BLOCK_S * stride_s + d_step
        base_q2 = Q_ptr + b_h_offset + i * BLOCK_S * stride_s + 2 * d_step
        base_q3 = Q_ptr + b_h_offset + i * BLOCK_S * stride_s + 3 * d_step
        
        base_o0 = O_ptr + b_h_offset + i * BLOCK_S * stride_s + 0
        base_o1 = O_ptr + b_h_offset + i * BLOCK_S * stride_s + d_step
        base_o2 = O_ptr + b_h_offset + i * BLOCK_S * stride_s + 2 * d_step
        base_o3 = O_ptr + b_h_offset + i * BLOCK_S * stride_s + 3 * d_step
        
        base_do0 = dO_ptr + b_h_offset + i * BLOCK_S * stride_s + 0
        base_do1 = dO_ptr + b_h_offset + i * BLOCK_S * stride_s + d_step
        base_do2 = dO_ptr + b_h_offset + i * BLOCK_S * stride_s + 2 * d_step
        base_do3 = dO_ptr + b_h_offset + i * BLOCK_S * stride_s + 3 * d_step
        
        mask_q = (i * BLOCK_S + rows[:, None]) < S
        
        Q_i0 = tl.load(base_q0 + rows[:, None] * stride_s + cols0[None, :], mask=mask_q, other=0.0)
        Q_i1 = tl.load(base_q1 + rows[:, None] * stride_s + cols1[None, :], mask=mask_q, other=0.0)
        Q_i2 = tl.load(base_q2 + rows[:, None] * stride_s + cols2[None, :], mask=mask_q, other=0.0)
        Q_i3 = tl.load(base_q3 + rows[:, None] * stride_s + cols3[None, :], mask=mask_q, other=0.0)
        
        O_i0 = tl.load(base_o0 + rows[:, None] * stride_s + cols0[None, :], mask=mask_q, other=0.0)
        O_i1 = tl.load(base_o1 + rows[:, None] * stride_s + cols1[None, :], mask=mask_q, other=0.0)
        O_i2 = tl.load(base_o2 + rows[:, None] * stride_s + cols2[None, :], mask=mask_q, other=0.0)
        O_i3 = tl.load(base_o3 + rows[:, None] * stride_s + cols3[None, :], mask=mask_q, other=0.0)
        
        dO_i0 = tl.load(base_do0 + rows[:, None] * stride_s + cols0[None, :], mask=mask_q, other=0.0)
        dO_i1 = tl.load(base_do1 + rows[:, None] * stride_s + cols1[None, :], mask=mask_q, other=0.0)
        dO_i2 = tl.load(base_do2 + rows[:, None] * stride_s + cols2[None, :], mask=mask_q, other=0.0)
        dO_i3 = tl.load(base_do3 + rows[:, None] * stride_s + cols3[None, :], mask=mask_q, other=0.0)
        
        q_idx_base = i * BLOCK_S + rows
        L_i = tl.load(L_ptr + b_h_offset_L + q_idx_base, mask=q_idx_base < S, other=0.0)
        
        D_i = 0.0
        D_i += tl.sum(O_i0.to(tl.float32) * dO_i0.to(tl.float32), axis=1)
        D_i += tl.sum(O_i1.to(tl.float32) * dO_i1.to(tl.float32), axis=1)
        D_i += tl.sum(O_i2.to(tl.float32) * dO_i2.to(tl.float32), axis=1)
        D_i += tl.sum(O_i3.to(tl.float32) * dO_i3.to(tl.float32), axis=1)
        
        s = tl.dot(Q_i0, K_j0.T) + tl.dot(Q_i1, K_j1.T) + \
            tl.dot(Q_i2, K_j2.T) + tl.dot(Q_i3, K_j3.T)
        
        q_idx_2d = q_idx_base[:, None]
        k_idx_base = j * BLOCK_S + rows
        k_idx_2d = k_idx_base[None, :]
        causal_mask = (q_idx_2d >= k_idx_2d) & (q_idx_2d < S) & (k_idx_2d < S)
        
        p = tl.exp(s * scale - L_i[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        dp = tl.dot(dO_i0.to(tl.float32), V_j0.T.to(tl.float32)) + \
             tl.dot(dO_i1.to(tl.float32), V_j1.T.to(tl.float32)) + \
             tl.dot(dO_i2.to(tl.float32), V_j2.T.to(tl.float32)) + \
             tl.dot(dO_i3.to(tl.float32), V_j3.T.to(tl.float32))
        
        ds = p * (dp - D_i[:, None]) * scale
        
        dV_acc0 = tl.dot(p.T, dO_i0.to(tl.float32), dV_acc0)
        dV_acc1 = tl.dot(p.T, dO_i1.to(tl.float32), dV_acc1)
        dV_acc2 = tl.dot(p.T, dO_i2.to(tl.float32), dV_acc2)
        dV_acc3 = tl.dot(p.T, dO_i3.to(tl.float32), dV_acc3)
        
        dK_acc0 = tl.dot(ds.T, Q_i0.to(tl.float32), dK_acc0)
        dK_acc1 = tl.dot(ds.T, Q_i1.to(tl.float32), dK_acc1)
        dK_acc2 = tl.dot(ds.T, Q_i2.to(tl.float32), dK_acc2)
        dK_acc3 = tl.dot(ds.T, Q_i3.to(tl.float32), dK_acc3)
    
    dv_base0 = dV_ptr + b_h_offset + j * BLOCK_S * stride_s + 0
    dv_base1 = dV_ptr + b_h_offset + j * BLOCK_S * stride_s + d_step
    dv_base2 = dV_ptr + b_h_offset + j * BLOCK_S * stride_s + 2 * d_step
    dv_base3 = dV_ptr + b_h_offset + j * BLOCK_S * stride_s + 3 * d_step
    
    dk_base0 = dK_ptr + b_h_offset + j * BLOCK_S * stride_s + 0
    dk_base1 = dK_ptr + b_h_offset + j * BLOCK_S * stride_s + d_step
    dk_base2 = dK_ptr + b_h_offset + j * BLOCK_S * stride_s + 2 * d_step
    dk_base3 = dK_ptr + b_h_offset + j * BLOCK_S * stride_s + 3 * d_step
    
    mask_store = (j * BLOCK_S + rows[:, None]) < S
    
    tl.store(dv_base0 + rows[:, None] * stride_s + cols0[None, :], dV_acc0.to(tl.bfloat16), mask=mask_store)
    tl.store(dv_base1 + rows[:, None] * stride_s + cols1[None, :], dV_acc1.to(tl.bfloat16), mask=mask_store)
    tl.store(dv_base2 + rows[:, None] * stride_s + cols2[None, :], dV_acc2.to(tl.bfloat16), mask=mask_store)
    tl.store(dv_base3 + rows[:, None] * stride_s + cols3[None, :], dV_acc3.to(tl.bfloat16), mask=mask_store)
    
    tl.store(dk_base0 + rows[:, None] * stride_s + cols0[None, :], dK_acc0.to(tl.bfloat16), mask=mask_store)
    tl.store(dk_base1 + rows[:, None] * stride_s + cols1[None, :], dK_acc1.to(tl.bfloat16), mask=mask_store)
    tl.store(dk_base2 + rows[:, None] * stride_s + cols2[None, :], dK_acc2.to(tl.bfloat16), mask=mask_store)
    tl.store(dk_base3 + rows[:, None] * stride_s + cols3[None, :], dK_acc3.to(tl.bfloat16), mask=mask_store)


@triton.jit
def dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    S, H, B,
    stride_b, stride_h, stride_s, stride_l_b, stride_l_h,
    scale,
    BLOCK_S: tl.constexpr,
    d_step: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    i = tl.program_id(0)
    
    b_h_offset = b_idx * stride_b + h_idx * stride_h
    b_h_offset_L = b_idx * stride_l_b + h_idx * stride_l_h
    
    rows = tl.arange(0, BLOCK_S)
    
    cols0 = tl.arange(0, d_step)
    cols1 = tl.arange(0, d_step)
    cols2 = tl.arange(0, d_step)
    cols3 = tl.arange(0, d_step)
    
    base_q0 = Q_ptr + b_h_offset + i * BLOCK_S * stride_s + 0
    base_q1 = Q_ptr + b_h_offset + i * BLOCK_S * stride_s + d_step
    base_q2 = Q_ptr + b_h_offset + i * BLOCK_S * stride_s + 2 * d_step
    base_q3 = Q_ptr + b_h_offset + i * BLOCK_S * stride_s + 3 * d_step
    
    base_o0 = O_ptr + b_h_offset + i * BLOCK_S * stride_s + 0
    base_o1 = O_ptr + b_h_offset + i * BLOCK_S * stride_s + d_step
    base_o2 = O_ptr + b_h_offset + i * BLOCK_S * stride_s + 2 * d_step
    base_o3 = O_ptr + b_h_offset + i * BLOCK_S * stride_s + 3 * d_step
    
    base_do0 = dO_ptr + b_h_offset + i * BLOCK_S * stride_s + 0
    base_do1 = dO_ptr + b_h_offset + i * BLOCK_S * stride_s + d_step
    base_do2 = dO_ptr + b_h_offset + i * BLOCK_S * stride_s + 2 * d_step
    base_do3 = dO_ptr + b_h_offset + i * BLOCK_S * stride_s + 3 * d_step
    
    mask_q = (i * BLOCK_S + rows[:, None]) < S
    
    Q_i0 = tl.load(base_q0 + rows[:, None] * stride_s + cols0[None, :], mask=mask_q, other=0.0)
    Q_i1 = tl.load(base_q1 + rows[:, None] * stride_s + cols1[None, :], mask=mask_q, other=0.0)
    Q_i2 = tl.load(base_q2 + rows[:, None] * stride_s + cols2[None, :], mask=mask_q, other=0.0)
    Q_i3 = tl.load(base_q3 + rows[:, None] * stride_s + cols3[None, :], mask=mask_q, other=0.0)
    
    O_i0 = tl.load(base_o0 + rows[:, None] * stride_s + cols0[None, :], mask=mask_q, other=0.0)
    O_i1 = tl.load(base_o1 + rows[:, None] * stride_s + cols1[None, :], mask=mask_q, other=0.0)
    O_i2 = tl.load(base_o2 + rows[:, None] * stride_s + cols2[None, :], mask=mask_q, other=0.0)
    O_i3 = tl.load(base_o3 + rows[:, None] * stride_s + cols3[None, :], mask=mask_q, other=0.0)
    
    dO_i0 = tl.load(base_do0 + rows[:, None] * stride_s + cols0[None, :], mask=mask_q, other=0.0)
    dO_i1 = tl.load(base_do1 + rows[:, None] * stride_s + cols1[None, :], mask=mask_q, other=0.0)
    dO_i2 = tl.load(base_do2 + rows[:, None] * stride_s + cols2[None, :], mask=mask_q, other=0.0)
    dO_i3 = tl.load(base_do3 + rows[:, None] * stride_s + cols3[None, :], mask=mask_q, other=0.0)
    
    q_idx_base = i * BLOCK_S + rows
    L_i = tl.load(L_ptr + b_h_offset_L + q_idx_base, mask=q_idx_base < S, other=0.0)
    
    D_i = 0.0
    D_i += tl.sum(O_i0.to(tl.float32) * dO_i0.to(tl.float32), axis=1)
    D_i += tl.sum(O_i1.to(tl.float32) * dO_i1.to(tl.float32), axis=1)
    D_i += tl.sum(O_i2.to(tl.float32) * dO_i2.to(tl.float32), axis=1)
    D_i += tl.sum(O_i3.to(tl.float32) * dO_i3.to(tl.float32), axis=1)
    
    dQ_acc0 = tl.zeros((BLOCK_S, d_step), tl.float32)
    dQ_acc1 = tl.zeros((BLOCK_S, d_step), tl.float32)
    dQ_acc2 = tl.zeros((BLOCK_S, d_step), tl.float32)
    dQ_acc3 = tl.zeros((BLOCK_S, d_step), tl.float32)
    
    num_blocks = tl.cdiv(S, BLOCK_S)
    
    for j in range(0, i + 1):
        base_k0 = K_ptr + b_h_offset + j * BLOCK_S * stride_s + 0
        base_k1 = K_ptr + b_h_offset + j * BLOCK_S * stride_s + d_step
        base_k2 = K_ptr + b_h_offset + j * BLOCK_S * stride_s + 2 * d_step
        base_k3 = K_ptr + b_h_offset + j * BLOCK_S * stride_s + 3 * d_step
        
        base_v0 = V_ptr + b_h_offset + j * BLOCK_S * stride_s + 0
        base_v1 = V_ptr + b_h_offset + j * BLOCK_S * stride_s + d_step
        base_v2 = V_ptr + b_h_offset + j * BLOCK_S * stride_s + 2 * d_step
        base_v3 = V_ptr + b_h_offset + j * BLOCK_S * stride_s + 3 * d_step
        
        mask_k = (j * BLOCK_S + rows[:, None]) < S
        
        K_j0 = tl.load(base_k0 + rows[:, None] * stride_s + cols0[None, :], mask=mask_k, other=0.0)
        K_j1 = tl.load(base_k1 + rows[:, None] * stride_s + cols1[None, :], mask=mask_k, other=0.0)
        K_j2 = tl.load(base_k2 + rows[:, None] * stride_s + cols2[None, :], mask=mask_k, other=0.0)
        K_j3 = tl.load(base_k3 + rows[:, None] * stride_s + cols3[None, :], mask=mask_k, other=0.0)
        
        V_j0 = tl.load(base_v0 + rows[:, None] * stride_s + cols0[None, :], mask=mask_k, other=0.0)
        V_j1 = tl.load(base_v1 + rows[:, None] * stride_s + cols1[None, :], mask=mask_k, other=0.0)
        V_j2 = tl.load(base_v2 + rows[:, None] * stride_s + cols2[None, :], mask=mask_k, other=0.0)
        V_j3 = tl.load(base_v3 + rows[:, None] * stride_s + cols3[None, :], mask=mask_k, other=0.0)
        
        s = tl.dot(Q_i0, K_j0.T) + tl.dot(Q_i1, K_j1.T) + \
            tl.dot(Q_i2, K_j2.T) + tl.dot(Q_i3, K_j3.T)
        
        q_idx_2d = q_idx_base[:, None]
        k_idx_base = j * BLOCK_S + rows
        k_idx_2d = k_idx_base[None, :]
        causal_mask = (q_idx_2d >= k_idx_2d) & (q_idx_2d < S) & (k_idx_2d < S)
        
        p = tl.exp(s * scale - L_i[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        dp = tl.dot(dO_i0.to(tl.float32), V_j0.T.to(tl.float32)) + \
             tl.dot(dO_i1.to(tl.float32), V_j1.T.to(tl.float32)) + \
             tl.dot(dO_i2.to(tl.float32), V_j2.T.to(tl.float32)) + \
             tl.dot(dO_i3.to(tl.float32), V_j3.T.to(tl.float32))
        
        ds = p * (dp - D_i[:, None]) * scale
        
        dQ_acc0 = tl.dot(ds, K_j0.to(tl.float32), dQ_acc0)
        dQ_acc1 = tl.dot(ds, K_j1.to(tl.float32), dQ_acc1)
        dQ_acc2 = tl.dot(ds, K_j2.to(tl.float32), dQ_acc2)
        dQ_acc3 = tl.dot(ds, K_j3.to(tl.float32), dQ_acc3)
    
    dq_base0 = dQ_ptr + b_h_offset + i * BLOCK_S * stride_s + 0
    dq_base1 = dQ_ptr + b_h_offset + i * BLOCK_S * stride_s + d_step
    dq_base2 = dQ_ptr + b_h_offset + i * BLOCK_S * stride_s + 2 * d_step
    dq_base3 = dQ_ptr + b_h_offset + i * BLOCK_S * stride_s + 3 * d_step
    
    mask_store = (i * BLOCK_S + rows[:, None]) < S
    
    tl.store(dq_base0 + rows[:, None] * stride_s + cols0[None, :], dQ_acc0.to(tl.bfloat16), mask=mask_store)
    tl.store(dq_base1 + rows[:, None] * stride_s + cols1[None, :], dQ_acc1.to(tl.bfloat16), mask=mask_store)
    tl.store(dq_base2 + rows[:, None] * stride_s + cols2[None, :], dQ_acc2.to(tl.bfloat16), mask=mask_store)
    tl.store(dq_base3 + rows[:, None] * stride_s + cols3[None, :], dQ_acc3.to(tl.bfloat16), mask=mask_store)


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
    d_step = 32
    
    dQ.zero_()
    dK.zero_()
    dV.zero_()
    
    num_blocks = triton.cdiv(S, BLOCK_S)
    grid = (num_blocks, H, B)
    
    dv_dk_kernel[grid](
        Q, K, V, O, dO, L, dV, dK,
        S, H, B,
        stride_b, stride_h, stride_s, stride_l_b, stride_l_h,
        scale,
        BLOCK_S=BLOCK_S,
        d_step=d_step,
        num_warps=8,
        num_stages=3,
    )
    
    dq_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        S, H, B,
        stride_b, stride_h, stride_s, stride_l_b, stride_l_h,
        scale,
        BLOCK_S=BLOCK_S,
        d_step=d_step,
        num_warps=8,
        num_stages=3,
    )