import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=3),
    ],
    key=["S_len", "d"],
)
@triton.jit
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, d, H,
    HEAD_STRIDE: tl.constexpr,
    S_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    scale: tl.constexpr,
):
    i_start = tl.program_id(0) * BLOCK
    b_h_idx = tl.program_id(1)
    
    row = tl.arange(0, BLOCK)
    col0 = tl.arange(0, 64)
    col1 = tl.arange(0, 64)
    
    Q_ptr_bh = Q_ptr + b_h_idx * HEAD_STRIDE
    K_ptr_bh = K_ptr + b_h_idx * HEAD_STRIDE
    V_ptr_bh = V_ptr + b_h_idx * HEAD_STRIDE
    O_ptr_bh = O_ptr + b_h_idx * HEAD_STRIDE
    dO_ptr_bh = dO_ptr + b_h_idx * HEAD_STRIDE
    dQ_ptr_bh = dQ_ptr + b_h_idx * HEAD_STRIDE
    
    mask_i0 = (i_start + row[:, None]) < S_len
    mask_i1 = (i_start + row[:, None]) < S_len
    
    q0 = tl.load(Q_ptr_bh + (i_start + row[:, None]) * S_STRIDE + col0[None, :], mask=mask_i0, other=0.0, cache_modifier=".ca")
    q1 = tl.load(Q_ptr_bh + (i_start + row[:, None]) * S_STRIDE + (64 + col1[None, :]), mask=mask_i1, other=0.0, cache_modifier=".ca")
    
    o0 = tl.load(O_ptr_bh + (i_start + row[:, None]) * S_STRIDE + col0[None, :], mask=mask_i0, other=0.0, cache_modifier=".ca")
    o1 = tl.load(O_ptr_bh + (i_start + row[:, None]) * S_STRIDE + (64 + col1[None, :]), mask=mask_i1, other=0.0, cache_modifier=".ca")
    
    do0 = tl.load(dO_ptr_bh + (i_start + row[:, None]) * S_STRIDE + col0[None, :], mask=mask_i0, other=0.0, cache_modifier=".ca")
    do1 = tl.load(dO_ptr_bh + (i_start + row[:, None]) * S_STRIDE + (64 + col1[None, :]), mask=mask_i1, other=0.0, cache_modifier=".ca")
    
    d_sum = tl.sum(o0 * do0, axis=1, keep_dims=True) + tl.sum(o1 * do1, axis=1, keep_dims=True)
    
    l_val = tl.load(L_ptr + b_h_idx * S_len + i_start + row, mask=(i_start + row < S_len), other=0.0)
    
    dQ0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dQ1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    for j_start in range(0, S_len, BLOCK):
        k = tl.arange(0, BLOCK)
        
        mask_j0 = (j_start + k[:, None]) < S_len
        mask_j1 = (j_start + k[:, None]) < S_len
        
        k0 = tl.load(K_ptr_bh + (j_start + k[:, None]) * S_STRIDE + col0[None, :], mask=mask_j0, other=0.0, cache_modifier=".ca")
        k1 = tl.load(K_ptr_bh + (j_start + k[:, None]) * S_STRIDE + (64 + col1[None, :]), mask=mask_j1, other=0.0, cache_modifier=".ca")
        
        v0 = tl.load(V_ptr_bh + (j_start + k[:, None]) * S_STRIDE + col0[None, :], mask=mask_j0, other=0.0, cache_modifier=".ca")
        v1 = tl.load(V_ptr_bh + (j_start + k[:, None]) * S_STRIDE + (64 + col1[None, :]), mask=mask_j1, other=0.0, cache_modifier=".ca")
        
        S = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        P = tl.exp(S * scale - l_val[:, None])
        
        dP = tl.dot(do0, v0.T) + tl.dot(do1, v1.T)
        
        dS = P * (dP - d_sum) * scale
        
        dQ0_acc = tl.dot(dS, k0, acc=dQ0_acc)
        dQ1_acc = tl.dot(dS, k1, acc=dQ1_acc)
        
    valid0 = (i_start + row[:, None]) < S_len
    valid1 = (i_start + row[:, None]) < S_len
    
    ptr_dQ0 = dQ_ptr_bh + (i_start + row[:, None]) * S_STRIDE + col0[None, :]
    ptr_dQ1 = dQ_ptr_bh + (i_start + row[:, None]) * S_STRIDE + (64 + col1[None, :])
    
    tl.store(ptr_dQ0, dQ0_acc.to(tl.bfloat16), mask=valid0)
    tl.store(ptr_dQ1, dQ1_acc.to(tl.bfloat16), mask=valid1)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=3),
    ],
    key=["S_len", "d"],
)
@triton.jit
def bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, d, H,
    HEAD_STRIDE: tl.constexpr,
    S_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    scale: tl.constexpr,
):
    j_start = tl.program_id(0) * BLOCK
    b_h_idx = tl.program_id(1)
    
    row = tl.arange(0, BLOCK)
    col0 = tl.arange(0, 64)
    col1 = tl.arange(0, 64)
    
    Q_ptr_bh = Q_ptr + b_h_idx * HEAD_STRIDE
    K_ptr_bh = K_ptr + b_h_idx * HEAD_STRIDE
    V_ptr_bh = V_ptr + b_h_idx * HEAD_STRIDE
    O_ptr_bh = O_ptr + b_h_idx * HEAD_STRIDE
    dO_ptr_bh = dO_ptr + b_h_idx * HEAD_STRIDE
    dK_ptr_bh = dK_ptr + b_h_idx * HEAD_STRIDE
    dV_ptr_bh = dV_ptr + b_h_idx * HEAD_STRIDE
    
    mask_j0 = (j_start + row[:, None]) < S_len
    mask_j1 = (j_start + row[:, None]) < S_len
    
    k0 = tl.load(K_ptr_bh + (j_start + row[:, None]) * S_STRIDE + col0[None, :], mask=mask_j0, other=0.0, cache_modifier=".ca")
    k1 = tl.load(K_ptr_bh + (j_start + row[:, None]) * S_STRIDE + (64 + col1[None, :]), mask=mask_j1, other=0.0, cache_modifier=".ca")
    
    v0 = tl.load(V_ptr_bh + (j_start + row[:, None]) * S_STRIDE + col0[None, :], mask=mask_j0, other=0.0, cache_modifier=".ca")
    v1 = tl.load(V_ptr_bh + (j_start + row[:, None]) * S_STRIDE + (64 + col1[None, :]), mask=mask_j1, other=0.0, cache_modifier=".ca")
    
    dK0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dK1_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV0_acc = tl.zeros((BLOCK, 64), tl.float32)
    dV1_acc = tl.zeros((BLOCK, 64), tl.float32)
    
    for i_start in range(0, S_len, BLOCK):
        k = tl.arange(0, BLOCK)
        
        mask_i0 = (i_start + k[:, None]) < S_len
        mask_i1 = (i_start + k[:, None]) < S_len
        
        q0 = tl.load(Q_ptr_bh + (i_start + k[:, None]) * S_STRIDE + col0[None, :], mask=mask_i0, other=0.0, cache_modifier=".ca")
        q1 = tl.load(Q_ptr_bh + (i_start + k[:, None]) * S_STRIDE + (64 + col1[None, :]), mask=mask_i1, other=0.0, cache_modifier=".ca")
        
        o0 = tl.load(O_ptr_bh + (i_start + k[:, None]) * S_STRIDE + col0[None, :], mask=mask_i0, other=0.0, cache_modifier=".ca")
        o1 = tl.load(O_ptr_bh + (i_start + k[:, None]) * S_STRIDE + (64 + col1[None, :]), mask=mask_i1, other=0.0, cache_modifier=".ca")
        
        do0 = tl.load(dO_ptr_bh + (i_start + k[:, None]) * S_STRIDE + col0[None, :], mask=mask_i0, other=0.0, cache_modifier=".ca")
        do1 = tl.load(dO_ptr_bh + (i_start + k[:, None]) * S_STRIDE + (64 + col1[None, :]), mask=mask_i1, other=0.0, cache_modifier=".ca")
        
        d_sum = tl.sum(o0 * do0, axis=1, keep_dims=True) + tl.sum(o1 * do1, axis=1, keep_dims=True)
        
        l_val = tl.load(L_ptr + b_h_idx * S_len + i_start + k, mask=(i_start + k < S_len), other=0.0)
        
        S = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        P = tl.exp(S * scale - l_val[:, None])
        
        dP = tl.dot(do0, v0.T) + tl.dot(do1, v1.T)
        
        dS = P * (dP - d_sum) * scale
        
        dK0_acc = tl.dot(dS.T, q0, acc=dK0_acc)
        dK1_acc = tl.dot(dS.T, q1, acc=dK1_acc)
        
        dV0_acc = tl.dot(P.T, do0, acc=dV0_acc)
        dV1_acc = tl.dot(P.T, do1, acc=dV1_acc)
        
    valid0 = (j_start + row[:, None]) < S_len
    valid1 = (j_start + row[:, None]) < S_len
    
    ptr_dK0 = dK_ptr_bh + (j_start + row[:, None]) * S_STRIDE + col0[None, :]
    ptr_dK1 = dK_ptr_bh + (j_start + row[:, None]) * S_STRIDE + (64 + col1[None, :])
    ptr_dV0 = dV_ptr_bh + (j_start + row[:, None]) * S_STRIDE + col0[None, :]
    ptr_dV1 = dV_ptr_bh + (j_start + row[:, None]) * S_STRIDE + (64 + col1[None, :])
    
    tl.store(ptr_dK0, dK0_acc.to(tl.bfloat16), mask=valid0)
    tl.store(ptr_dK1, dK1_acc.to(tl.bfloat16), mask=valid1)
    tl.store(ptr_dV0, dV0_acc.to(tl.bfloat16), mask=valid0)
    tl.store(ptr_dV1, dV1_acc.to(tl.bfloat16), mask=valid1)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    b, h, s_len, d = Q.shape
    device = Q.device
    torch.cuda.set_device(device)
    
    scale = 1.0 / math.sqrt(d)
    
    grid = (triton.cdiv(s_len, 128), b * h)
    
    bwd_dq_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        s_len, d, h,
        HEAD_STRIDE = s_len * d,
        S_STRIDE = d,
        scale=scale,
    )
    
    bwd_dkv_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        s_len, d, h,
        HEAD_STRIDE = s_len * d,
        S_STRIDE = d,
        scale=scale,
    )