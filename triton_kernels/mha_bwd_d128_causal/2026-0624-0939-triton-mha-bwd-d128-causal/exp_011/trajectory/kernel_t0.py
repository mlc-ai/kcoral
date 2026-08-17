import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dKdV_kernel(
    ptr_Q,
    ptr_K,
    ptr_V,
    ptr_O,
    ptr_dO,
    ptr_L,
    ptr_dK,
    ptr_dV,
    S,
    d,
    stride_row,
    tau,
    BLOCK_SIZE: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    kv_idx = tl.program_id(1)
    num_blocks = tl.num_programs(1)
    
    base_offset = b_h_idx * S * d
    base_offset_l = b_h_idx * S
    
    row_offsets = tl.arange(0, BLOCK_SIZE)
    k_row = kv_idx * BLOCK_SIZE + row_offsets
    mask_k = k_row < S
    
    k0 = tl.load(ptr_K + base_offset + k_row[:, None] * stride_row + tl.arange(0, 64)[None, :], mask=(mask_k[:, None]), other=0.0)
    k1 = tl.load(ptr_K + base_offset + k_row[:, None] * stride_row + 64 + tl.arange(0, 64)[None, :], mask=(mask_k[:, None]), other=0.0)
    v0 = tl.load(ptr_V + base_offset + k_row[:, None] * stride_row + tl.arange(0, 64)[None, :], mask=(mask_k[:, None]), other=0.0)
    v1 = tl.load(ptr_V + base_offset + k_row[:, None] * stride_row + 64 + tl.arange(0, 64)[None, :], mask=(mask_k[:, None]), other=0.0)
    
    dk_acc0 = tl.zeros((BLOCK_SIZE, 64), tl.float32)
    dk_acc1 = tl.zeros((BLOCK_SIZE, 64), tl.float32)
    dv_acc0 = tl.zeros((BLOCK_SIZE, 64), tl.float32)
    dv_acc1 = tl.zeros((BLOCK_SIZE, 64), tl.float32)
    
    for q_idx in range(kv_idx, num_blocks):
        q_row = q_idx * BLOCK_SIZE + row_offsets
        mask_q = q_row < S
        
        q0 = tl.load(ptr_Q + base_offset + q_row[:, None] * stride_row + tl.arange(0, 64)[None, :], mask=(mask_q[:, None]), other=0.0)
        q1 = tl.load(ptr_Q + base_offset + q_row[:, None] * stride_row + 64 + tl.arange(0, 64)[None, :], mask=(mask_q[:, None]), other=0.0)
        do0 = tl.load(ptr_dO + base_offset + q_row[:, None] * stride_row + tl.arange(0, 64)[None, :], mask=(mask_q[:, None]), other=0.0)
        do1 = tl.load(ptr_dO + base_offset + q_row[:, None] * stride_row + 64 + tl.arange(0, 64)[None, :], mask=(mask_q[:, None]), other=0.0)
        o0  = tl.load(ptr_O  + base_offset + q_row[:, None] * stride_row + tl.arange(0, 64)[None, :], mask=(mask_q[:, None]), other=0.0)
        o1  = tl.load(ptr_O  + base_offset + q_row[:, None] * stride_row + 64 + tl.arange(0, 64)[None, :], mask=(mask_q[:, None]), other=0.0)
        
        l = tl.load(ptr_L + base_offset_l + q_row, mask=mask_q, other=0.0)
        
        d_val = tl.sum(do0 * o0 + do1 * o1, axis=1)
        
        s_acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        s_acc = tl.dot(q0, k0.T, s_acc)
        s_acc = tl.dot(q1, k1.T, s_acc)
        
        dp_acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        dp_acc = tl.dot(do0, v0.T, dp_acc)
        dp_acc = tl.dot(do1, v1.T, dp_acc)
        
        s_scaled = s_acc * tau
        p = tl.exp(s_scaled - l[:, None])
        
        global_q = q_row[:, None]
        global_k = k_row[None, :]
        causal_mask = global_k <= global_q
        valid_mask = mask_q[:, None] & causal_mask
        p = p * valid_mask
        
        ds = p * (dp_acc - d_val[:, None]) * tau
        
        dv_acc0 = tl.dot(p.T, do0, dv_acc0)
        dv_acc1 = tl.dot(p.T, do1, dv_acc1)
        dk_acc0 = tl.dot(ds.T, q0, dk_acc0)
        dk_acc1 = tl.dot(ds.T, q1, dk_acc1)
        
    out_k_row = kv_idx * BLOCK_SIZE + row_offsets
    mask_out_k = out_k_row < S
    tl.store(ptr_dK + base_offset + out_k_row[:, None] * stride_row + tl.arange(0, 64)[None, :], dk_acc0, mask=(mask_out_k[:, None]))
    tl.store(ptr_dK + base_offset + out_k_row[:, None] * stride_row + 64 + tl.arange(0, 64)[None, :], dk_acc1, mask=(mask_out_k[:, None]))
    tl.store(ptr_dV + base_offset + out_k_row[:, None] * stride_row + tl.arange(0, 64)[None, :], dv_acc0, mask=(mask_out_k[:, None]))
    tl.store(ptr_dV + base_offset + out_k_row[:, None] * stride_row + 64 + tl.arange(0, 64)[None, :], dv_acc1, mask=(mask_out_k[:, None]))


@triton.jit
def _bwd_dQ_kernel(
    ptr_Q,
    ptr_K,
    ptr_V,
    ptr_O,
    ptr_dO,
    ptr_L,
    ptr_dQ,
    S,
    d,
    stride_row,
    tau,
    BLOCK_SIZE: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    q_idx = tl.program_id(1)
    num_blocks = tl.num_programs(1)
    
    base_offset = b_h_idx * S * d
    base_offset_l = b_h_idx * S
    
    row_offsets = tl.arange(0, BLOCK_SIZE)
    q_row = q_idx * BLOCK_SIZE + row_offsets
    mask_q = q_row < S
    
    q0 = tl.load(ptr_Q + base_offset + q_row[:, None] * stride_row + tl.arange(0, 64)[None, :], mask=(mask_q[:, None]), other=0.0)
    q1 = tl.load(ptr_Q + base_offset + q_row[:, None] * stride_row + 64 + tl.arange(0, 64)[None, :], mask=(mask_q[:, None]), other=0.0)
    do0 = tl.load(ptr_dO + base_offset + q_row[:, None] * stride_row + tl.arange(0, 64)[None, :], mask=(mask_q[:, None]), other=0.0)
    do1 = tl.load(ptr_dO + base_offset + q_row[:, None] * stride_row + 64 + tl.arange(0, 64)[None, :], mask=(mask_q[:, None]), other=0.0)
    o0  = tl.load(ptr_O  + base_offset + q_row[:, None] * stride_row + tl.arange(0, 64)[None, :], mask=(mask_q[:, None]), other=0.0)
    o1  = tl.load(ptr_O  + base_offset + q_row[:, None] * stride_row + 64 + tl.arange(0, 64)[None, :], mask=(mask_q[:, None]), other=0.0)
    
    l = tl.load(ptr_L + base_offset_l + q_row, mask=mask_q, other=0.0)
    d_val = tl.sum(do0 * o0 + do1 * o1, axis=1)
    
    dq_acc0 = tl.zeros((BLOCK_SIZE, 64), tl.float32)
    dq_acc1 = tl.zeros((BLOCK_SIZE, 64), tl.float32)
    
    for kv_idx in range(0, q_idx + 1):
        k_row = kv_idx * BLOCK_SIZE + row_offsets
        mask_k = k_row < S
        
        k0 = tl.load(ptr_K + base_offset + k_row[:, None] * stride_row + tl.arange(0, 64)[None, :], mask=(mask_k[:, None]), other=0.0)
        k1 = tl.load(ptr_K + base_offset + k_row[:, None] * stride_row + 64 + tl.arange(0, 64)[None, :], mask=(mask_k[:, None]), other=0.0)
        v0 = tl.load(ptr_V + base_offset + k_row[:, None] * stride_row + tl.arange(0, 64)[None, :], mask=(mask_k[:, None]), other=0.0)
        v1 = tl.load(ptr_V + base_offset + k_row[:, None] * stride_row + 64 + tl.arange(0, 64)[None, :], mask=(mask_k[:, None]), other=0.0)
        
        s_acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        s_acc = tl.dot(q0, k0.T, s_acc)
        s_acc = tl.dot(q1, k1.T, s_acc)
        
        dp_acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        dp_acc = tl.dot(do0, v0.T, dp_acc)
        dp_acc = tl.dot(do1, v1.T, dp_acc)
        
        s_scaled = s_acc * tau
        p = tl.exp(s_scaled - l[:, None])
        
        global_q = q_row[:, None]
        global_k = k_row[None, :]
        causal_mask = global_k <= global_q
        valid_mask = mask_q[:, None] & mask_k[None, :] & causal_mask
        p = p * valid_mask
        
        ds = p * (dp_acc - d_val[:, None]) * tau
        
        dq_acc0 = tl.dot(ds, k0, dq_acc0)
        dq_acc1 = tl.dot(ds, k1, dq_acc1)
        
    out_q_row = q_idx * BLOCK_SIZE + row_offsets
    mask_out_q = out_q_row < S
    tl.store(ptr_dQ + base_offset + out_q_row[:, None] * stride_row + tl.arange(0, 64)[None, :], dq_acc0, mask=(mask_out_q[:, None]))
    tl.store(ptr_dQ + base_offset + out_q_row[:, None] * stride_row + 64 + tl.arange(0, 64)[None, :], dq_acc1, mask=(mask_out_q[:, None]))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    block_size = 64
    grid = (B * H, triton.cdiv(S, block_size))
    
    tau = 1.0 / (d ** 0.5)
    stride_row = d
    
    _bwd_dKdV_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        S, d, stride_row, tau,
        num_warps=8, num_stages=3, BLOCK_SIZE=block_size
    )
    _bwd_dQ_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        S, d, stride_row, tau,
        num_warps=8, num_stages=3, BLOCK_SIZE=block_size
    )