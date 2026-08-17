import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dKdV_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    dO_desc,
    L_desc,
    ptr_dK,
    ptr_dV,
    S,
    d,
    tau,
    stride_row,
    BLOCK: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    kv_idx = tl.program_id(1)
    num_blocks = tl.num_programs(1)
    
    base_offset = b_h_idx * S * d
    
    k_row_base = kv_idx * BLOCK
    cols_left = tl.arange(0, 64)
    cols_right = 64 + tl.arange(0, 64)
    
    k_left = K_desc.load([b_h_idx, k_row_base, 0]).squeeze(0)
    k_right = K_desc.load([b_h_idx, k_row_base, 64]).squeeze(0)
    v_left = V_desc.load([b_h_idx, k_row_base, 0]).squeeze(0)
    v_right = V_desc.load([b_h_idx, k_row_base, 64]).squeeze(0)
    
    dk_acc_left = tl.zeros((BLOCK, 64), tl.float32)
    dk_acc_right = tl.zeros((BLOCK, 64), tl.float32)
    dv_acc_left = tl.zeros((BLOCK, 64), tl.float32)
    dv_acc_right = tl.zeros((BLOCK, 64), tl.float32)
    
    for q_idx in range(kv_idx, num_blocks):
        q_row_base = q_idx * BLOCK
        
        Q_desc.prefetch([b_h_idx, q_row_base, 0])
        Q_desc.prefetch([b_h_idx, q_row_base, 64])
        dO_desc.prefetch([b_h_idx, q_row_base, 0])
        dO_desc.prefetch([b_h_idx, q_row_base, 64])
        O_desc.prefetch([b_h_idx, q_row_base, 0])
        O_desc.prefetch([b_h_idx, q_row_base, 64])
        
        q_left = Q_desc.load([b_h_idx, q_row_base, 0]).squeeze(0)
        q_right = Q_desc.load([b_h_idx, q_row_base, 64]).squeeze(0)
        do_left = dO_desc.load([b_h_idx, q_row_base, 0]).squeeze(0)
        do_right = dO_desc.load([b_h_idx, q_row_base, 64]).squeeze(0)
        o_left = O_desc.load([b_h_idx, q_row_base, 0]).squeeze(0)
        o_right = O_desc.load([b_h_idx, q_row_base, 64]).squeeze(0)
        
        l = L_desc.load([b_h_idx, q_row_base]).squeeze(0)
        
        d_val = tl.sum(o_left * do_left + o_right * do_right, axis=1)
        
        s_acc = tl.dot(q_left, k_left.T) + tl.dot(q_right, k_right.T)
        dp_acc = tl.dot(do_left, v_left.T) + tl.dot(do_right, v_right.T)
        
        p = tl.exp(s_acc * tau - l[:, None])
        
        q_row = q_row_base + tl.arange(0, BLOCK)
        k_row = k_row_base + tl.arange(0, BLOCK)
        causal_mask = k_row[None, :] <= q_row[:, None]
        valid_mask = (q_row[:, None] < S) & (k_row[None, :] < S) & causal_mask
        p = p * valid_mask
        
        ds = p * (dp_acc - d_val[:, None]) * tau
        
        dv_acc_left = tl.dot(p.T, do_left, dv_acc_left)
        dv_acc_right = tl.dot(p.T, do_right, dv_acc_right)
        dk_acc_left = tl.dot(ds.T, q_left, dk_acc_left)
        dk_acc_right = tl.dot(ds.T, q_right, dk_acc_right)
    
    out_k_row = k_row_base + tl.arange(0, BLOCK)
    mask_left = (out_k_row[:, None] < S)
    mask_right = (out_k_row[:, None] < S)
    
    dk_left = dk_acc_left.to(tl.bfloat16)
    dk_right = dk_acc_right.to(tl.bfloat16)
    dv_left = dv_acc_left.to(tl.bfloat16)
    dv_right = dv_acc_right.to(tl.bfloat16)
    
    tl.store(ptr_dK + base_offset + out_k_row[:, None] * stride_row + cols_left[None, :], dk_left, mask=(mask_left))
    tl.store(ptr_dK + base_offset + out_k_row[:, None] * stride_row + cols_right[None, :], dk_right, mask=(mask_right))
    tl.store(ptr_dV + base_offset + out_k_row[:, None] * stride_row + cols_left[None, :], dv_left, mask=(mask_left))
    tl.store(ptr_dV + base_offset + out_k_row[:, None] * stride_row + cols_right[None, :], dv_right, mask=(mask_right))


@triton.jit
def _bwd_dQ_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    dO_desc,
    L_desc,
    ptr_dQ,
    S,
    d,
    tau,
    stride_row,
    BLOCK: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    q_idx = tl.program_id(1)
    
    base_offset = b_h_idx * S * d
    
    q_row_base = q_idx * BLOCK
    cols_left = tl.arange(0, 64)
    cols_right = 64 + tl.arange(0, 64)
    
    q_left = Q_desc.load([b_h_idx, q_row_base, 0]).squeeze(0)
    q_right = Q_desc.load([b_h_idx, q_row_base, 64]).squeeze(0)
    do_left = dO_desc.load([b_h_idx, q_row_base, 0]).squeeze(0)
    do_right = dO_desc.load([b_h_idx, q_row_base, 64]).squeeze(0)
    o_left = O_desc.load([b_h_idx, q_row_base, 0]).squeeze(0)
    o_right = O_desc.load([b_h_idx, q_row_base, 64]).squeeze(0)
    
    l = L_desc.load([b_h_idx, q_row_base]).squeeze(0)
    d_val = tl.sum(o_left * do_left + o_right * do_right, axis=1)
    
    dq_acc_left = tl.zeros((BLOCK, 64), tl.float32)
    dq_acc_right = tl.zeros((BLOCK, 64), tl.float32)
    
    for kv_idx in range(0, q_idx + 1):
        k_row_base = kv_idx * BLOCK
        
        K_desc.prefetch([b_h_idx, k_row_base, 0])
        K_desc.prefetch([b_h_idx, k_row_base, 64])
        V_desc.prefetch([b_h_idx, k_row_base, 0])
        V_desc.prefetch([b_h_idx, k_row_base, 64])
        
        k_left = K_desc.load([b_h_idx, k_row_base, 0]).squeeze(0)
        k_right = K_desc.load([b_h_idx, k_row_base, 64]).squeeze(0)
        v_left = V_desc.load([b_h_idx, k_row_base, 0]).squeeze(0)
        v_right = V_desc.load([b_h_idx, k_row_base, 64]).squeeze(0)
        
        s_acc = tl.dot(q_left, k_left.T) + tl.dot(q_right, k_right.T)
        dp_acc = tl.dot(do_left, v_left.T) + tl.dot(do_right, v_right.T)
        
        p = tl.exp(s_acc * tau - l[:, None])
        
        q_row = q_row_base + tl.arange(0, BLOCK)
        k_row = k_row_base + tl.arange(0, BLOCK)
        causal_mask = k_row[None, :] <= q_row[:, None]
        valid_mask = (q_row[:, None] < S) & (k_row[None, :] < S) & causal_mask
        p = p * valid_mask
        
        ds = p * (dp_acc - d_val[:, None]) * tau
        
        dq_acc_left = tl.dot(ds, k_left, dq_acc_left)
        dq_acc_right = tl.dot(ds, k_right, dq_acc_right)
        
    out_q_row = q_row_base + tl.arange(0, BLOCK)
    mask_left = (out_q_row[:, None] < S)
    mask_right = (out_q_row[:, None] < S)
    
    dq_left = dq_acc_left.to(tl.bfloat16)
    dq_right = dq_acc_right.to(tl.bfloat16)
    
    tl.store(ptr_dQ + base_offset + out_q_row[:, None] * stride_row + cols_left[None, :], dq_left, mask=(mask_left))
    tl.store(ptr_dQ + base_offset + out_q_row[:, None] * stride_row + cols_right[None, :], dq_right, mask=(mask_right))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    tau = 1.0 / (d ** 0.5)
    BLOCK = 128
    stride_row = d
    
    desc_4d = lambda t: TensorDescriptor.from_tensor_4d(t, [1, 1, BLOCK, 64])
    
    Q_desc = desc_4d(Q)
    K_desc = desc_4d(K)
    V_desc = desc_4d(V)
    O_desc = desc_4d(O)
    dO_desc = desc_4d(dO)
    
    L_flat = L.flatten(0, 1)
    L_desc = TensorDescriptor.from_tensor(L_flat, [1, BLOCK])
    
    num_blocks = triton.cdiv(S, BLOCK)
    grid = (B * H, num_blocks)
    
    _bwd_dKdV_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc,
        dK, dV,
        S, d, tau, stride_row, BLOCK,
        num_warps=8, num_stages=2
    )
    _bwd_dQ_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc,
        dQ,
        S, d, tau, stride_row, BLOCK,
        num_warps=8, num_stages=2
    )