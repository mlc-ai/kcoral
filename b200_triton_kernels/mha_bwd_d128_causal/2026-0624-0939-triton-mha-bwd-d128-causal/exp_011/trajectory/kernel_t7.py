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
    dK_desc,
    dV_desc,
    S,
    d,
    tau,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b_idx = tl.program_id(0)
    kv_idx = tl.program_id(1)
    num_blocks = tl.num_programs(1)
    
    b = b_idx // H
    h = b_idx % H
    
    k_row_base = kv_idx * BLOCK
    
    k_left = K_desc.load([b, h, k_row_base, 0]).squeeze(0).squeeze(0)
    k_right = K_desc.load([b, h, k_row_base, 64]).squeeze(0).squeeze(0)
    v_left = V_desc.load([b, h, k_row_base, 0]).squeeze(0).squeeze(0)
    v_right = V_desc.load([b, h, k_row_base, 64]).squeeze(0).squeeze(0)
    
    dk_acc_left = tl.zeros((BLOCK, 64), tl.float32)
    dk_acc_right = tl.zeros((BLOCK, 64), tl.float32)
    dv_acc_left = tl.zeros((BLOCK, 64), tl.float32)
    dv_acc_right = tl.zeros((BLOCK, 64), tl.float32)
    
    for q_idx in range(kv_idx, num_blocks):
        q_row_base = q_idx * BLOCK
        
        q_left = Q_desc.load([b, h, q_row_base, 0]).squeeze(0).squeeze(0)
        q_right = Q_desc.load([b, h, q_row_base, 64]).squeeze(0).squeeze(0)
        do_left = dO_desc.load([b, h, q_row_base, 0]).squeeze(0).squeeze(0)
        do_right = dO_desc.load([b, h, q_row_base, 64]).squeeze(0).squeeze(0)
        o_left = O_desc.load([b, h, q_row_base, 0]).squeeze(0).squeeze(0)
        o_right = O_desc.load([b, h, q_row_base, 64]).squeeze(0).squeeze(0)
        
        l = L_desc.load([b, h, q_row_base]).squeeze(0).squeeze(0)
        
        d_val = tl.sum(o_left * do_left + o_right * do_right, axis=1)
        
        s_acc = tl.dot(q_left, k_left.T) + tl.dot(q_right, k_right.T)
        dp_acc = tl.dot(do_left, v_left.T) + tl.dot(do_right, v_right.T)
        
        p = tl.exp(s_acc * tau - l[:, None])
        
        q_row = q_row_base + tl.arange(0, BLOCK)
        k_row = k_row_base + tl.arange(0, BLOCK)
        global_q = q_row[:, None]
        global_k = k_row[None, :]
        causal_mask = global_k <= global_q
        valid_mask = (global_q < S) & (global_k < S) & causal_mask
        p = p * valid_mask
        
        ds = p * (dp_acc - d_val[:, None]) * tau
        
        dv_acc_left = tl.dot(p.T, do_left, dv_acc_left)
        dv_acc_right = tl.dot(p.T, do_right, dv_acc_right)
        dk_acc_left = tl.dot(ds.T, q_left, dk_acc_left)
        dk_acc_right = tl.dot(ds.T, q_right, dk_acc_right)
    
    dk_left = dk_acc_left.to(tl.bfloat16)
    dk_right = dk_acc_right.to(tl.bfloat16)
    dv_left = dv_acc_left.to(tl.bfloat16)
    dv_right = dv_acc_right.to(tl.bfloat16)
    
    dK_desc.store([b, h, k_row_base, 0], dk_left[None, None, :, :])
    dK_desc.store([b, h, k_row_base, 64], dk_right[None, None, :, :])
    dV_desc.store([b, h, k_row_base, 0], dv_left[None, None, :, :])
    dV_desc.store([b, h, k_row_base, 64], dv_right[None, None, :, :])


@triton.jit
def _bwd_dQ_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    dO_desc,
    L_desc,
    dQ_desc,
    S,
    d,
    tau,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b_idx = tl.program_id(0)
    q_idx = tl.program_id(1)
    
    b = b_idx // H
    h = b_idx % H
    
    q_row_base = q_idx * BLOCK
    
    q_left = Q_desc.load([b, h, q_row_base, 0]).squeeze(0).squeeze(0)
    q_right = Q_desc.load([b, h, q_row_base, 64]).squeeze(0).squeeze(0)
    do_left = dO_desc.load([b, h, q_row_base, 0]).squeeze(0).squeeze(0)
    do_right = dO_desc.load([b, h, q_row_base, 64]).squeeze(0).squeeze(0)
    o_left = O_desc.load([b, h, q_row_base, 0]).squeeze(0).squeeze(0)
    o_right = O_desc.load([b, h, q_row_base, 64]).squeeze(0).squeeze(0)
    
    l = L_desc.load([b, h, q_row_base]).squeeze(0).squeeze(0)
    
    d_val = tl.sum(o_left * do_left + o_right * do_right, axis=1)
    
    dq_acc_left = tl.zeros((BLOCK, 64), tl.float32)
    dq_acc_right = tl.zeros((BLOCK, 64), tl.float32)
    
    for kv_idx in range(0, q_idx + 1):
        k_row_base = kv_idx * BLOCK
        
        k_left = K_desc.load([b, h, k_row_base, 0]).squeeze(0).squeeze(0)
        k_right = K_desc.load([b, h, k_row_base, 64]).squeeze(0).squeeze(0)
        v_left = V_desc.load([b, h, k_row_base, 0]).squeeze(0).squeeze(0)
        v_right = V_desc.load([b, h, k_row_base, 64]).squeeze(0).squeeze(0)
        
        s_acc = tl.dot(q_left, k_left.T) + tl.dot(q_right, k_right.T)
        dp_acc = tl.dot(do_left, v_left.T) + tl.dot(do_right, v_right.T)
        
        p = tl.exp(s_acc * tau - l[:, None])
        
        q_row = q_row_base + tl.arange(0, BLOCK)
        k_row = k_row_base + tl.arange(0, BLOCK)
        global_q = q_row[:, None]
        global_k = k_row[None, :]
        causal_mask = global_k <= global_q
        valid_mask = (global_q < S) & (global_k < S) & causal_mask
        p = p * valid_mask
        
        ds = p * (dp_acc - d_val[:, None]) * tau
        
        dq_acc_left = tl.dot(ds, k_left, dq_acc_left)
        dq_acc_right = tl.dot(ds, k_right, dq_acc_right)
        
    dq_left = dq_acc_left.to(tl.bfloat16)
    dq_right = dq_acc_right.to(tl.bfloat16)
    
    dQ_desc.store([b, h, q_row_base, 0], dq_left[None, None, :, :])
    dQ_desc.store([b, h, q_row_base, 64], dq_right[None, None, :, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    tau = 1.0 / (d ** 0.5)
    BLOCK = 128
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK, 64])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK, 64])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK, 64])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK, 64])
    dO_desc = TensorDescriptor.from_tensor(dO, [1, 1, BLOCK, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ, [1, 1, BLOCK, 64])
    dK_desc = TensorDescriptor.from_tensor(dK, [1, 1, BLOCK, 64])
    dV_desc = TensorDescriptor.from_tensor(dV, [1, 1, BLOCK, 64])
    
    L_desc = TensorDescriptor.from_tensor(L, [1, 1, BLOCK])
    
    num_blocks = triton.cdiv(S, BLOCK)
    grid = (B * H, num_blocks)
    
    _bwd_dKdV_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc,
        dK_desc, dV_desc,
        S, d, tau, H, BLOCK,
        num_warps=8, num_stages=2
    )
    _bwd_dQ_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc,
        dQ_desc,
        S, d, tau, H, BLOCK,
        num_warps=8, num_stages=2
    )