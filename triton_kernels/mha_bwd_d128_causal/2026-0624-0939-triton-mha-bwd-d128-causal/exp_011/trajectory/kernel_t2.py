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
    BLOCK: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    kv_idx = tl.program_id(1)
    num_blocks = tl.num_programs(1)
    
    k_row_base = kv_idx * BLOCK
    k_row = k_row_base + tl.arange(0, BLOCK)
    mask_k = k_row < S
    
    k_left = tl.reshape(K_desc.load([b_h_idx, k_row_base, 0]), (BLOCK, 64))
    k_right = tl.reshape(K_desc.load([b_h_idx, k_row_base, 64]), (BLOCK, 64))
    v_left = tl.reshape(V_desc.load([b_h_idx, k_row_base, 0]), (BLOCK, 64))
    v_right = tl.reshape(V_desc.load([b_h_idx, k_row_base, 64]), (BLOCK, 64))
    
    ak_left = tl.zeros((BLOCK, 64), tl.float32)
    ak_right = tl.zeros((BLOCK, 64), tl.float32)
    av_left = tl.zeros((BLOCK, 64), tl.float32)
    av_right = tl.zeros((BLOCK, 64), tl.float32)
    
    for q_idx in range(kv_idx, num_blocks):
        q_row_base = q_idx * BLOCK
        q_row = q_row_base + tl.arange(0, BLOCK)
        
        q_left = tl.reshape(Q_desc.load([b_h_idx, q_row_base, 0]), (BLOCK, 64))
        q_right = tl.reshape(Q_desc.load([b_h_idx, q_row_base, 64]), (BLOCK, 64))
        do_left = tl.reshape(dO_desc.load([b_h_idx, q_row_base, 0]), (BLOCK, 64))
        do_right = tl.reshape(dO_desc.load([b_h_idx, q_row_base, 64]), (BLOCK, 64))
        o_left = tl.reshape(O_desc.load([b_h_idx, q_row_base, 0]), (BLOCK, 64))
        o_right = tl.reshape(O_desc.load([b_h_idx, q_row_base, 64]), (BLOCK, 64))
        
        l = tl.reshape(L_desc.load([b_h_idx, q_row_base]), (BLOCK,))
        
        d_val = tl.sum(o_left * do_left + o_right * do_right, axis=1)
        
        s_acc = tl.dot(q_left, k_left.T) + tl.dot(q_right, k_right.T)
        dp_acc = tl.dot(do_left, v_left.T) + tl.dot(do_right, v_right.T)
        
        p = tl.exp(s_acc * tau - l[:, None])
        
        global_q = q_row[:, None]
        global_k = k_row[None, :]
        causal_mask = global_k <= global_q
        valid_mask = (q_row[:, None] < S) & (k_row[None, :] < S) & causal_mask
        p = p * valid_mask
        
        ds = p * (dp_acc - d_val[:, None]) * tau
        
        av_left = tl.dot(p.T, do_left, av_left)
        av_right = tl.dot(p.T, do_right, av_right)
        ak_left = tl.dot(ds.T, q_left, ak_left)
        ak_right = tl.dot(ds.T, q_right, ak_right)
    
    K_desc.store([b_h_idx, k_row_base, 0], ak_left.to(tl.bfloat16)[None, :, :])
    K_desc.store([b_h_idx, k_row_base, 64], ak_right.to(tl.bfloat16)[None, :, :])
    V_desc.store([b_h_idx, k_row_base, 0], av_left.to(tl.bfloat16)[None, :, :])
    V_desc.store([b_h_idx, k_row_base, 64], av_right.to(tl.bfloat16)[None, :, :])


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
    BLOCK: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    q_idx = tl.program_id(1)
    num_blocks = tl.num_programs(1)
    
    q_row_base = q_idx * BLOCK
    q_row = q_row_base + tl.arange(0, BLOCK)
    
    q_left = tl.reshape(Q_desc.load([b_h_idx, q_row_base, 0]), (BLOCK, 64))
    q_right = tl.reshape(Q_desc.load([b_h_idx, q_row_base, 64]), (BLOCK, 64))
    do_left = tl.reshape(dO_desc.load([b_h_idx, q_row_base, 0]), (BLOCK, 64))
    do_right = tl.reshape(dO_desc.load([b_h_idx, q_row_base, 64]), (BLOCK, 64))
    o_left = tl.reshape(O_desc.load([b_h_idx, q_row_base, 0]), (BLOCK, 64))
    o_right = tl.reshape(O_desc.load([b_h_idx, q_row_base, 64]), (BLOCK, 64))
    
    l = tl.reshape(L_desc.load([b_h_idx, q_row_base]), (BLOCK,))
    d_val = tl.sum(o_left * do_left + o_right * do_right, axis=1)
    
    ak_left = tl.zeros((BLOCK, 64), tl.float32)
    ak_right = tl.zeros((BLOCK, 64), tl.float32)
    
    for kv_idx in range(0, q_idx + 1):
        k_row_base = kv_idx * BLOCK
        k_row = k_row_base + tl.arange(0, BLOCK)
        
        k_left = tl.reshape(K_desc.load([b_h_idx, k_row_base, 0]), (BLOCK, 64))
        k_right = tl.reshape(K_desc.load([b_h_idx, k_row_base, 64]), (BLOCK, 64))
        v_left = tl.reshape(V_desc.load([b_h_idx, k_row_base, 0]), (BLOCK, 64))
        v_right = tl.reshape(V_desc.load([b_h_idx, k_row_base, 64]), (BLOCK, 64))
        
        s_acc = tl.dot(q_left, k_left.T) + tl.dot(q_right, k_right.T)
        dp_acc = tl.dot(do_left, v_left.T) + tl.dot(do_right, v_right.T)
        
        p = tl.exp(s_acc * tau - l[:, None])
        
        global_q = q_row[:, None]
        global_k = k_row[None, :]
        causal_mask = global_k <= global_q
        valid_mask = (q_row[:, None] < S) & (k_row[None, :] < S) & causal_mask
        p = p * valid_mask
        
        ds = p * (dp_acc - d_val[:, None]) * tau
        
        ak_left = tl.dot(ds, k_left, ak_left)
        ak_right = tl.dot(ds, k_right, ak_right)
        
    Q_desc.store([b_h_idx, q_row_base, 0], ak_left.to(tl.bfloat16)[None, :, :])
    Q_desc.store([b_h_idx, q_row_base, 64], ak_right.to(tl.bfloat16)[None, :, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    tau = 1.0 / (d ** 0.5)
    BLOCK = 128
    
    Q_flat = Q.flatten(0, 1)
    K_flat = K.flatten(0, 1)
    V_flat = V.flatten(0, 1)
    O_flat = O.flatten(0, 1)
    dO_flat = dO.flatten(0, 1)
    dQ_flat = dQ.flatten(0, 1)
    dK_flat = dK.flatten(0, 1)
    dV_flat = dV.flatten(0, 1)
    
    block_shape_3d = [1, BLOCK, 64]
    Q_desc = TensorDescriptor.from_tensor(Q_flat, block_shape_3d)
    K_desc = TensorDescriptor.from_tensor(K_flat, block_shape_3d)
    V_desc = TensorDescriptor.from_tensor(V_flat, block_shape_3d)
    O_desc = TensorDescriptor.from_tensor(O_flat, block_shape_3d)
    dO_desc = TensorDescriptor.from_tensor(dO_flat, block_shape_3d)
    dQ_desc = TensorDescriptor.from_tensor(dQ_flat, block_shape_3d)
    dK_desc = TensorDescriptor.from_tensor(dK_flat, block_shape_3d)
    dV_desc = TensorDescriptor.from_tensor(dV_flat, block_shape_3d)
    
    L_desc = TensorDescriptor.from_tensor(L, [1, 1, BLOCK])
    
    num_blocks = triton.cdiv(S, BLOCK)
    grid = (B * H, num_blocks)
    
    _bwd_dKdV_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc,
        dK_desc, dV_desc,
        S, d, tau, BLOCK,
        num_warps=8, num_stages=3
    )
    _bwd_dQ_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc,
        dQ_desc,
        S, d, tau, BLOCK,
        num_warps=8, num_stages=3
    )