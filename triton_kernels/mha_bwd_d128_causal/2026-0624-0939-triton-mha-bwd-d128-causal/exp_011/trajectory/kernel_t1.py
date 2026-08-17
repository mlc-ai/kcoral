import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _precompute_D_kernel(
    O_ptr,
    dO_ptr,
    D_ptr,
    S,
    d,
    BLOCK_SIZE: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < d
    
    base = b_h_idx * S * d + row_idx * d + cols
    o = tl.load(O_ptr + base, mask=mask, other=0.0)
    do = tl.load(dO_ptr + base, mask=mask, other=0.0)
    
    d_val = tl.sum(o * do)
    tl.store(D_ptr + b_h_idx * S + row_idx, d_val)


@triton.jit
def _bwd_dKdV_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    dO_desc,
    L_desc,
    D_ptr,
    ptr_dK,
    ptr_dV,
    S,
    d,
    tau,
    BLOCK_SIZE: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    kv_idx = tl.program_id(1)
    num_blocks = tl.num_programs(1)
    
    k_row_base = kv_idx * BLOCK_SIZE
    k_row = k_row_base + tl.arange(0, BLOCK_SIZE)
    mask_k = k_row < S
    
    k = K_desc.load([b_h_idx, k_row_base, 0]).squeeze(0)
    v = V_desc.load([b_h_idx, k_row_base, 0]).squeeze(0)
    
    acc_dk = tl.zeros((BLOCK_SIZE, d), tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE, d), tl.float32)
    
    for q_idx in range(kv_idx, num_blocks):
        q_row_base = q_idx * BLOCK_SIZE
        q_row = q_row_base + tl.arange(0, BLOCK_SIZE)
        mask_q = q_row < S
        
        q = Q_desc.load([b_h_idx, q_row_base, 0]).squeeze(0)
        do = dO_desc.load([b_h_idx, q_row_base, 0]).squeeze(0)
        
        l = L_desc.load([b_h_idx, q_row_base]).squeeze(0)
        
        d_val = tl.load(D_ptr + b_h_idx * S + q_row, mask=mask_q, other=0.0)
        
        acc_s = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        acc_s = tl.dot(q, k.T, acc_s)
        
        acc_dp = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        acc_dp = tl.dot(do, v.T, acc_dp)
        
        s_scaled = acc_s * tau
        p = tl.exp(s_scaled - l[:, None])
        
        global_q = q_row[:, None]
        global_k = k_row[None, :]
        causal_mask = global_k <= global_q
        valid_mask = mask_q[:, None] & mask_k[None, :] & causal_mask
        p = p * valid_mask
        
        ds = p * (acc_dp - d_val[:, None]) * tau
        
        acc_dv = tl.dot(p.T, do, acc_dv)
        acc_dk = tl.dot(ds.T, q, acc_dk)
    
    c_dtype = ptr_dK.dtype.element_ty
    dk = acc_dk.to(c_dtype)
    dv = acc_dv.to(c_dtype)
    
    K_desc.store([b_h_idx, k_row_base, 0], dk[None, :, :])
    V_desc.store([b_h_idx, k_row_base, 0], dv[None, :, :])


@triton.jit
def _bwd_dQ_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    dO_desc,
    L_desc,
    D_ptr,
    ptr_dQ,
    S,
    d,
    tau,
    BLOCK_SIZE: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    q_idx = tl.program_id(1)
    num_blocks = tl.num_programs(1)
    
    q_row_base = q_idx * BLOCK_SIZE
    q_row = q_row_base + tl.arange(0, BLOCK_SIZE)
    mask_q = q_row < S
    
    q = Q_desc.load([b_h_idx, q_row_base, 0]).squeeze(0)
    do = dO_desc.load([b_h_idx, q_row_base, 0]).squeeze(0)
    
    l = L_desc.load([b_h_idx, q_row_base]).squeeze(0)
    
    d_val = tl.load(D_ptr + b_h_idx * S + q_row, mask=mask_q, other=0.0)
    
    acc_dq = tl.zeros((BLOCK_SIZE, d), tl.float32)
    
    for kv_idx in range(0, q_idx + 1):
        k_row_base = kv_idx * BLOCK_SIZE
        k_row = k_row_base + tl.arange(0, BLOCK_SIZE)
        mask_k = k_row < S
        
        k = K_desc.load([b_h_idx, k_row_base, 0]).squeeze(0)
        v = V_desc.load([b_h_idx, k_row_base, 0]).squeeze(0)
        
        acc_s = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        acc_s = tl.dot(q, k.T, acc_s)
        
        acc_dp = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        acc_dp = tl.dot(do, v.T, acc_dp)
        
        s_scaled = acc_s * tau
        p = tl.exp(s_scaled - l[:, None])
        
        global_q = q_row[:, None]
        global_k = k_row[None, :]
        causal_mask = global_k <= global_q
        valid_mask = mask_q[:, None] & mask_k[None, :] & causal_mask
        p = p * valid_mask
        
        ds = p * (acc_dp - d_val[:, None]) * tau
        
        acc_dq = tl.dot(ds, k, acc_dq)
        
    c_dtype = ptr_dQ.dtype.element_ty
    dq = acc_dq.to(c_dtype)
    
    Q_desc.store([b_h_idx, q_row_base, 0], dq[None, :, :])


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    D = torch.empty((B * H, S), dtype=torch.float32, device=Q.device)
    
    grid_D = (B * H, S, 1)
    _precompute_D_kernel[grid_D](
        O, dO, D, S, d, BLOCK_SIZE=128
    )
    
    tau = 1.0 / (d ** 0.5)
    block_size = 128
    
    Q_flat = Q.flatten(0, 1)
    K_flat = K.flatten(0, 1)
    V_flat = V.flatten(0, 1)
    O_flat = O.flatten(0, 1)
    dO_flat = dO.flatten(0, 1)
    dQ_flat = dQ.flatten(0, 1)
    dK_flat = dK.flatten(0, 1)
    dV_flat = dV.flatten(0, 1)
    
    Q_desc = TensorDescriptor.from_tensor(Q_flat, [1, block_size, block_size])
    K_desc = TensorDescriptor.from_tensor(K_flat, [1, block_size, block_size])
    V_desc = TensorDescriptor.from_tensor(V_flat, [1, block_size, block_size])
    O_desc = TensorDescriptor.from_tensor(O_flat, [1, block_size, block_size])
    dO_desc = TensorDescriptor.from_tensor(dO_flat, [1, block_size, block_size])
    dQ_desc = TensorDescriptor.from_tensor(dQ_flat, [1, block_size, block_size])
    dK_desc = TensorDescriptor.from_tensor(dK_flat, [1, block_size, block_size])
    dV_desc = TensorDescriptor.from_tensor(dV_flat, [1, block_size, block_size])
    
    L_flat = L.flatten(0, 1)
    L_desc = TensorDescriptor.from_tensor(L_flat, [1, block_size])
    
    grid = (B * H, triton.cdiv(S, block_size))
    
    _bwd_dKdV_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, D,
        dK_flat, dV_flat,
        S, d, tau,
        num_warps=8, num_stages=3, BLOCK_SIZE=block_size
    )
    _bwd_dQ_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L_desc, D,
        dQ_flat,
        S, d, tau,
        num_warps=8, num_stages=3, BLOCK_SIZE=block_size
    )