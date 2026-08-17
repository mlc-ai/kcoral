import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dq_kernel(
    base_Q, base_K, base_V, base_O, base_dO, base_L, base_dQ,
    S, alpha: tl.constexpr, d: tl.constexpr):
    
    q_tile = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    seq_idx = q_tile * 64
    
    row_offsets_q = seq_idx + tl.arange(0, 64)
    col_offsets0 = tl.arange(0, 64)
    col_offsets1 = 64 + tl.arange(0, 64)
    
    ptr_q0 = base_Q + bh_idx * S * 128 + row_offsets_q[:, None] * 128 + col_offsets0[None, :]
    ptr_q1 = base_Q + bh_idx * S * 128 + row_offsets_q[:, None] * 128 + col_offsets1[None, :]
    
    ptr_do0 = base_dO + bh_idx * S * 128 + row_offsets_q[:, None] * 128 + col_offsets0[None, :]
    ptr_do1 = base_dO + bh_idx * S * 128 + row_offsets_q[:, None] * 128 + col_offsets1[None, :]
    
    mask_q = (row_offsets_q[:, None] < S) & ((col_offsets0[None, :] < 128) | (col_offsets1[None, :] < 128))
    
    q0 = tl.load(ptr_q0, mask=mask_q, other=0.0)
    q1 = tl.load(ptr_q1, mask=mask_q, other=0.0)
    
    do0 = tl.load(ptr_do0, mask=mask_q, other=0.0)
    do1 = tl.load(ptr_do1, mask=mask_q, other=0.0)
    
    ptr_lse = base_L + bh_idx * S + row_offsets_q
    mask_lse = row_offsets_q < S
    lse = tl.load(ptr_lse, mask=mask_lse, other=0.0)
    
    acc_dQ0 = tl.zeros((64, 64), tl.float32)
    acc_dQ1 = tl.zeros((64, 64), tl.float32)
    
    num_tiles = triton.cdiv(S, 64)
    
    for kv_tile in range(num_tiles):
        kv_seq_idx = kv_tile * 64
        row_offsets_kv = kv_seq_idx + tl.arange(0, 64)
        
        ptr_k0 = base_K + bh_idx * S * 128 + row_offsets_kv[:, None] * 128 + col_offsets0[None, :]
        ptr_k1 = base_K + bh_idx * S * 128 + row_offsets_kv[:, None] * 128 + col_offsets1[None, :]
        
        k0 = tl.load(ptr_k0, mask=mask_q, other=0.0)
        k1 = tl.load(ptr_k1, mask=mask_q, other=0.0)
        
        ptr_v0 = base_V + bh_idx * S * 128 + row_offsets_kv[:, None] * 128 + col_offsets0[None, :]
        ptr_v1 = base_V + bh_idx * S * 128 + row_offsets_kv[:, None] * 128 + col_offsets1[None, :]
        
        v0 = tl.load(ptr_v0, mask=mask_q, other=0.0)
        v1 = tl.load(ptr_v1, mask=mask_q, other=0.0)
        
        s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        s = s * alpha
        
        p = tl.exp(s - lse[:, None])
        
        p *= (row_offsets_q[:, None] < S) & (row_offsets_kv[None, :] < S)
        
        dp = tl.dot(do0, v0.T) + tl.dot(do1, v1.T)
        
        ds = p * dp
        
        acc_dQ0 = tl.dot(ds, k0, acc_dQ0)
        acc_dQ1 = tl.dot(ds, k1, acc_dQ1)
    
    ptr_dQ0 = base_dQ + bh_idx * S * 128 + row_offsets_q[:, None] * 128 + col_offsets0[None, :]
    ptr_dQ1 = base_dQ + bh_idx * S * 128 + row_offsets_q[:, None] * 128 + col_offsets1[None, :]
    
    tl.store(ptr_dQ0, (acc_dQ0 * alpha).to(tl.bfloat16), mask=mask_q)
    tl.store(ptr_dQ1, (acc_dQ1 * alpha).to(tl.bfloat16), mask=mask_q)


@triton.jit
def _bwd_dkv_kernel(
    base_Q, base_K, base_V, base_O, base_dO, base_L, base_dK, base_dV,
    S, alpha: tl.constexpr, d: tl.constexpr):
    
    kv_tile = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    kv_seq_idx = kv_tile * 64
    
    row_offsets_kv = kv_seq_idx + tl.arange(0, 64)
    col_offsets0 = tl.arange(0, 64)
    col_offsets1 = 64 + tl.arange(0, 64)
    
    ptr_k0 = base_K + bh_idx * S * 128 + row_offsets_kv[:, None] * 128 + col_offsets0[None, :]
    ptr_k1 = base_K + bh_idx * S * 128 + row_offsets_kv[:, None] * 128 + col_offsets1[None, :]
    
    mask_k = (row_offsets_kv[:, None] < S) & ((col_offsets0[None, :] < 128) | (col_offsets1[None, :] < 128))
    
    k0 = tl.load(ptr_k0, mask=mask_k, other=0.0)
    k1 = tl.load(ptr_k1, mask=mask_k, other=0.0)
    
    ptr_v0 = base_V + bh_idx * S * 128 + row_offsets_kv[:, None] * 128 + col_offsets0[None, :]
    ptr_v1 = base_V + bh_idx * S * 128 + row_offsets_kv[:, None] * 128 + col_offsets1[None, :]
    
    v0 = tl.load(ptr_v0, mask=mask_k, other=0.0)
    v1 = tl.load(ptr_v1, mask=mask_k, other=0.0)
    
    acc_dK0 = tl.zeros((64, 64), tl.float32)
    acc_dK1 = tl.zeros((64, 64), tl.float32)
    
    acc_dV0 = tl.zeros((64, 64), tl.float32)
    acc_dV1 = tl.zeros((64, 64), tl.float32)
    
    num_tiles = triton.cdiv(S, 64)
    
    for q_tile in range(num_tiles):
        seq_idx = q_tile * 64
        row_offsets_q = seq_idx + tl.arange(0, 64)
        
        ptr_q0 = base_Q + bh_idx * S * 128 + row_offsets_q[:, None] * 128 + col_offsets0[None, :]
        ptr_q1 = base_Q + bh_idx * S * 128 + row_offsets_q[:, None] * 128 + col_offsets1[None, :]
        
        q0 = tl.load(ptr_q0, mask=mask_k, other=0.0)
        q1 = tl.load(ptr_q1, mask=mask_k, other=0.0)
        
        ptr_do0 = base_dO + bh_idx * S * 128 + row_offsets_q[:, None] * 128 + col_offsets0[None, :]
        ptr_do1 = base_dO + bh_idx * S * 128 + row_offsets_q[:, None] * 128 + col_offsets1[None, :]
        
        do0 = tl.load(ptr_do0, mask=mask_k, other=0.0)
        do1 = tl.load(ptr_do1, mask=mask_k, other=0.0)
        
        ptr_lse = base_L + bh_idx * S + row_offsets_q
        mask_lse = row_offsets_q < S
        lse = tl.load(ptr_lse, mask=mask_lse, other=0.0)
        
        s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        s = s * alpha
        
        p = tl.exp(s - lse[:, None])
        
        p *= (row_offsets_q[:, None] < S) & (row_offsets_kv[None, :] < S)
        
        dp = tl.dot(do0, v0.T) + tl.dot(do1, v1.T)
        
        ds = p * dp
        
        p_T = p.T
        ds_T = ds.T
        
        acc_dV0 = tl.dot(p_T, do0, acc_dV0)
        acc_dV1 = tl.dot(p_T, do1, acc_dV1)
        
        acc_dK0 = tl.dot(ds_T, q0, acc_dK0)
        acc_dK1 = tl.dot(ds_T, q1, acc_dK1)
    
    ptr_dK0 = base_dK + bh_idx * S * 128 + row_offsets_kv[:, None] * 128 + col_offsets0[None, :]
    ptr_dK1 = base_dK + bh_idx * S * 128 + row_offsets_kv[:, None] * 128 + col_offsets1[None, :]
    
    tl.store(ptr_dK0, (acc_dK0 * alpha).to(tl.bfloat16), mask=mask_k)
    tl.store(ptr_dK1, (acc_dK1 * alpha).to(tl.bfloat16), mask=mask_k)
    
    ptr_dV0 = base_dV + bh_idx * S * 128 + row_offsets_kv[:, None] * 128 + col_offsets0[None, :]
    ptr_dV1 = base_dV + bh_idx * S * 128 + row_offsets_kv[:, None] * 128 + col_offsets1[None, :]
    
    tl.store(ptr_dV0, acc_dV0.to(tl.bfloat16), mask=mask_k)
    tl.store(ptr_dV1, acc_dV1.to(tl.bfloat16), mask=mask_k)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward pass of Multi-Head Attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    alpha = 1.0 / math.sqrt(d)
    
    num_tiles = triton.cdiv(S, 64)
    grid_dq = (num_tiles, B * H)
    grid_dkv = (num_tiles, B * H)
    
    _bwd_dq_kernel[grid_dq](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), 
        dO.data_ptr(), L.data_ptr(), dQ.data_ptr(), S, alpha=alpha, d=d,
        num_warps=4, num_stages=3,
    )
    
    _bwd_dkv_kernel[grid_dkv](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), 
        dO.data_ptr(), L.data_ptr(), dK.data_ptr(), dV.data_ptr(), S, alpha=alpha, d=d,
        num_warps=4, num_stages=3,
    )