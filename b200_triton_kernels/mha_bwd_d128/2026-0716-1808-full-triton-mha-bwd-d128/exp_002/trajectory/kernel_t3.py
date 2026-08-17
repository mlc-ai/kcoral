import math
import torch
import triton
import triton.language as tl


@triton.jit
def load_tile(base_ptr, bh_idx, row_start, col_start, S, d: tl.constexpr):
    row_offsets = row_start + tl.arange(0, 64)
    col_offsets = col_start + tl.arange(0, 64)
    ptr = base_ptr + bh_idx * S * d + row_offsets[:, None] * d + col_offsets[None, :]
    mask = (row_offsets[:, None] < S) & (col_offsets[None, :] < d)
    return tl.load(ptr, mask=mask, other=0.0)


@triton.jit
def store_tile(base_ptr, bh_idx, row_start, col_start, value, S, d: tl.constexpr):
    row_offsets = row_start + tl.arange(0, 64)
    col_offsets = col_start + tl.arange(0, 64)
    ptr = base_ptr + bh_idx * S * d + row_offsets[:, None] * d + col_offsets[None, :]
    mask = (row_offsets[:, None] < S) & (col_offsets[None, :] < d)
    tl.store(ptr, value.to(tl.bfloat16), mask=mask)


@triton.jit
def load_L(base_ptr, bh_idx, row_start, S):
    row_offsets = row_start + tl.arange(0, 64)
    ptr = base_ptr + bh_idx * S + row_offsets
    mask = row_offsets < S
    return tl.load(ptr, mask=mask, other=0.0)


@triton.jit
def _bwd_dq_kernel(
    base_Q, base_K, base_V, base_O, base_dO, base_L, base_dQ,
    S, alpha: tl.constexpr, d: tl.constexpr, WARP_SPECIALIZE: tl.constexpr):
    
    q_tile = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    seq_idx = q_tile * 64
    
    acc_dQ0 = tl.zeros((64, 64), tl.float32)
    acc_dQ1 = tl.zeros((64, 64), tl.float32)
    
    q0 = load_tile(base_Q, bh_idx, seq_idx, 0, S, d)
    q1 = load_tile(base_Q, bh_idx, seq_idx, 64, S, d)
    
    do0 = load_tile(base_dO, bh_idx, seq_idx, 0, S, d)
    do1 = load_tile(base_dO, bh_idx, seq_idx, 64, S, d)
    
    lse = load_L(base_L, bh_idx, seq_idx, S)
    
    num_tiles = triton.cdiv(S, 64)
    
    for k_step in range(num_tiles):
        kv_seq_idx = k_step * 64
        
        k0 = load_tile(base_K, bh_idx, kv_seq_idx, 0, S, d)
        k1 = load_tile(base_K, bh_idx, kv_seq_idx, 64, S, d)
        
        v0 = load_tile(base_V, bh_idx, kv_seq_idx, 0, S, d)
        v1 = load_tile(base_V, bh_idx, kv_seq_idx, 64, S, d)
        
        s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        s = s * alpha
        
        p = tl.exp(s - lse[:, None])
        
        dp = tl.dot(do0, v0.T) + tl.dot(do1, v1.T)
        
        ds = p * dp
        
        acc_dQ0 = tl.dot(ds, k0, acc_dQ0)
        acc_dQ1 = tl.dot(ds, k1, acc_dQ1)
    
    store_tile(base_dQ, bh_idx, seq_idx, 0, acc_dQ0 * alpha, S, d)
    store_tile(base_dQ, bh_idx, seq_idx, 64, acc_dQ1 * alpha, S, d)


@triton.jit
def _bwd_dkv_kernel(
    base_Q, base_K, base_V, base_O, base_dO, base_L, base_dK, base_dV,
    S, alpha: tl.constexpr, d: tl.constexpr, WARP_SPECIALIZE: tl.constexpr):
    
    q_tile = tl.program_id(0)
    kv_tile = tl.program_id(1)
    bh_idx = tl.program_id(2)
    
    seq_idx = q_tile * 64
    kv_seq_idx = kv_tile * 64
    
    acc_dK0 = tl.zeros((64, 64), tl.float32)
    acc_dK1 = tl.zeros((64, 64), tl.float32)
    acc_dV0 = tl.zeros((64, 64), tl.float32)
    acc_dV1 = tl.zeros((64, 64), tl.float32)
    
    k0 = load_tile(base_K, bh_idx, kv_seq_idx, 0, S, d)
    k1 = load_tile(base_K, bh_idx, kv_seq_idx, 64, S, d)
    
    v0 = load_tile(base_V, bh_idx, kv_seq_idx, 0, S, d)
    v1 = load_tile(base_V, bh_idx, kv_seq_idx, 64, S, d)
    
    q0 = load_tile(base_Q, bh_idx, seq_idx, 0, S, d)
    q1 = load_tile(base_Q, bh_idx, seq_idx, 64, S, d)
    
    do0 = load_tile(base_dO, bh_idx, seq_idx, 0, S, d)
    do1 = load_tile(base_dO, bh_idx, seq_idx, 64, S, d)
    
    lse = load_L(base_L, bh_idx, seq_idx, S)
    
    s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
    s = s * alpha
    
    p = tl.exp(s - lse[:, None])
    
    dp = tl.dot(do0, v0.T) + tl.dot(do1, v1.T)
    
    ds = p * dp
    
    ds_T = ds.T
    
    acc_dV0 = tl.dot(p.T, do0, acc_dV0)
    acc_dV1 = tl.dot(p.T, do1, acc_dV1)
    
    acc_dK0 = tl.dot(ds_T, q0, acc_dK0)
    acc_dK1 = tl.dot(ds_T, q1, acc_dK1)
    
    store_tile(base_dK, bh_idx, kv_seq_idx, 0, acc_dK0 * alpha, S, d)
    store_tile(base_dK, bh_idx, kv_seq_idx, 64, acc_dK1 * alpha, S, d)
    
    store_tile(base_dV, bh_idx, kv_seq_idx, 0, acc_dV0, S, d)
    store_tile(base_dV, bh_idx, kv_seq_idx, 64, acc_dV1, S, d)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward pass of Multi-Head Attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    alpha = 1.0 / math.sqrt(d)
    
    num_tiles = triton.cdiv(S, 64)
    grid_dq = (num_tiles, B * H)
    grid_dkv = (num_tiles, num_tiles, B * H)
    
    print(f"Grid dQ: {grid_dq}, Grid dKV: {grid_dkv}, S={S}, B*H={B*H}")
    
    _bwd_dq_kernel[grid_dq](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), 
        dO.data_ptr(), L.data_ptr(), dQ.data_ptr(), S, alpha=alpha, d=d, WARP_SPECIALIZE=False,
        num_warps=4, num_stages=3,
    )
    
    _bwd_dkv_kernel[grid_dkv](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), 
        dO.data_ptr(), L.data_ptr(), dK.data_ptr(), dV.data_ptr(), S, alpha=alpha, d=d, WARP_SPECIALIZE=False,
        num_warps=4, num_stages=3,
    )