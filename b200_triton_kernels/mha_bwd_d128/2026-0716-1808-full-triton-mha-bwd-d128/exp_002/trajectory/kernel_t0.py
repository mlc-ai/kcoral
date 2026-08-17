import math
import torch
import triton
import triton.language as tl

from triton.runtime.driver import driver

ir_builder = driver.get_ir_builder()

global_mem_load = ir_builder.create_global_memory_load
shared_mem_store = ir_builder.create_shared_memory_store
shared_mem_load = ir_builder.create_shared_memory_load
global_mem_store = ir_builder.create_global_memory_store
sync = ir_builder.create_sync
extern_shared_ptr = ir_builder.create_extern_shared_ptr


@triton.jit
def ptr_load(ptr, offset, val):
    res = global_mem_load(ptr, [offset], [], [], [])
    shared_mem_store(res, ptr, [offset], [], [])
    return res

@triton.jit
def ptr_store(ptr, offset, val):
    res = shared_mem_load(ptr, [offset], [], [])
    global_mem_store(res, ptr, [offset], [], [])

@triton.jit
def load_g2s(ptr, row_offsets, col_offsets, S):
    ptr_load(ptr, row_offsets[:, None] * 128 + col_offsets[None, :], 0)
    mask = ((row_offsets[:, None] < S) & (col_offsets[None, :] < 128)).to(torch.int32)
    ptr_load(ptr, row_offsets[:, None] * 128 + col_offsets[None, :], mask)
    return ptr

@triton.jit
def store_g2s(ptr, row_offsets, col_offsets, S):
    mask = ((row_offsets[:, None] < S) & (col_offsets[None, :] < 128)).to(torch.int32)
    for c in col_offsets[None, :]:
        for r in row_offsets[:, None]:
            idx = r * 128 + c
            val = shared_mem_load(ptr, [idx], [], [])
            m = ((r < S) & (c < 128)).to(torch.int32)
            global_mem_store(val, ptr, [idx], [], [], m)

@triton.jit
def local_zeros(shape, dtype):
    return tl.full(shape, 0.0, dtype)

@triton.jit
def dot(acc, a, b):
    return tl.dot(a, b, acc)

@triton.jit
def gemm_128x128x64(a, b):
    acc = [[local_zeros((16, 16), dtype=torch.float32) for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            for k in range(4):
                acc[i][j] = dot(acc[i][j], a[i*64:(i+1)*64, :], b[:, j*64:(j+1)*64])
    return cat(merge(acc), 1)

@triton.jit
def merge(x):
    return cat(x, 0)

@triton.jit
def cat(x, dim):
    return tl.cat(x, dim=dim)

@triton.jit
def b2f(b):
    return b.to(torch.float32)


@triton.jit
@triton.heuristics(values={"BLOCK": 128})
@triton.heuristics(values={"NUM_STAGES": 2})
@triton.heuristics(values={"alpha": 1.0 / math.sqrt(128)})
def _bwd_dq_kernel(
    base_Q, base_K, base_V, base_O, base_dO, base_L, base_dQ,
    S, BLOCK: tl.constexpr, NUM_STAGES: tl.constexpr, alpha: tl.constexpr,
):
    seq_idx = tl.program_id(0) * BLOCK
    bh_idx = tl.program_id(1)
    
    row_offsets = seq_idx + tl.arange(0, 128)
    col_offsets0 = tl.arange(0, 64)
    col_offsets1 = 64 + tl.arange(0, 64)
    
    q0 = load_g2s(extern_shared_ptr(base_Q + bh_idx * S * 128 + seq_idx * 128, 16384), row_offsets, col_offsets0, S)
    q1 = load_g2s(extern_shared_ptr(base_Q + bh_idx * S * 128 + seq_idx * 128 + 64, 16384), row_offsets, col_offsets1, S)
    
    o0 = load_g2s(extern_shared_ptr(base_O + bh_idx * S * 128 + seq_idx * 128, 16384), row_offsets, col_offsets0, S)
    o1 = load_g2s(extern_shared_ptr(base_O + bh_idx * S * 128 + seq_idx * 128 + 64, 16384), row_offsets, col_offsets1, S)
    
    do0 = load_g2s(extern_shared_ptr(base_dO + bh_idx * S * 128 + seq_idx * 128, 16384), row_offsets, col_offsets0, S)
    do1 = load_g2s(extern_shared_ptr(base_dO + bh_idx * S * 128 + seq_idx * 128 + 64, 16384), row_offsets, col_offsets1, S)
    
    lse = tl.load(base_L + bh_idx * S + row_offsets, mask=row_offsets < S, other=0.0)
    
    acc_dQ0 = local_zeros((128, 64), dtype=torch.float32)
    acc_dQ1 = local_zeros((128, 64), dtype=torch.float32)
    
    num_k_tiles = (S + 127) // 128
    
    for k_tile in range(num_k_tiles):
        kv_seq_idx = k_tile * 128
        kv_row_offsets = kv_seq_idx + tl.arange(0, 128)
        
        k0 = load_g2s(extern_shared_ptr(base_K + bh_idx * S * 128 + kv_seq_idx * 128, 16384), kv_row_offsets, col_offsets0, S)
        k1 = load_g2s(extern_shared_ptr(base_K + bh_idx * S * 128 + kv_seq_idx * 128 + 64, 16384), kv_row_offsets, col_offsets1, S)
        
        v0 = load_g2s(extern_shared_ptr(base_V + bh_idx * S * 128 + kv_seq_idx * 128, 16384), kv_row_offsets, col_offsets0, S)
        v1 = load_g2s(extern_shared_ptr(base_V + bh_idx * S * 128 + kv_seq_idx * 128 + 64, 16384), kv_row_offsets, col_offsets1, S)
        
        sync()
        
        s = gemm_128x128x64(q0, k0.T) + gemm_128x128x64(q1, k1.T)
        
        dp = gemm_128x128x64(do0, v0.T) + gemm_128x128x64(do1, v1.T)
        
        p = tl.exp(s * alpha - lse[:, None])
        
        ds = p * dp
        
        acc_dQ0 = dot(acc_dQ0, ds[:, :64], k0)
        acc_dQ1 = dot(acc_dQ1, ds[:, 64:], k1)
        
        sync()
        
    acc_dQ0 = acc_dQ0 * alpha
    acc_dQ1 = acc_dQ1 * alpha
    
    store_g2s(extern_shared_ptr(base_dQ + bh_idx * S * 128 + seq_idx * 128, 16384), row_offsets, col_offsets0, S, acc_dQ0.to(dtype=torch.bfloat16))
    store_g2s(extern_shared_ptr(base_dQ + bh_idx * S * 128 + seq_idx * 128 + 64, 16384), row_offsets, col_offsets1, S, acc_dQ1.to(dtype=torch.bfloat16))


@triton.jit
@triton.heuristics(values={"BLOCK": 128})
@triton.heuristics(values={"NUM_STAGES": 2})
@triton.heuristics(values={"alpha": 1.0 / math.sqrt(128)})
def _bwd_dkv_kernel(
    base_Q, base_K, base_V, base_O, base_dO, base_L, base_dK, base_dV,
    S, BLOCK: tl.constexpr, NUM_STAGES: tl.constexpr, alpha: tl.constexpr,
):
    seq_idx = tl.program_id(0) * BLOCK
    bh_idx = tl.program_id(1)
    
    row_offsets = seq_idx + tl.arange(0, 128)
    col_offsets0 = tl.arange(0, 64)
    col_offsets1 = 64 + tl.arange(0, 64)
    
    k0 = load_g2s(extern_shared_ptr(base_K + bh_idx * S * 128 + seq_idx * 128, 16384), row_offsets, col_offsets0, S)
    k1 = load_g2s(extern_shared_ptr(base_K + bh_idx * S * 128 + seq_idx * 128 + 64, 16384), row_offsets, col_offsets1, S)
    
    v0 = load_g2s(extern_shared_ptr(base_V + bh_idx * S * 128 + seq_idx * 128, 16384), row_offsets, col_offsets0, S)
    v1 = load_g2s(extern_shared_ptr(base_V + bh_idx * S * 128 + seq_idx * 128 + 64, 16384), row_offsets, col_offsets1, S)
    
    acc_dK0 = local_zeros((128, 64), dtype=torch.float32)
    acc_dK1 = local_zeros((128, 64), dtype=torch.float32)
    
    acc_dV0 = local_zeros((128, 64), dtype=torch.float32)
    acc_dV1 = local_zeros((128, 64), dtype=torch.float32)
    
    num_q_tiles = (S + 127) // 128
    
    for q_tile in range(num_q_tiles):
        q_seq_idx = q_tile * 128
        q_row_offsets = q_seq_idx + tl.arange(0, 128)
        
        q0 = load_g2s(extern_shared_ptr(base_Q + bh_idx * S * 128 + q_seq_idx * 128, 16384), q_row_offsets, col_offsets0, S)
        q1 = load_g2s(extern_shared_ptr(base_Q + bh_idx * S * 128 + q_seq_idx * 128 + 64, 16384), q_row_offsets, col_offsets1, S)
        
        o0 = load_g2s(extern_shared_ptr(base_O + bh_idx * S * 128 + q_seq_idx * 128, 16384), q_row_offsets, col_offsets0, S)
        o1 = load_g2s(extern_shared_ptr(base_O + bh_idx * S * 128 + q_seq_idx * 128 + 64, 16384), q_row_offsets, col_offsets1, S)
        
        do0 = load_g2s(extern_shared_ptr(base_dO + bh_idx * S * 128 + q_seq_idx * 128, 16384), q_row_offsets, col_offsets0, S)
        do1 = load_g2s(extern_shared_ptr(base_dO + bh_idx * S * 128 + q_seq_idx * 128 + 64, 16384), q_row_offsets, col_offsets1, S)
        
        lse = tl.load(base_L + bh_idx * S + q_row_offsets, mask=q_row_offsets < S, other=0.0)
        
        sync()
        
        s = gemm_128x128x64(q0, k0.T) + gemm_128x128x64(q1, k1.T)
        
        dp = gemm_128x128x64(do0, v0.T) + gemm_128x128x64(do1, v1.T)
        
        p = tl.exp(s * alpha - lse[:, None])
        
        p_T = p.T
        
        ds = p * dp
        ds_T = ds.T
        
        dv0 = gemm_128x128x64(p_T, do0)
        dv1 = gemm_128x128x64(p_T, do1)
        
        dk0 = gemm_128x128x64(ds_T, q0)
        dk1 = gemm_128x128x64(ds_T, q1)
        
        acc_dV0 = dot(acc_dV0, dv0, tl.ones((64, 1), dtype=torch.float32))
        acc_dV1 = dot(acc_dV1, dv1, tl.ones((64, 1), dtype=torch.float32))
        
        acc_dK0 = dot(acc_dK0, dk0, tl.ones((64, 1), dtype=torch.float32))
        acc_dK1 = dot(acc_dK1, dk1, tl.ones((64, 1), dtype=torch.float32))
        
        sync()
        
    acc_dK0 = acc_dK0 * alpha
    acc_dK1 = acc_dK1 * alpha
    
    store_g2s(extern_shared_ptr(base_dK + bh_idx * S * 128 + seq_idx * 128, 16384), row_offsets, col_offsets0, S, acc_dK0.to(dtype=torch.bfloat16))
    store_g2s(extern_shared_ptr(base_dK + bh_idx * S * 128 + seq_idx * 128 + 64, 16384), row_offsets, col_offsets1, S, acc_dK1.to(dtype=torch.bfloat16))
    
    store_g2s(extern_shared_ptr(base_dV + bh_idx * S * 128 + seq_idx * 128, 16384), row_offsets, col_offsets0, S, acc_dV0.to(dtype=torch.bfloat16))
    store_g2s(extern_shared_ptr(base_dV + bh_idx * S * 128 + seq_idx * 128 + 64, 16384), row_offsets, col_offsets1, S, acc_dV1.to(dtype=torch.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward pass of Multi-Head Attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    
    num_tiles = (S + 127) // 128
    grid = (num_tiles, B * H)
    
    print(f"Grid: {grid}, S={S}, B*H={B*H}")
    
    _bwd_dq_kernel[grid](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), 
        dO.data_ptr(), L.data_ptr(), dQ.data_ptr(), S,
    )
    
    _bwd_dkv_kernel[grid](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), 
        dO.data_ptr(), L.data_ptr(), dK.data_ptr(), dV.data_ptr(), S,
    )