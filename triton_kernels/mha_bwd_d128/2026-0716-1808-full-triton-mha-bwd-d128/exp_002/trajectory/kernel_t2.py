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
def load_g2s(ptr, row_offsets, col_offsets, S):
    ptr = ptr_load(ptr, row_offsets[:, None] * 128 + col_offsets[None, :], 0)
    mask = ((row_offsets[:, None] < S) & (col_offsets[None, :] < 128)).to(tl.int32)
    ptr_load(ptr, row_offsets[:, None] * 128 + col_offsets[None, :], mask)
    return ptr

@triton.jit
def load_sub_tile(ptr, row, col, h, w):
    tile = []
    for r in range(row, row+h):
        row_data = []
        for c in range(col, col+w):
            val = shared_mem_load(ptr, [r * 32 + c], [], [])
            row_data.append(val)
        tile.append(row_data)
    return tile

@triton.jit
def store_g2s(ptr, row_offsets, col_offsets, S, val):
    mask = ((row_offsets[:, None] < S) & (col_offsets[None, :] < 128)).to(tl.int32)
    for c in col_offsets[None, :]:
        for r in row_offsets[:, None]:
            idx = r * 128 + c
            val = shared_mem_load(ptr, [idx], [], [])
            m = ((r < S) & (c < 128)).to(tl.int32)
            global_mem_store(val, ptr, [idx], [], [], m)

@triton.jit
def load_L(ptr, row_offsets, S):
    ptr = extern_shared_ptr(ptr, 128)
    for i in range(32):
        if row_offsets[i] < S:
            val = global_mem_load(ptr, [row_offsets[i]], [], [], [])
        else:
            val = 0.0
        shared_mem_store(val, ptr, [i], [], [])
    return ptr

@triton.jit
def local_zeros(shape, dtype):
    return tl.full(shape, 0.0, dtype)

@triton.jit
def dot(acc, a, b):
    return tl.dot(a, b, acc)

@triton.jit
def gemm_32x32x32(a, b):
    acc = [[local_zeros((16, 16), dtype=tl.float32) for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a_tile = load_sub_tile(a, k*16, 0, 16, 16)
                b_tile = load_sub_tile(b, k*16, j*16, 16, 16)
                acc[i][j] = dot(acc[i][j], a_tile, b_tile)
    merged = merge(acc)
    return merged

@triton.jit
def merge(x):
    return cat(x, 0)

@triton.jit
def cat(x, dim):
    return tl.cat(x, dim=dim)


@triton.jit
def _bwd_dq_kernel(
    base_Q, base_K, base_V, base_O, base_dO, base_L, base_dQ,
    S, alpha: tl.constexpr,
):
    q_tile = tl.program_id(0)
    kv_tile = tl.program_id(1)
    bh_idx = tl.program_id(2)
    
    seq_idx = q_tile * 32
    kv_seq_idx = kv_tile * 32
    
    row_offsets = seq_idx + tl.arange(0, 32)
    kv_row_offsets = kv_seq_idx + tl.arange(0, 32)
    
    col_offsets0 = tl.arange(0, 32)
    col_offsets1 = 32 + tl.arange(0, 32)
    col_offsets2 = 64 + tl.arange(0, 32)
    col_offsets3 = 96 + tl.arange(0, 32)
    
    base_ptr_Q = base_Q + -bh_idx * S * 128
    base_ptr_K = base_K + -bh_idx * S * 128
    base_ptr_V = base_V + -bh_idx * S * 128
    base_ptr_dO = base_dO + -bh_idx * S * 128
    base_ptr_L = base_L + -bh_idx * S
    base_ptr_dQ = base_dQ + -bh_idx * S * 128
    
    q0 = load_g2s(base_ptr_Q, row_offsets, col_offsets0, S)
    q1 = load_g2s(base_ptr_Q, row_offsets, col_offsets1, S)
    q2 = load_g2s(base_ptr_Q, row_offsets, col_offsets2, S)
    q3 = load_g2s(base_ptr_Q, row_offsets, col_offsets3, S)
    
    do0 = load_g2s(base_ptr_dO, row_offsets, col_offsets0, S)
    do1 = load_g2s(base_ptr_dO, row_offsets, col_offsets1, S)
    do2 = load_g2s(base_ptr_dO, row_offsets, col_offsets2, S)
    do3 = load_g2s(base_ptr_dO, row_offsets, col_offsets3, S)
    
    lse = load_L(base_ptr_L, row_offsets, S)
    
    acc_dQ0 = local_zeros((32, 32), dtype=tl.float32)
    acc_dQ1 = local_zeros((32, 32), dtype=tl.float32)
    acc_dQ2 = local_zeros((32, 32), dtype=tl.float32)
    acc_dQ3 = local_zeros((32, 32), dtype=tl.float32)
    
    k0 = load_g2s(base_ptr_K, kv_row_offsets, col_offsets0, S)
    k1 = load_g2s(base_ptr_K, kv_row_offsets, col_offsets1, S)
    k2 = load_g2s(base_ptr_K, kv_row_offsets, col_offsets2, S)
    k3 = load_g2s(base_ptr_K, kv_row_offsets, col_offsets3, S)
    
    v0 = load_g2s(base_ptr_V, kv_row_offsets, col_offsets0, S)
    v1 = load_g2s(base_ptr_V, kv_row_offsets, col_offsets1, S)
    v2 = load_g2s(base_ptr_V, kv_row_offsets, col_offsets2, S)
    v3 = load_g2s(base_ptr_V, kv_row_offsets, col_offsets3, S)
    
    sync()
    
    acc_S = local_zeros((32, 32), dtype=tl.float32)
    acc_S = dot(acc_S, q0, k0.T)
    acc_S = dot(acc_S, q1, k1.T)
    acc_S = dot(acc_S, q2, k2.T)
    acc_S = dot(acc_S, q3, k3.T)
    
    s = acc_S * alpha
    
    exp_s = tl.exp(s)
    p = exp_s - lse[:, None]
    
    acc_dP = local_zeros((32, 32), dtype=tl.float32)
    acc_dP = dot(acc_dP, do0, v0.T)
    acc_dP = dot(acc_dP, do1, v1.T)
    acc_dP = dot(acc_dP, do2, v2.T)
    acc_dP = dot(acc_dP, do3, v3.T)
    
    ds = p * acc_dP
    
    acc_dQ0 = dot(acc_dQ0, ds, k0)
    acc_dQ1 = dot(acc_dQ1, ds, k1)
    acc_dQ2 = dot(acc_dQ2, ds, k2)
    acc_dQ3 = dot(acc_dQ3, ds, k3)
    
    sync()
    
    acc_dQ0 = acc_dQ0 * alpha
    acc_dQ1 = acc_dQ1 * alpha
    acc_dQ2 = acc_dQ2 * alpha
    acc_dQ3 = acc_dQ3 * alpha
    
    global_mem_store(acc_dQ0.to(tl.bfloat16), base_ptr_dQ, [row_offsets[:, None] * 128 + col_offsets0[None, :]], [], [], ((row_offsets[:, None] < S) & (col_offsets0[None, :] < 128)).to(tl.int32))
    global_mem_store(acc_dQ1.to(tl.bfloat16), base_ptr_dQ, [row_offsets[:, None] * 128 + col_offsets1[None, :]], [], [], ((row_offsets[:, None] < S) & (col_offsets1[None, :] < 128)).to(tl.int32))
    global_mem_store(acc_dQ2.to(tl.bfloat16), base_ptr_dQ, [row_offsets[:, None] * 128 + col_offsets2[None, :]], [], [], ((row_offsets[:, None] < S) & (col_offsets2[None, :] < 128)).to(tl.int32))
    global_mem_store(acc_dQ3.to(tl.bfloat16), base_ptr_dQ, [row_offsets[:, None] * 128 + col_offsets3[None, :]], [], [], ((row_offsets[:, None] < S) & (col_offsets3[None, :] < 128)).to(tl.int32))


@triton.jit
def _bwd_dkv_kernel(
    base_Q, base_K, base_V, base_O, base_dO, base_L, base_dK, base_dV,
    S, alpha: tl.constexpr,
):
    q_tile = tl.program_id(0)
    kv_tile = tl.program_id(1)
    bh_idx = tl.program_id(2)
    
    seq_idx = q_tile * 32
    kv_seq_idx = kv_tile * 32
    
    row_offsets = seq_idx + tl.arange(0, 32)
    kv_row_offsets = kv_seq_idx + tl.arange(0, 32)
    
    col_offsets0 = tl.arange(0, 32)
    col_offsets1 = 32 + tl.arange(0, 32)
    col_offsets2 = 64 + tl.arange(0, 32)
    col_offsets3 = 96 + tl.arange(0, 32)
    
    base_ptr_Q = base_Q + -bh_idx * S * 128
    base_ptr_K = base_K + -bh_idx * S * 128
    base_ptr_V = base_V + -bh_idx * S * 128
    base_ptr_dO = base_dO + -bh_idx * S * 128
    base_ptr_L = base_L + -bh_idx * S
    base_ptr_dK = base_dK + -bh_idx * S * 128
    base_ptr_dV = base_dV + -bh_idx * S * 128
    
    q0 = load_g2s(base_ptr_Q, row_offsets, col_offsets0, S)
    q1 = load_g2s(base_ptr_Q, row_offsets, col_offsets1, S)
    q2 = load_g2s(base_ptr_Q, row_offsets, col_offsets2, S)
    q3 = load_g2s(base_ptr_Q, row_offsets, col_offsets3, S)
    
    do0 = load_g2s(base_ptr_dO, row_offsets, col_offsets0, S)
    do1 = load_g2s(base_ptr_dO, row_offsets, col_offsets1, S)
    do2 = load_g2s(base_ptr_dO, row_offsets, col_offsets2, S)
    do3 = load_g2s(base_ptr_dO, row_offsets, col_offsets3, S)
    
    lse = load_L(base_ptr_L, row_offsets, S)
    
    acc_dK0 = local_zeros((32, 32), dtype=tl.float32)
    acc_dK1 = local_zeros((32, 32), dtype=tl.float32)
    acc_dK2 = local_zeros((32, 32), dtype=tl.float32)
    acc_dK3 = local_zeros((32, 32), dtype=tl.float32)
    
    acc_dV0 = local_zeros((32, 32), dtype=tl.float32)
    acc_dV1 = local_zeros((32, 32), dtype=tl.float32)
    acc_dV2 = local_zeros((32, 32), dtype=tl.float32)
    acc_dV3 = local_zeros((32, 32), dtype=tl.float32)
    
    k0 = load_g2s(base_ptr_K, kv_row_offsets, col_offsets0, S)
    k1 = load_g2s(base_ptr_K, kv_row_offsets, col_offsets1, S)
    k2 = load_g2s(base_ptr_K, kv_row_offsets, col_offsets2, S)
    k3 = load_g2s(base_ptr_K, kv_row_offsets, col_offsets3, S)
    
    v0 = load_g2s(base_ptr_V, kv_row_offsets, col_offsets0, S)
    v1 = load_g2s(base_ptr_V, kv_row_offsets, col_offsets1, S)
    v2 = load_g2s(base_ptr_V, kv_row_offsets, col_offsets2, S)
    v3 = load_g2s(base_ptr_V, kv_row_offsets, col_offsets3, S)
    
    sync()
    
    acc_S = local_zeros((32, 32), dtype=tl.float32)
    acc_S = dot(acc_S, q0, k0.T)
    acc_S = dot(acc_S, q1, k1.T)
    acc_S = dot(acc_S, q2, k2.T)
    acc_S = dot(acc_S, q3, k3.T)
    
    s = acc_S * alpha
    
    exp_s = tl.exp(s)
    p = exp_s - lse[:, None]
    
    acc_dP = local_zeros((32, 32), dtype=tl.float32)
    acc_dP = dot(acc_dP, do0, v0.T)
    acc_dP = dot(acc_dP, do1, v1.T)
    acc_dP = dot(acc_dP, do2, v2.T)
    acc_dP = dot(acc_dP, do3, v3.T)
    
    ds = p * acc_dP
    
    ds_T = ds.T
    p_T = p.T
    
    acc_dV0 = dot(acc_dV0, p_T, do0)
    acc_dV1 = dot(acc_dV1, p_T, do1)
    acc_dV2 = dot(acc_dV2, p_T, do2)
    acc_dV3 = dot(acc_dV3, p_T, do3)
    
    acc_dK0 = dot(acc_dK0, ds_T, q0)
    acc_dK1 = dot(acc_dK1, ds_T, q1)
    acc_dK2 = dot(acc_dK2, ds_T, q2)
    acc_dK3 = dot(acc_dK3, ds_T, q3)
    
    sync()
    
    acc_dK0 = acc_dK0 * alpha
    acc_dK1 = acc_dK1 * alpha
    acc_dK2 = acc_dK2 * alpha
    acc_dK3 = acc_dK3 * alpha
    
    global_mem_store(acc_dK0.to(tl.bfloat16), base_ptr_dK, [kv_row_offsets[:, None] * 128 + col_offsets0[None, :]], [], [], ((kv_row_offsets[:, None] < S) & (col_offsets0[None, :] < 128)).to(tl.int32))
    global_mem_store(acc_dK1.to(tl.bfloat16), base_ptr_dK, [kv_row_offsets[:, None] * 128 + col_offsets1[None, :]], [], [], ((kv_row_offsets[:, None] < S) & (col_offsets1[None, :] < 128)).to(tl.int32))
    global_mem_store(acc_dK2.to(tl.bfloat16), base_ptr_dK, [kv_row_offsets[:, None] * 128 + col_offsets2[None, :]], [], [], ((kv_row_offsets[:, None] < S) & (col_offsets2[None, :] < 128)).to(tl.int32))
    global_mem_store(acc_dK3.to(tl.bfloat16), base_ptr_dK, [kv_row_offsets[:, None] * 128 + col_offsets3[None, :]], [], [], ((kv_row_offsets[:, None] < S) & (col_offsets3[None, :] < 128)).to(tl.int32))
    
    global_mem_store(acc_dV0.to(tl.bfloat16), base_ptr_dV, [kv_row_offsets[:, None] * 128 + col_offsets0[None, :]], [], [], ((kv_row_offsets[:, None] < S) & (col_offsets0[None, :] < 128)).to(tl.int32))
    global_mem_store(acc_dV1.to(tl.bfloat16), base_ptr_dV, [kv_row_offsets[:, None] * 128 + col_offsets1[None, :]], [], [], ((kv_row_offsets[:, None] < S) & (col_offsets1[None, :] < 128)).to(tl.int32))
    global_mem_store(acc_dV2.to(tl.bfloat16), base_ptr_dV, [kv_row_offsets[:, None] * 128 + col_offsets2[None, :]], [], [], ((kv_row_offsets[:, None] < S) & (col_offsets2[None, :] < 128)).to(tl.int32))
    global_mem_store(acc_dV3.to(tl.bfloat16), base_ptr_dV, [kv_row_offsets[:, None] * 128 + col_offsets3[None, :]], [], [], ((kv_row_offsets[:, None] < S) & (col_offsets3[None, :] < 128)).to(tl.int32))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward pass of Multi-Head Attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    alpha = 1.0 / math.sqrt(d)
    
    num_tiles = triton.cdiv(S, 32)
    grid = (num_tiles, num_tiles, B * H)
    
    print(f"Grid: {grid}, S={S}, B*H={B*H}")
    
    _bwd_dq_kernel[grid](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), 
        dO.data_ptr(), L.data_ptr(), dQ.data_ptr(), S, alpha=alpha,
        num_warps=4,
    )
    
    _bwd_dkv_kernel[grid](
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), 
        dO.data_ptr(), L.data_ptr(), dK.data_ptr(), dV.data_ptr(), S, alpha=alpha,
        num_warps=4,
    )