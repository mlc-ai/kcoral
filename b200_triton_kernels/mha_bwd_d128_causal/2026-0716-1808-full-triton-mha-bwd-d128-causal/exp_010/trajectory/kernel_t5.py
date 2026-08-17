import math
import torch
import triton
import triton.language as tl


# Hint for occupancy optimization given our register and shared-memory footprint
@triton.jit(__launch_bounds__(256))
def bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    B, H, S, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    j = tl.program_id(1)
    
    def load_2d_tile(base_ptr, bh, row_start, col_start, S, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        tile_base = base_ptr + bh * S * 128 + row_start * 128 + col_start
        offsets = tile_base + tl.arange(0, BLOCK_M)[:, None] * 128 + tl.arange(0, BLOCK_N)[None, :]
        valid_rows = (row_start + tl.arange(0, BLOCK_M)) < S
        valid_cols = (col_start + tl.arange(0, BLOCK_N)) < 128
        mask = valid_rows[:, None] & valid_cols[None, :]
        return tl.load(offsets, mask=mask, other=0.0)
    
    def store_2d_tile(base_ptr, bh, row_start, col_start, S, value, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        tile_base = base_ptr + bh * S * 128 + row_start * 128 + col_start
        offsets = tile_base + tl.arange(0, BLOCK_M)[:, None] * 128 + tl.arange(0, BLOCK_N)[None, :]
        valid_rows = (row_start + tl.arange(0, BLOCK_M)) < S
        valid_cols = (col_start + tl.arange(0, BLOCK_N)) < 128
        mask = valid_rows[:, None] & valid_cols[None, :]
        tl.store(offsets, value.to(tl.bfloat16), mask=mask)
        
    K0 = load_2d_tile(K_ptr, bh, j * BLOCK_N, 0, S, BLOCK_N, 64)
    K1 = load_2d_tile(K_ptr, bh, j * BLOCK_N, 64, S, BLOCK_N, 64)
    V0 = load_2d_tile(V_ptr, bh, j * BLOCK_N, 0, S, BLOCK_N, 64)
    V1 = load_2d_tile(V_ptr, bh, j * BLOCK_N, 64, S, BLOCK_N, 64)
    
    dK0 = tl.zeros((BLOCK_N, 64), tl.float32)
    dK1 = tl.zeros((BLOCK_N, 64), tl.float32)
    dV0 = tl.zeros((BLOCK_N, 64), tl.float32)
    dV1 = tl.zeros((BLOCK_N, 64), tl.float32)
    
    num_Q_tiles = triton.cdiv(S, BLOCK_M)
    
    for i in range(j, num_Q_tiles):
        q0 = load_2d_tile(Q_ptr, bh, i * BLOCK_M, 0, S, BLOCK_M, 64)
        q1 = load_2d_tile(Q_ptr, bh, i * BLOCK_M, 64, S, BLOCK_M, 64)
        o0 = load_2d_tile(O_ptr, bh, i * BLOCK_M, 0, S, BLOCK_M, 64)
        o1 = load_2d_tile(O_ptr, bh, i * BLOCK_M, 64, S, BLOCK_M, 64)
        do0 = load_2d_tile(dO_ptr, bh, i * BLOCK_M, 0, S, BLOCK_M, 64)
        do1 = load_2d_tile(dO_ptr, bh, i * BLOCK_M, 64, S, BLOCK_M, 64)
        
        D = (do0.to(tl.float32) * o0.to(tl.float32)).sum(axis=1) + \
            (do1.to(tl.float32) * o1.to(tl.float32)).sum(axis=1)
        
        row_idx_D = i * BLOCK_M + tl.arange(0, BLOCK_M)
        valid_rows_D = row_idx_D < S
        
        L_i = tl.load(L_ptr + bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M), mask=valid_rows_D, other=0.0)
        
        k0_T = K0.T
        k1_T = K1.T
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(q0.to(tl.float32), k0_T.to(tl.float32), acc=s)
        s = tl.dot(q1.to(tl.float32), k1_T.to(tl.float32), acc=s)
        
        v0_T = V0.T
        v1_T = V1.T
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(do0.to(tl.float32), v0_T.to(tl.float32), acc=dp)
        dp = tl.dot(do1.to(tl.float32), v1_T.to(tl.float32), acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        row_idx = tl.arange(0, BLOCK_N)
        col_idx = tl.arange(0, BLOCK_M)
        ds = ds * ((i * BLOCK_M + col_idx[:, None]) >= (j * BLOCK_N + row_idx[None, :]))
        
        valid_q = (i * BLOCK_M + col_idx) < S
        valid_k = (j * BLOCK_N + row_idx) < S
        ds = ds * valid_q[:, None]
        ds = ds * valid_k[None, :]
        
        ds_T = ds.T
        
        dK0 = tl.dot(ds_T.to(tl.float32), q0.to(tl.float32), acc=dK0)
        dK1 = tl.dot(ds_T.to(tl.float32), q1.to(tl.float32), acc=dK1)
        
        p_T = p.T
        dV0 = tl.dot(p_T.to(tl.float32), do0.to(tl.float32), acc=dV0)
        dV1 = tl.dot(p_T.to(tl.float32), do1.to(tl.float32), acc=dV1)
        
    store_2d_tile(dK_ptr, bh, j * BLOCK_N, 0, S, dK0, BLOCK_M=BLOCK_N, BLOCK_N=64)
    store_2d_tile(dK_ptr, bh, j * BLOCK_N, 64, S, dK1, BLOCK_M=BLOCK_N, BLOCK_N=64)
    store_2d_tile(dV_ptr, bh, j * BLOCK_N, 0, S, dV0, BLOCK_M=BLOCK_N, BLOCK_N=64)
    store_2d_tile(dV_ptr, bh, j * BLOCK_N, 64, S, dV1, BLOCK_M=BLOCK_N, BLOCK_N=64)


# Hint for occupancy optimization
@triton.jit(__launch_bounds__(256))
def bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    B, H, S, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    i = tl.program_id(1)
    
    def load_2d_tile(base_ptr, bh, row_start, col_start, S, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        tile_base = base_ptr + bh * S * 128 + row_start * 128 + col_start
        offsets = tile_base + tl.arange(0, BLOCK_M)[:, None] * 128 + tl.arange(0, BLOCK_N)[None, :]
        valid_rows = (row_start + tl.arange(0, BLOCK_M)) < S
        valid_cols = (col_start + tl.arange(0, BLOCK_N)) < 128
        mask = valid_rows[:, None] & valid_cols[None, :]
        return tl.load(offsets, mask=mask, other=0.0)

    def store_2d_tile(base_ptr, bh, row_start, col_start, S, value, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        tile_base = base_ptr + bh * S * 128 + row_start * 128 + col_start
        offsets = tile_base + tl.arange(0, BLOCK_M)[:, None] * 128 + tl.arange(0, BLOCK_N)[None, :]
        valid_rows = (row_start + tl.arange(0, BLOCK_M)) < S
        valid_cols = (col_start + tl.arange(0, BLOCK_N)) < 128
        mask = valid_rows[:, None] & valid_cols[None, :]
        tl.store(offsets, value.to(tl.bfloat16), mask=mask)

    q0 = load_2d_tile(Q_ptr, bh, i * BLOCK_M, 0, S, BLOCK_M, 64)
    q1 = load_2d_tile(Q_ptr, bh, i * BLOCK_M, 64, S, BLOCK_M, 64)
    o0 = load_2d_tile(O_ptr, bh, i * BLOCK_M, 0, S, BLOCK_M, 64)
    o1 = load_2d_tile(O_ptr, bh, i * BLOCK_M, 64, S, BLOCK_M, 64)
    do0 = load_2d_tile(dO_ptr, bh, i * BLOCK_M, 0, S, BLOCK_M, 64)
    do1 = load_2d_tile(dO_ptr, bh, i * BLOCK_M, 64, S, BLOCK_M, 64)
    
    D = (do0.to(tl.float32) * o0.to(tl.float32)).sum(axis=1) + \
        (do1.to(tl.float32) * o1.to(tl.float32)).sum(axis=1)
    
    row_idx_D = i * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_rows_D = row_idx_D < S
    
    L_i = tl.load(L_ptr + bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M), mask=valid_rows_D, other=0.0)
    
    dQ0 = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    num_K_tiles = triton.cdiv(S, BLOCK_N)
    max_j = min(i + 1, num_K_tiles)
    
    for j in range(0, max_j):
        K0 = load_2d_tile(K_ptr, bh, j * BLOCK_N, 0, S, BLOCK_N, 64)
        K1 = load_2d_tile(K_ptr, bh, j * BLOCK_N, 64, S, BLOCK_N, 64)
        V0 = load_2d_tile(V_ptr, bh, j * BLOCK_N, 0, S, BLOCK_N, 64)
        V1 = load_2d_tile(V_ptr, bh, j * BLOCK_N, 64, S, BLOCK_N, 64)
        
        k0_T = K0.T
        k1_T = K1.T
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(q0.to(tl.float32), k0_T.to(tl.float32), acc=s)
        s = tl.dot(q1.to(tl.float32), k1_T.to(tl.float32), acc=s)
        
        v0_T = V0.T
        v1_T = V1.T
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(do0.to(tl.float32), v0_T.to(tl.float32), acc=dp)
        dp = tl.dot(do1.to(tl.float32), v1_T.to(tl.float32), acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        row_idx = tl.arange(0, BLOCK_N)
        col_idx = tl.arange(0, BLOCK_M)
        ds = ds * ((i * BLOCK_M + col_idx[:, None]) >= (j * BLOCK_N + row_idx[None, :]))
        
        valid_q = (i * BLOCK_M + col_idx) < S
        valid_k = (j * BLOCK_N + row_idx) < S
        ds = ds * valid_q[:, None]
        ds = ds * valid_k[None, :]
        
        dQ0 = tl.dot(ds.to(tl.float32), K0.to(tl.float32), acc=dQ0)
        dQ1 = tl.dot(ds.to(tl.float32), K1.to(tl.float32), acc=dQ1)
        
    store_2d_tile(dQ_ptr, bh, i * BLOCK_M, 0, S, dQ0, BLOCK_M=BLOCK_M, BLOCK_N=64)
    store_2d_tile(dQ_ptr, bh, i * BLOCK_M, 64, S, dQ1, BLOCK_M=BLOCK_M, BLOCK_N=64)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    grid = (B * H, triton.cdiv(S, BLOCK_M))
    
    bwd_dkdv_kernel[grid](
        Q, K, V, O, dO, L, dK, dV,
        B, H, S, d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=8, num_stages=4)
        
    bwd_dq_kernel[grid](
        Q, K, V, O, dO, L, dQ,
        B, H, S, d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=8, num_stages=4)