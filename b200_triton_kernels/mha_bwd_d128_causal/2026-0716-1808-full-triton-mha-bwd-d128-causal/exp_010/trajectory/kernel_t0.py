import math
import torch
import triton
import triton.language as tl


@triton.jit
def load_2d_tile(base, bh, row_start, col_start, S, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    tile_base = base + bh * S * 128 + row_start * 128 + col_start
    offsets = tile_base + tl.arange(0, BLOCK_M)[:, None] * 128 + tl.arange(0, BLOCK_N)[None, :]
    valid_rows = (row_start + tl.arange(0, BLOCK_M)) < S
    valid_cols = (col_start + tl.arange(0, BLOCK_N)) < 128
    mask = valid_rows[:, None] & valid_cols[None, :]
    return tl.load(offsets, mask=mask, other=0.0)


@triton.jit
def store_2d_tile(base, bh, row_start, col_start, S, value, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    tile_base = base + bh * S * 128 + row_start * 128 + col_start
    offsets = tile_base + tl.arange(0, BLOCK_M)[:, None] * 128 + tl.arange(0, BLOCK_N)[None, :]
    valid_rows = (row_start + tl.arange(0, BLOCK_M)) < S
    valid_cols = (col_start + tl.arange(0, BLOCK_N)) < 128
    mask = valid_rows[:, None] & valid_cols[None, :]
    tl.store(offsets, value, mask=mask)


@triton.jit
def is_causal(global_k_idx, global_q_idx):
    return global_k_idx <= global_q_idx


@triton.jit
def bwd_dkdv_kernel(
    Q, K, V, O, dO, L, dK, dV, S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    scale = 1.0 / math.sqrt(128)
    bh = tl.program_id(0)
    j = tl.program_id(1)
    
    dK0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dK1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dV0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dV1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    K0 = load_2d_tile(K, bh, j * BLOCK_N, 0, S, BLOCK_M, BLOCK_N)
    K1 = load_2d_tile(K, bh, j * BLOCK_N, 64, S, BLOCK_M, BLOCK_N)
    V0 = load_2d_tile(V, bh, j * BLOCK_N, 0, S, BLOCK_M, BLOCK_N)
    V1 = load_2d_tile(V, bh, j * BLOCK_N, 64, S, BLOCK_M, BLOCK_N)
    
    for i in range(j, tl.cdiv(S, BLOCK_M)):
        Q0 = load_2d_tile(Q, bh, i * BLOCK_M, 0, S, BLOCK_M, BLOCK_N)
        Q1 = load_2d_tile(Q, bh, i * BLOCK_M, 64, S, BLOCK_M, BLOCK_N)
        dO0 = load_2d_tile(dO, bh, i * BLOCK_M, 0, S, BLOCK_M, BLOCK_N)
        dO1 = load_2d_tile(dO, bh, i * BLOCK_M, 64, S, BLOCK_M, BLOCK_N)
        O0 = load_2d_tile(O, bh, i * BLOCK_M, 0, S, BLOCK_M, BLOCK_N)
        O1 = load_2d_tile(O, bh, i * BLOCK_M, 64, S, BLOCK_M, BLOCK_N)
        
        D = (dO0 * O0).sum(axis=1) + (dO1 * O1).sum(axis=1)
        
        valid_rows = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
        L_i = tl.load(L + bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M), mask=valid_rows, other=0.0)
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(Q0, K0.T, acc=s)
        s = tl.dot(Q1, K1.T, acc=s)
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(dO0, V0.T, acc=dp)
        dp = tl.dot(dO1, V1.T, acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        row = tl.arange(0, BLOCK_M)
        col = tl.arange(0, BLOCK_N)
        ds = ds * is_causal(j * BLOCK_N + col, i * BLOCK_M + row)
        
        dK0 = tl.dot(ds.T, Q0, acc=dK0)
        dK1 = tl.dot(ds.T, Q1, acc=dK1)
        dV0 = tl.dot(p.T, dO0, acc=dV0)
        dV1 = tl.dot(p.T, dO1, acc=dV1)
    
    store_2d_tile(dK, bh, j * BLOCK_N, 0, S, dK0.to(tl.bfloat16), BLOCK_M, BLOCK_N)
    store_2d_tile(dK, bh, j * BLOCK_N, 64, S, dK1.to(tl.bfloat16), BLOCK_M, BLOCK_N)
    store_2d_tile(dV, bh, j * BLOCK_N, 0, S, dV0.to(tl.bfloat16), BLOCK_M, BLOCK_N)
    store_2d_tile(dV, bh, j * BLOCK_N, 64, S, dV1.to(tl.bfloat16), BLOCK_M, BLOCK_N)


@triton.jit
def bwd_dq_kernel(
    Q, K, V, O, dO, L, dQ, S,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    scale = 1.0 / math.sqrt(128)
    bh = tl.program_id(0)
    i = tl.program_id(1)
    
    dQ0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dQ1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    Q0 = load_2d_tile(Q, bh, i * BLOCK_M, 0, S, BLOCK_M, BLOCK_N)
    Q1 = load_2d_tile(Q, bh, i * BLOCK_M, 64, S, BLOCK_M, BLOCK_N)
    dO0 = load_2d_tile(dO, bh, i * BLOCK_M, 0, S, BLOCK_M, BLOCK_N)
    dO1 = load_2d_tile(dO, bh, i * BLOCK_M, 64, S, BLOCK_M, BLOCK_N)
    O0 = load_2d_tile(O, bh, i * BLOCK_M, 0, S, BLOCK_M, BLOCK_N)
    O1 = load_2d_tile(O, bh, i * BLOCK_M, 64, S, BLOCK_M, BLOCK_N)
    
    D = (dO0 * O0).sum(axis=1) + (dO1 * O1).sum(axis=1)
    
    valid_rows = (i * BLOCK_M + tl.arange(0, BLOCK_M)) < S
    L_i = tl.load(L + bh * S + i * BLOCK_M + tl.arange(0, BLOCK_M), mask=valid_rows, other=0.0)
    
    for j in range(0, i + 1):
        K0 = load_2d_tile(K, bh, j * BLOCK_N, 0, S, BLOCK_M, BLOCK_N)
        K1 = load_2d_tile(K, bh, j * BLOCK_N, 64, S, BLOCK_M, BLOCK_N)
        V0 = load_2d_tile(V, bh, j * BLOCK_N, 0, S, BLOCK_M, BLOCK_N)
        V1 = load_2d_tile(V, bh, j * BLOCK_N, 64, S, BLOCK_M, BLOCK_N)
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(Q0, K0.T, acc=s)
        s = tl.dot(Q1, K1.T, acc=s)
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(dO0, V0.T, acc=dp)
        dp = tl.dot(dO1, V1.T, acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        row = tl.arange(0, BLOCK_M)
        col = tl.arange(0, BLOCK_N)
        ds = ds * is_causal(j * BLOCK_N + col, i * BLOCK_M + row)
        
        dQ0 = tl.dot(ds, K0, acc=dQ0)
        dQ1 = tl.dot(ds, K1, acc=dQ1)
    
    store_2d_tile(dQ, bh, i * BLOCK_M, 0, S, dQ0.to(tl.bfloat16), BLOCK_M, BLOCK_N)
    store_2d_tile(dQ, bh, i * BLOCK_M, 64, S, dQ1.to(tl.bfloat16), BLOCK_M, BLOCK_N)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    BLOCK_SIZE = 64
    grid = (B * H, triton.cdiv(S, BLOCK_SIZE))
    
    bwd_dkdv_kernel[grid](Q, K, V, O, dO, L, dK, dV, S, BLOCK_M=BLOCK_SIZE, BLOCK_N=BLOCK_SIZE, num_warps=4, num_stages=2)
    bwd_dq_kernel[grid](Q, K, V, O, dO, L, dQ, S, BLOCK_M=BLOCK_SIZE, BLOCK_N=BLOCK_SIZE, num_warps=4, num_stages=2)