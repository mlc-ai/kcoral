import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# Corrected kernel utilizing TMA, shared-memory pipelining, and accurate loop scoping

@triton.jit
def bwd_dkdv_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
    L_ptr, S, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    j = tl.program_id(1)
    
    shared::var("cur_Q0", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("cur_Q1", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("cur_O0", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("cur_O1", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("cur_dO0", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("cur_dO1", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("next_Q0", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("next_Q1", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("next_O0", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("next_O1", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("next_dO0", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("next_dO1", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("c_K0", BLOCK_N * BLOCK_N * 2, align=128)
    shared::var("c_K1", BLOCK_N * BLOCK_N * 2, align=128)
    shared::var("c_V0", BLOCK_N * BLOCK_N * 2, align=128)
    shared::var("c_V1", BLOCK_N * BLOCK_N * 2, align=128)
    
    tma.async_copy(c_K0, desc_K, (bh * S + j * BLOCK_N, 0))
    tma.async_copy(c_K1, desc_K, (bh * S + j * BLOCK_N, 64))
    tma.async_copy(c_V0, desc_V, (bh * S + j * BLOCK_N, 0))
    tma.async_copy(c_V1, desc_V, (bh * S + j * BLOCK_N, 64))
    tma.commit_group()
    
    dK0 = tl.zeros((BLOCK_N, 64), tl.float32)
    dK1 = tl.zeros((BLOCK_N, 64), tl.float32)
    dV0 = tl.zeros((BLOCK_N, 64), tl.float32)
    dV1 = tl.zeros((BLOCK_N, 64), tl.float32)
    
    num_Q_tiles = tl.cdiv(S, BLOCK_M)
    
    if j < num_Q_tiles:
        tma.async_copy(cur_Q0, desc_Q, (bh * S + j * BLOCK_M, 0))
        tma.async_copy(cur_Q1, desc_Q, (bh * S + j * BLOCK_M, 64))
        tma.async_copy(cur_O0, desc_O, (bh * S + j * BLOCK_M, 0))
        tma.async_copy(cur_O1, desc_O, (bh * S + j * BLOCK_M, 64))
        tma.async_copy(cur_dO0, desc_dO, (bh * S + j * BLOCK_M, 0))
        tma.async_copy(cur_dO1, desc_dO, (bh * S + j * BLOCK_M, 64))
        tma.commit_group()
        
    for i in range(j, num_Q_tiles):
        next_i = i + 1
        if next_i < num_Q_tiles:
            tma.async_copy(next_Q0, desc_Q, (bh * S + next_i * BLOCK_M, 0))
            tma.async_copy(next_Q1, desc_Q, (bh * S + next_i * BLOCK_M, 64))
            tma.async_copy(next_O0, desc_O, (bh * S + next_i * BLOCK_M, 0))
            tma.async_copy(next_O1, desc_O, (bh * S + next_i * BLOCK_M, 64))
            tma.async_copy(next_dO0, desc_dO, (bh * S + next_i * BLOCK_M, 0))
            tma.async_copy(next_dO1, desc_dO, (bh * S + next_i * BLOCK_M, 64))
            tma.commit_group()
            
        tma.wait_group()
        
        c_Q0 = cur_Q0[0:BLOCK_M * BLOCK_N:BLOCK_N]
        c_Q1 = cur_Q1[0:BLOCK_M * BLOCK_N:BLOCK_N]
        c_O0 = cur_O0[0:BLOCK_M * BLOCK_N:BLOCK_N]
        c_O1 = cur_O1[0:BLOCK_M * BLOCK_N:BLOCK_N]
        c_dO0 = cur_dO0[0:BLOCK_M * BLOCK_N:BLOCK_N]
        c_dO1 = cur_dO1[0:BLOCK_M * BLOCK_N:BLOCK_N]
        
        D = (c_dO0.to(tl.float32) * c_O0.to(tl.float32)).sum(axis=1) + \
            (c_dO1.to(tl.float32) * c_O1.to(tl.float32)).sum(axis=1)
        
        row_idx = tl.arange(0, BLOCK_M)
        valid_rows = (i * BLOCK_M + row_idx) < S
        
        L_i = tl.load(L_ptr + bh * S + i * BLOCK_M + row_idx, mask=valid_rows, other=0.0)
        
        c_K0_s = c_K0[0:BLOCK_N * BLOCK_N:BLOCK_N]
        c_K1_s = c_K1[0:BLOCK_N * BLOCK_N:BLOCK_N]
        c_V0_s = c_V0[0:BLOCK_N * BLOCK_N:BLOCK_N]
        c_V1_s = c_V1[0:BLOCK_N * BLOCK_N:BLOCK_N]
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(c_Q0, c_K0_s.T, acc=s)
        s = tl.dot(c_Q1, c_K1_s.T, acc=s)
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(c_dO0, c_V0_s.T, acc=dp)
        dp = tl.dot(c_dO1, c_V1_s.T, acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        col_idx = tl.arange(0, BLOCK_N)
        mask_causal = (j * BLOCK_N + col_idx[None, :]) <= (i * BLOCK_M + row_idx[:, None])
        valid_cols = (j * BLOCK_N + col_idx) < S
        
        ds = ds * mask_causal * valid_rows[:, None] * valid_cols[None, :]
        p = p * mask_causal * valid_rows[:, None] * valid_cols[None, :]
        
        dK0 += tl.dot(ds.T, c_Q0, acc=dK0)
        dK1 += tl.dot(ds.T, c_Q1, acc=dK1)
        dV0 += tl.dot(p.T, c_dO0, acc=dV0)
        dV1 += tl.dot(p.T, c_dO1, acc=dV1)
        
        cur_Q0, next_Q0 = next_Q0, cur_Q0
        cur_Q1, next_Q1 = next_Q1, cur_Q1
        cur_O0, next_O0 = next_O0, cur_O0
        cur_O1, next_O1 = next_O1, cur_O1
        cur_dO0, next_dO0 = next_dO0, cur_dO0
        cur_dO1, next_dO1 = next_dO1, cur_dO1
        
    desc_dK.store([bh * S + j * BLOCK_N, 0], dK0.to(tl.bfloat16))
    desc_dK.store([bh * S + j * BLOCK_N, 64], dK1.to(tl.bfloat16))
    desc_dV.store([bh * S + j * BLOCK_N, 0], dV0.to(tl.bfloat16))
    desc_dV.store([bh * S + j * BLOCK_N, 64], dV1.to(tl.bfloat16))


@triton.jit
def bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dQ,
    L_ptr, S, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    i = tl.program_id(1)
    
    shared::var("cur_Q0", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("cur_Q1", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("cur_O0", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("cur_O1", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("cur_dO0", BLOCK_M * BLOCK_N * 2, align=128)
    shared::var("cur_dO1", BLOCK_M * BLOCK_N * 2, align=128)
    
    tma.async_copy(cur_Q0, desc_Q, (bh * S + i * BLOCK_M, 0))
    tma.async_copy(cur_Q1, desc_Q, (bh * S + i * BLOCK_M, 64))
    tma.async_copy(cur_O0, desc_O, (bh * S + i * BLOCK_M, 0))
    tma.async_copy(cur_O1, desc_O, (bh * S + i * BLOCK_M, 64))
    tma.async_copy(cur_dO0, desc_dO, (bh * S + i * BLOCK_M, 0))
    tma.async_copy(cur_dO1, desc_dO, (bh * S + i * BLOCK_M, 64))
    tma.commit_group()
    
    tma.wait_group()
    
    c_Q0 = cur_Q0[0:BLOCK_M * BLOCK_N:BLOCK_N]
    c_Q1 = cur_Q1[0:BLOCK_M * BLOCK_N:BLOCK_N]
    c_O0 = cur_O0[0:BLOCK_M * BLOCK_N:BLOCK_N]
    c_O1 = cur_O1[0:BLOCK_M * BLOCK_N:BLOCK_N]
    c_dO0 = cur_dO0[0:BLOCK_M * BLOCK_N:BLOCK_N]
    c_dO1 = cur_dO1[0:BLOCK_M * BLOCK_N:BLOCK_N]
    
    D = (c_dO0.to(tl.float32) * c_O0.to(tl.float32)).sum(axis=1) + \
        (c_dO1.to(tl.float32) * c_O1.to(tl.float32)).sum(axis=1)
    
    row_idx = tl.arange(0, BLOCK_M)
    valid_rows = (i * BLOCK_M + row_idx) < S
    
    L_i = tl.load(L_ptr + bh * S + i * BLOCK_M + row_idx, mask=valid_rows, other=0.0)
    
    dQ0 = tl.zeros((BLOCK_M, 64), tl.float32)
    dQ1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    shared::var("c_K0", BLOCK_N * BLOCK_N * 2, align=128)
    shared::var("c_K1", BLOCK_N * BLOCK_N * 2, align=128)
    shared::var("c_V0", BLOCK_N * BLOCK_N * 2, align=128)
    shared::var("c_V1", BLOCK_N * BLOCK_N * 2, align=128)
    shared::var("next_K0", BLOCK_N * BLOCK_N * 2, align=128)
    shared::var("next_K1", BLOCK_N * BLOCK_N * 2, align=128)
    shared::var("next_V0", BLOCK_N * BLOCK_N * 2, align=128)
    shared::var("next_V1", BLOCK_N * BLOCK_N * 2, align=128)
    
    num_K_tiles = tl.cdiv(S, BLOCK_N)
    max_j = min(i + 1, num_K_tiles)
    
    if 0 < max_j:
        tma.async_copy(c_K0, desc_K, (bh * S + 0 * BLOCK_N, 0))
        tma.async_copy(c_K1, desc_K, (bh * S + 0 * BLOCK_N, 64))
        tma.async_copy(c_V0, desc_V, (bh * S + 0 * BLOCK_N, 0))
        tma.async_copy(c_V1, desc_V, (bh * S + 0 * BLOCK_N, 64))
        tma.commit_group()
        
    for j in range(0, max_j):
        next_j = j + 1
        if next_j < max_j:
            tma.async_copy(next_K0, desc_K, (bh * S + next_j * BLOCK_N, 0))
            tma.async_copy(next_K1, desc_K, (bh * S + next_j * BLOCK_N, 64))
            tma.async_copy(next_V0, desc_V, (bh * S + next_j * BLOCK_N, 0))
            tma.async_copy(next_V1, desc_V, (bh * S + next_j * BLOCK_N, 64))
            tma.commit_group()
            
        tma.wait_group()
        
        c_K0_s = c_K0[0:BLOCK_N * BLOCK_N:BLOCK_N]
        c_K1_s = c_K1[0:BLOCK_N * BLOCK_N:BLOCK_N]
        c_V0_s = c_V0[0:BLOCK_N * BLOCK_N:BLOCK_N]
        c_V1_s = c_V1[0:BLOCK_N * BLOCK_N:BLOCK_N]
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(c_Q0, c_K0_s.T, acc=s)
        s = tl.dot(c_Q1, c_K1_s.T, acc=s)
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(c_dO0, c_V0_s.T, acc=dp)
        dp = tl.dot(c_dO1, c_V1_s.T, acc=dp)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        col_idx = tl.arange(0, BLOCK_N)
        mask_causal = (j * BLOCK_N + col_idx[None, :]) <= (i * BLOCK_M + row_idx[:, None])
        valid_cols = (j * BLOCK_N + col_idx) < S
        
        ds = ds * mask_causal * valid_rows[:, None] * valid_cols[None, :]
        
        dQ0 += tl.dot(ds, c_K0_s, acc=dQ0)
        dQ1 += tl.dot(ds, c_K1_s, acc=dQ1)
        
        c_K0, next_K0 = next_K0, c_K0
        c_K1, next_K1 = next_K1, c_K1
        c_V0, next_V0 = next_V0, c_V0
        c_V1, next_V1 = next_V1, c_V1
        
    desc_dQ.store([bh * S + i * BLOCK_M, 0], dQ0.to(tl.bfloat16))
    desc_dQ.store([bh * S + i * BLOCK_M, 64], dQ1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    def make_desc_2d(t):
        t_flat = t.view(B * H * S, d)
        return TensorDescriptor.from_tensor(t_flat, block_shape=(BLOCK_M, 64))
    
    Q_desc = make_desc_2d(Q)
    K_desc = make_desc_2d(K)
    V_desc = make_desc_2d(V)
    O_desc = make_desc_2d(O)
    dO_desc = make_desc_2d(dO)
    dQ_desc = make_desc_2d(dQ)
    dK_desc = make_desc_2d(dK)
    dV_desc = make_desc_2d(dV)
    
    grid = (B * H, triton.cdiv(S, BLOCK_M))
    
    bwd_dkdv_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, dK_desc, dV_desc,
        L, S, d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2)
        
    bwd_dq_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, dQ_desc,
        L, S, d, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2)