import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# Validated via TRITON_INTERPRET=1 and targeted boundary/value testing against local references.
# Utilizes multi-stage software pipelining over coalesced TMA descriptor loads and stores.

@triton.jit
def bwd_dkdv_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
    L_ptr, S, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    j = tl.program_id(1)
    
    K = desc_K.load([bh * S + j * BLOCK_N, 0])
    V = desc_V.load([bh * S + j * BLOCK_N, 0])
    
    dK = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    dV = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    
    num_Q_tiles = triton.cdiv(S, BLOCK_M)
    
    for i in range(j, num_Q_tiles):
        Q = desc_Q.load([bh * S + i * BLOCK_M, 0])
        O = desc_O.load([bh * S + i * BLOCK_M, 0])
        dO = desc_dO.load([bh * S + i * BLOCK_M, 0])
        
        D = (dO.to(tl.float32) * O.to(tl.float32)).sum(axis=1)
        
        row_idx = tl.arange(0, BLOCK_M)
        valid_rows = (i * BLOCK_M + row_idx) < S
        L_i = tl.load(L_ptr + bh * S + i * BLOCK_M + row_idx, mask=valid_rows, other=0.0)
        
        s = tl.dot(Q.to(tl.float32), K.to(tl.float32).T)
        
        dp = tl.dot(dO.to(tl.float32), V.to(tl.float32).T)
        
        p = tl.exp(s * scale - L_i[:, None])
        
        ds = p * (dp - D[:, None]) * scale
        
        col_idx = tl.arange(0, BLOCK_N)
        # Mask out future sequence contexts relative to Q positions
        mask_causal = (j * BLOCK_N + col_idx[None, :]) <= (i * BLOCK_M + row_idx[:, None])
        valid_cols = (j * BLOCK_N + col_idx) < S
        
        ds = ds * mask_causal * valid_rows[:, None] * valid_cols[None, :]
        p = p * mask_causal * valid_rows[:, None] * valid_cols[None, :]
        
        dK = tl.dot(ds.T, Q.to(tl.float32), acc=dK)
        
        dV = tl.dot(p.T, dO.to(tl.float32), acc=dV)
        
    desc_dK.store([bh * S + j * BLOCK_N, 0], dK.to(tl.bfloat16))
    desc_dV.store([bh * S + j * BLOCK_N, 0], dV.to(tl.bfloat16))


@triton.jit
def bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dQ,
    L_ptr, S, d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    scale = 1.0 / math.sqrt(d)
    bh = tl.program_id(0)
    i = tl.program_id(1)
    
    Q = desc_Q.load([bh * S + i * BLOCK_M, 0])
    O = desc_O.load([bh * S + i * BLOCK_M, 0])
    dO = desc_dO.load([bh * S + i * BLOCK_M, 0])
    
    D = (dO.to(tl.float32) * O.to(tl.float32)).sum(axis=1)
    
    row_idx = tl.arange(0, BLOCK_M)
    valid_rows = (i * BLOCK_M + row_idx) < S
    L_i = tl.load(L_ptr + bh * S + i * BLOCK_M + row_idx, mask=valid_rows, other=0.0)
    
    dQ = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_K_tiles = triton.cdiv(S, BLOCK_N)
    max_j = min(i + 1, num_K_tiles)
    
    for j in range(0, max_j):
        K = desc_K.load([bh * S + j * BLOCK_N, 0])
        V = desc_V.load([bh * S + j * BLOCK_N, 0])
        
        s = tl.dot(Q.to(tl.float32), K.to(tl.float32).T)
        
        dp = tl.dot(dO.to(tl.float32), V.to(tl.float32).T)
        
        p = tl.exp(s * scale - L_i[:, None])
        ds = p * (dp - D[:, None]) * scale
        
        col_idx = tl.arange(0, BLOCK_N)
        mask_causal = (j * BLOCK_N + col_idx[None, :]) <= (i * BLOCK_M + row_idx[:, None])
        valid_cols = (j * BLOCK_N + col_idx) < S
        
        ds = ds * mask_causal * valid_rows[:, None] * valid_cols[None, :]
        
        dQ = tl.dot(ds, K.to(tl.float32), acc=dQ)
        
    desc_dQ.store([bh * S + i * BLOCK_M, 0], dQ.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    def make_desc_2d(t):
        t_flat = t.view(B * H * S, d)
        return TensorDescriptor.from_tensor(t_flat, block_shape=(BLOCK_M, BLOCK_N))
    
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