import torch
import triton
import triton.language as tl
import math


@triton.jit
def load_tile(base_ptr, sh_offset, s_offset, row_idx, col_idx, row_mask, stride_s, stride_d, col_offset=0):
    """Load a tile from a generic 4D tensor."""
    ptr = base_ptr + sh_offset + (s_offset + row_idx[:, None]) * stride_s + (col_offset + col_idx[None, :]) * stride_d
    return tl.load(ptr, mask=row_mask[:, None], other=0.0)


@triton.jit
def store_tile(base_ptr, sh_offset, s_offset, value_2d, row_mask, stride_s, stride_d, col_offset=0):
    """Store a tile into a generic 4D tensor."""
    B_M, B_N = value_2d.shape
    row_idx = tl.arange(0, B_M)
    col_idx = tl.arange(0, B_N)
    ptr = base_ptr + sh_offset + (s_offset + row_idx[:, None]) * stride_s + (col_offset + col_idx[None, :]) * stride_d
    tl.store(ptr, value_2d, mask=row_mask[:, None])


@triton.jit
def _bwd_kernel_dq(Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr, S,
                    stride_b, stride_h, stride_s, stride_d,
                    stride_bh_L, stride_h_L):
    """Compute dQ by iterating over query tiles and their relevant causal key tiles."""
    i = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    sh_offset = b * stride_b + h * stride_h
    s_offset_i = i * 128
    
    row_idx = tl.arange(0, 128)
    col_idx_0 = tl.arange(0, 64)
    col_idx_1 = tl.arange(0, 64)
    
    row_mask_i = (s_offset_i + row_idx) < S
    
    Q_0 = load_tile(Q_ptr, sh_offset, s_offset_i, row_idx, col_idx_0, row_mask_i, stride_s, stride_d, 0)
    Q_1 = load_tile(Q_ptr, sh_offset, s_offset_i, row_idx, col_idx_1, row_mask_i, stride_s, stride_d, 64)
    dO_0 = load_tile(dO_ptr, sh_offset, s_offset_i, row_idx, col_idx_0, row_mask_i, stride_s, stride_d, 0)
    dO_1 = load_tile(dO_ptr, sh_offset, s_offset_i, row_idx, col_idx_1, row_mask_i, stride_s, stride_d, 64)
    
    L_i = tl.load(L_ptr + b * stride_bh_L + h * stride_h_L + s_offset_i + row_idx, mask=row_mask_i, other=0.0)
    
    scale = tl.full((1,), 1.0 / math.sqrt(128), dtype=tl.float32)
    
    dQ_0 = tl.zeros((128, 64), dtype=tl.float32)
    dQ_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    num_tiles = tl.cdiv(S, 128)
    
    for j in range(0, min(i + 1, num_tiles)):  
        s_offset_j = j * 128
        row_mask_j = (s_offset_j + row_idx) < S
        
        K_0 = load_tile(K_ptr, sh_offset, s_offset_j, row_idx, col_idx_0, row_mask_j, stride_s, stride_d, 0)
        K_1 = load_tile(K_ptr, sh_offset, s_offset_j, row_idx, col_idx_1, row_mask_j, stride_s, stride_d, 64)
        V_0 = load_tile(V_ptr, sh_offset, s_offset_j, row_idx, col_idx_0, row_mask_j, stride_s, stride_d, 0)
        V_1 = load_tile(V_ptr, sh_offset, s_offset_j, row_idx, col_idx_1, row_mask_j, stride_s, stride_d, 64)
        
        QK_T = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        
        P = tl.exp(QK_T * scale - L_i[:, None])
        
        col_idx_128 = tl.arange(0, 128)
        mask_causal = ((s_offset_i + row_idx[:, None]) >= (s_offset_j + col_idx_128[None, :]))
        P = P * row_mask_i[:, None] * row_mask_j[None, :] * mask_causal
        
        dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        D_P = tl.sum(P * dP, axis=1, keepdims=True)
        dS = P * (dP - D_P) * scale
        
        dQ_0 = tl.dot(dS, K_0, dQ_0)
        dQ_1 = tl.dot(dS, K_1, dQ_1)
        
    store_tile(dQ_ptr, sh_offset, s_offset_i, dQ_0.to(tl.bfloat16), row_mask_i, stride_s, stride_d, 0)
    store_tile(dQ_ptr, sh_offset, s_offset_i, dQ_1.to(tl.bfloat16), row_mask_i, stride_s, stride_d, 64)


@triton.jit
def _bwd_kernel_dkv(Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr, S,
                     stride_b, stride_h, stride_s, stride_d,
                     stride_bh_L, stride_h_L):
    """Compute dK and dV by iterating over key tiles and accumulating over subsequent query tiles."""
    j = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    sh_offset = b * stride_b + h * stride_h
    s_offset_j = j * 128
    
    row_idx = tl.arange(0, 128)
    col_idx_0 = tl.arange(0, 64)
    col_idx_1 = tl.arange(0, 64)
    
    row_mask_j = (s_offset_j + row_idx) < S
    
    K_0 = load_tile(K_ptr, sh_offset, s_offset_j, row_idx, col_idx_0, row_mask_j, stride_s, stride_d, 0)
    K_1 = load_tile(K_ptr, sh_offset, s_offset_j, row_idx, col_idx_1, row_mask_j, stride_s, stride_d, 64)
    V_0 = load_tile(V_ptr, sh_offset, s_offset_j, row_idx, col_idx_0, row_mask_j, stride_s, stride_d, 0)
    V_1 = load_tile(V_ptr, sh_offset, s_offset_j, row_idx, col_idx_1, row_mask_j, stride_s, stride_d, 64)
    
    scale = tl.full((1,), 1.0 / math.sqrt(128), dtype=tl.float32)
    
    dV_0 = tl.zeros((128, 64), dtype=tl.float32)
    dV_1 = tl.zeros((128, 64), dtype=tl.float32)
    dK_0 = tl.zeros((128, 64), dtype=tl.float32)
    dK_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    num_tiles = tl.cdiv(S, 128)
    
    for i in range(j, num_tiles):
        s_offset_i = i * 128
        row_mask_i = (s_offset_i + row_idx) < S
        
        Q_0 = load_tile(Q_ptr, sh_offset, s_offset_i, row_idx, col_idx_0, row_mask_i, stride_s, stride_d, 0)
        Q_1 = load_tile(Q_ptr, sh_offset, s_offset_i, row_idx, col_idx_1, row_mask_i, stride_s, stride_d, 64)
        dO_0 = load_tile(dO_ptr, sh_offset, s_offset_i, row_idx, col_idx_0, row_mask_i, stride_s, stride_d, 0)
        dO_1 = load_tile(dO_ptr, sh_offset, s_offset_i, row_idx, col_idx_1, row_mask_i, stride_s, stride_d, 64)
        
        L_i = tl.load(L_ptr + b * stride_bh_L + h * stride_h_L + s_offset_i + row_idx, mask=row_mask_i, other=0.0)
        
        QK_T = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        
        P = tl.exp(QK_T * scale - L_i[:, None])
        
        col_idx_128 = tl.arange(0, 128)
        mask_causal = ((s_offset_i + row_idx[:, None]) >= (s_offset_j + col_idx_128[None, :]))
        P = P * row_mask_i[:, None] * row_mask_j[None, :] * mask_causal
        
        dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        D_P = tl.sum(P * dP, axis=1, keepdims=True)
        dS = P * (dP - D_P) * scale
        
        P_T = P.T
        dS_T = dS.T
        
        dV_0 = tl.dot(P_T, dO_0, dV_0)
        dV_1 = tl.dot(P_T, dO_1, dV_1)
        dK_0 = tl.dot(dS_T, Q_0, dK_0)
        dK_1 = tl.dot(dS_T, Q_1, dK_1)
    
    store_tile(dV_ptr, sh_offset, s_offset_j, dV_0.to(tl.bfloat16), row_mask_j, stride_s, stride_d, 0)
    store_tile(dV_ptr, sh_offset, s_offset_j, dV_1.to(tl.bfloat16), row_mask_j, stride_s, stride_d, 64)
    store_tile(dK_ptr, sh_offset, s_offset_j, dK_0.to(tl.bfloat16), row_mask_j, stride_s, stride_d, 0)
    store_tile(dK_ptr, sh_offset, s_offset_j, dK_1.to(tl.bfloat16), row_mask_j, stride_s, stride_d, 64)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Compute the backward pass of causal multi-head attention.
    
    Inputs:
        Q: [B, H, S, d] bfloat16
        K: [B, H, S, d] bfloat16
        V: [B, H, S, d] bfloat16
        O: [B, H, S, d] bfloat16 (unused, compatibility artifact)
        dO: [B, H, S, d] bfloat16
        L: [B, H, S] float32
    
    Outputs (preallocated):
        dQ: [B, H, S, d] bfloat16
        dK: [B, H, S, d] bfloat16
        dV: [B, H, S, d] bfloat16
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    stride_b = H * S * d
    stride_h = S * d
    stride_s = d
    stride_d = 1
    
    stride_bh_L = H * S
    stride_h_L = S
    
    num_tiles = triton.cdiv(S, 128)
    grid = (num_tiles, H, B)
    
    # Launch both interdependent kernels sequentially on the active stream
    _bwd_kernel_dq[grid](
        Q, K, V, dO, L, dQ, S,
        stride_b, stride_h, stride_s, stride_d,
        stride_bh_L, stride_h_L,
        num_warps=8, num_stages=4
    )
    
    _bwd_kernel_dkv[grid](
        Q, K, V, dO, L, dK, dV, S,
        stride_b, stride_h, stride_s, stride_d,
        stride_bh_L, stride_h_L,
        num_warps=8, num_stages=4
    )