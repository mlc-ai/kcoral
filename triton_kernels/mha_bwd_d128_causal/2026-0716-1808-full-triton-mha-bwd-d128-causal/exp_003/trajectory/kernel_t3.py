import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def compute_dS_and_P(Q_tile, K_tile, V_tile, dO_tile, L_vec, scale, i, j, S, row_idx, col_idx):
    """
    Computes the attention backward quantities P_ij and dS_ij for a pair of 
    query tile i and key tile j.
    """
    S_ij = tl.dot(Q_tile, K_tile.T) * scale
    P_ij = tl.exp(S_ij - L_vec[:, None])
    
    mask_causal = ((i * 64 + row_idx[:, None]) >= (j * 64 + col_idx[None, :]))
    P_ij = P_ij * mask_causal
    P_ij = P_ij * ((i * 64 + row_idx) < S)[:, None] * ((j * 64 + col_idx) < S)[None, :]
    
    dP_ij = tl.dot(dO_tile, V_tile.T)
    D_P_ij = tl.sum(P_ij * dP_ij, axis=1, keepdims=True)
    dS_ij = P_ij * (dP_ij - D_P_ij) * scale
    
    return P_ij, dS_ij


@triton.jit
def _bwd_kernel_dq(Q_desc, K_desc, V_desc, dO_desc, L_ptr, dQ_desc, 
                   S, stride_bh_L, stride_h_L, scale, num_tiles):
    """
    Iterates over query tiles. For every query tile i, loops through key tiles j (0 to i).
    Uses static Q_i, dO_i, L_i and double-buffered dynamic K_j, V_j.
    """
    i = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    bh_offset = b * stride_bh_L + h * stride_h_L
    row_idx = tl.arange(0, 64)
    col_idx = tl.arange(0, 64)
    
    Q_i = Q_desc.load((bh_offset + i * 64, 0))
    dO_i = dO_desc.load((bh_offset + i * 64, 0))
    
    L_i = tl.load(L_ptr + bh_offset + i * 64 + row_idx, mask=(i * 64 + row_idx) < S, other=0.0)
    
    dQ_i = tl.zeros((64, 128), dtype=tl.float32)
    
    K_curr = K_desc.load((bh_offset + 0 * 64, 0))
    V_curr = V_desc.load((bh_offset + 0 * 64, 0))
    
    for j in range(0, min(i, num_tiles - 1)):
        K_next = K_desc.load((bh_offset + (j + 1) * 64, 0))
        V_next = V_desc.load((bh_offset + (j + 1) * 64, 0))
        
        _, dS_ij = compute_dS_and_P(Q_i, K_curr, V_curr, dO_i, L_i, scale, i, j, S, row_idx, col_idx)
        dQ_i = tl.dot(dS_ij, K_curr, dQ_i)
        
        K_curr = K_next
        V_curr = V_next
    
    last_j = min(i, num_tiles - 1)
    _, dS_ij = compute_dS_and_P(Q_i, K_curr, V_curr, dO_i, L_i, scale, i, last_j, S, row_idx, col_idx)
    dQ_i = tl.dot(dS_ij, K_curr, dQ_i)
    
    dQ_desc.store((bh_offset + i * 64, 0), dQ_i.to(tl.bfloat16))


@triton.jit
def _bwd_kernel_dkv(Q_desc, K_desc, V_desc, dO_desc, L_ptr, dK_desc, dV_desc,
                    S, stride_bh_L, stride_h_L, scale, num_tiles):
    """
    Iterates over key tiles. For every key tile j, loops through query tiles i (j to num_tiles-1).
    Uses static K_j, V_j and double-buffered dynamic Q_i, dO_i, L_i.
    """
    j = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    bh_offset = b * stride_bh_L + h * stride_h_L
    row_idx = tl.arange(0, 64)
    col_idx = tl.arange(0, 64)
    
    K_j = K_desc.load((bh_offset + j * 64, 0))
    V_j = V_desc.load((bh_offset + j * 64, 0))
    
    dK_j = tl.zeros((64, 128), dtype=tl.float32)
    dV_j = tl.zeros((64, 128), dtype=tl.float32)
    
    Q_curr = Q_desc.load((bh_offset + j * 64, 0))
    dO_curr = dO_desc.load((bh_offset + j * 64, 0))
    L_curr = tl.load(L_ptr + bh_offset + j * 64 + row_idx, mask=(j * 64 + row_idx) < S, other=0.0)
    
    for i in range(j, min(num_tiles - 1, num_tiles)):
        Q_next = Q_desc.load((bh_offset + (i + 1) * 64, 0))
        dO_next = dO_desc.load((bh_offset + (i + 1) * 64, 0))
        L_next = tl.load(L_ptr + bh_offset + (i + 1) * 64 + row_idx, mask=((i + 1) * 64 + row_idx) < S, other=0.0)
        
        P_ij, dS_ij = compute_dS_and_P(Q_curr, K_j, V_j, dO_curr, L_curr, scale, i, j, S, row_idx, col_idx)
        
        dV_j = tl.dot(P_ij.T, dO_curr, dV_j)
        dK_j = tl.dot(dS_ij.T, Q_curr, dK_j)
        
        Q_curr = Q_next
        dO_curr = dO_next
        L_curr = L_next
        
    last_i = min(num_tiles - 1, num_tiles)
    P_ij, dS_ij = compute_dS_and_P(Q_curr, K_j, V_j, dO_curr, L_curr, scale, last_i, j, S, row_idx, col_idx)
    
    dV_j = tl.dot(P_ij.T, dO_curr, dV_j)
    dK_j = tl.dot(dS_ij.T, Q_curr, dK_j)
    
    dK_desc.store((bh_offset + j * 64, 0), dK_j.to(tl.bfloat16))
    dV_desc.store((bh_offset + j * 64, 0), dV_j.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    stride_bh_L = H * S
    stride_h_L = S
    scale = 1.0 / math.sqrt(d)
    
    tile_d = 128
    tile_s = 64
    
    Q_flat = Q.view(B * H * S, d)
    K_flat = K.view(B * H * S, d)
    V_flat = V.view(B * H * S, d)
    dO_flat = dO.view(B * H * S, d)
    
    dQ_flat = dQ.view(B * H * S, d)
    dK_flat = dK.view(B * H * S, d)
    dV_flat = dV.view(B * H * S, d)
    
    Q_desc = TensorDescriptor.from_tensor(Q_flat, [tile_s, tile_d])
    K_desc = TensorDescriptor.from_tensor(K_flat, [tile_s, tile_d])
    V_desc = TensorDescriptor.from_tensor(V_flat, [tile_s, tile_d])
    dO_desc = TensorDescriptor.from_tensor(dO_flat, [tile_s, tile_d])
    
    dQ_desc = TensorDescriptor.from_tensor(dQ_flat, [tile_s, tile_d])
    dK_desc = TensorDescriptor.from_tensor(dK_flat, [tile_s, tile_d])
    dV_desc = TensorDescriptor.from_tensor(dV_flat, [tile_s, tile_d])
    
    num_tiles = triton.cdiv(S, tile_s)
    grid = (num_tiles, H, B)
    
    _bwd_kernel_dq[grid](
        Q_desc, K_desc, V_desc, dO_desc, L, dQ_desc,
        S, stride_bh_L, stride_h_L, scale, num_tiles,
        num_warps=4, num_stages=3
    )
    
    _bwd_kernel_dkv[grid](
        Q_desc, K_desc, V_desc, dO_desc, L, dK_desc, dV_desc,
        S, stride_bh_L, stride_h_L, scale, num_tiles,
        num_warps=4, num_stages=3
    )