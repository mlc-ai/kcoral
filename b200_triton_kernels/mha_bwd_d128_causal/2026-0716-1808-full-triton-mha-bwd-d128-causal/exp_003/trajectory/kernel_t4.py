import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _ds_kernel(Q_desc, K_desc, V_desc, dO_desc, L_ptr, P_ptr, dS_ptr, 
               S, stride_bh_L, stride_h_L, scale, num_tiles):
    """
    Phase 1 Kernel: Iterates over all causal (i, j) pairs.
    Computes P_ij and dS_ij, storing them to pre-allocated global memory buffers.
    """
    i = tl.program_id(0)
    j = tl.program_id(1)
    h = tl.program_id(2)
    b = tl.program_id(3)
    
    if j > i:
        return
    
    bh_offset = b * stride_bh_L + h * stride_h_L
    row_idx = tl.arange(0, 128)
    col_idx = tl.arange(0, 128)
    
    Q_0 = Q_desc.load((bh_offset + i * 128, 0))
    Q_1 = Q_desc.load((bh_offset + i * 128, 64))
    dO_0 = dO_desc.load((bh_offset + i * 128, 0))
    dO_1 = dO_desc.load((bh_offset + i * 128, 64))
    
    K_0 = K_desc.load((bh_offset + j * 128, 0))
    K_1 = K_desc.load((bh_offset + j * 128, 64))
    V_0 = V_desc.load((bh_offset + j * 128, 0))
    V_1 = V_desc.load((bh_offset + j * 128, 64))
    
    L_i = tl.load(L_ptr + bh_offset + i * 128 + row_idx, mask=(i * 128 + row_idx) < S, other=0.0)
    
    S_ij = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
    S_ij = S_ij * scale
    
    P_ij = tl.exp(S_ij - L_i[:, None])
    
    mask_causal = ((i * 128 + row_idx[:, None]) >= (j * 128 + col_idx[None, :]))
    P_ij = P_ij * mask_causal
    P_ij = P_ij * ((i * 128 + row_idx) < S)[:, None] * ((j * 128 + col_idx) < S)[None, :]
    
    dP_ij = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
    
    D_P_ij = tl.sum(P_ij * dP_ij, axis=1, keepdims=True)
    dS_ij = P_ij * (dP_ij - D_P_ij) * scale
    
    idx = (i * num_tiles + j) * 128
    P_desc.store((idx, 0), P_ij.to(tl.bfloat16))
    dS_desc.store((idx, 0), dS_ij.to(tl.bfloat16))


@triton.jit
def _dq_kernel(Q_desc, K_desc, dS_desc, dQ_desc, S, stride_bh_L, stride_h_L, num_tiles):
    """
    Phase 2 Kernel (dQ): Iterates over query tiles. Accumulates dQ_i over valid key tiles j (0 to i).
    """
    i = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    bh_offset = b * stride_bh_L + h * stride_h_L
    
    dQ_0 = tl.zeros((128, 64), dtype=tl.float32)
    dQ_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    for j in range(0, min(i + 1, num_tiles)):
        ds_idx = (i * num_tiles + j) * 128
        dS_ij = dS_desc.load((ds_idx, 0))
        
        K_0 = K_desc.load((bh_offset + j * 128, 0))
        K_1 = K_desc.load((bh_offset + j * 128, 64))
        
        dQ_0 = tl.dot(dS_ij, K_0, dQ_0)
        dQ_1 = tl.dot(dS_ij, K_1, dQ_1)
        
    dQ_desc.store((bh_offset + i * 128, 0), dQ_0.to(tl.bfloat16))
    dQ_desc.store((bh_offset + i * 128, 64), dQ_1.to(tl.bfloat16))


@triton.jit
def _dkv_kernel(Q_desc, K_desc, V_desc, dO_desc, P_desc, dS_desc, dK_desc, dV_desc,
                S, stride_bh_L, stride_h_L, num_tiles):
    """
    Phase 2 Kernel (dK, dV): Iterates over key tiles. Accumulates dK_j and dV_j over valid query tiles i (j to num_tiles-1).
    """
    j = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    bh_offset = b * stride_bh_L + h * stride_h_L
    
    dK_0 = tl.zeros((128, 64), dtype=tl.float32)
    dK_1 = tl.zeros((128, 64), dtype=tl.float32)
    dV_0 = tl.zeros((128, 64), dtype=tl.float32)
    dV_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    for i in range(j, num_tiles):
        ds_idx = (i * num_tiles + j) * 128
        
        dS_ij = dS_desc.load((ds_idx, 0))
        P_ij = P_desc.load((ds_idx, 0))
        
        dS_ij_T = dS_ij.T
        P_ij_T = P_ij.T
        
        Q_0 = Q_desc.load((bh_offset + i * 128, 0))
        Q_1 = Q_desc.load((bh_offset + i * 128, 64))
        dO_0 = dO_desc.load((bh_offset + i * 128, 0))
        dO_1 = dO_desc.load((bh_offset + i * 128, 64))
        
        dK_0 = tl.dot(dS_ij_T, Q_0, dK_0)
        dK_1 = tl.dot(dS_ij_T, Q_1, dK_1)
        
        dV_0 = tl.dot(P_ij_T, dO_0, dV_0)
        dV_1 = tl.dot(P_ij_T, dO_1, dV_1)
        
    dK_desc.store((bh_offset + j * 128, 0), dK_0.to(tl.bfloat16))
    dK_desc.store((bh_offset + j * 128, 64), dK_1.to(tl.bfloat16))
    dV_desc.store((bh_offset + j * 128, 0), dV_0.to(tl.bfloat16))
    dV_desc.store((bh_offset + j * 128, 64), dV_1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """
    Two-pass destination-passing causal Multi-Head Attention backward.
    Pass 1: Computes P and dS for all causal sequence block pairs.
    Pass 2: Accumulates dQ, dK, and dV leveraging the cached P and dS.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    
    stride_bh_L = H * S
    stride_h_L = S
    scale = 1.0 / math.sqrt(d)
    
    tile_d = 128
    tile_s = 128
    
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
    
    P_buf = torch.empty(num_tiles, num_tiles, tile_s, tile_s, device=Q.device, dtype=torch.bfloat16)
    dS_buf = torch.empty(num_tiles, num_tiles, tile_s, tile_s, device=Q.device, dtype=torch.bfloat16)
    
    P_flat = P_buf.view(num_tiles * num_tiles * tile_s, tile_s)
    dS_flat = dS_buf.view(num_tiles * num_tiles * tile_s, tile_s)
    
    P_desc = TensorDescriptor.from_tensor(P_flat, [tile_s, tile_s])
    dS_desc = TensorDescriptor.from_tensor(dS_flat, [tile_s, tile_s])
    
    grid_phase1 = (num_tiles, num_tiles, H, B)
    _ds_kernel[grid_phase1](
        Q_desc, K_desc, V_desc, dO_desc, L, P_buf, dS_buf,
        S, stride_bh_L, stride_h_L, scale, num_tiles,
        num_warps=4, num_stages=3
    )
    
    grid_phase2 = (num_tiles, H, B)
    _dq_kernel[grid_phase2](
        Q_desc, K_desc, dS_desc, dQ_desc,
        S, stride_bh_L, stride_h_L, num_tiles,
        num_warps=4, num_stages=3
    )
    
    _dkv_kernel[grid_phase2](
        Q_desc, K_desc, V_desc, dO_desc, P_desc, dS_desc, dK_desc, dV_desc,
        S, stride_bh_L, stride_h_L, num_tiles,
        num_warps=4, num_stages=3
    )