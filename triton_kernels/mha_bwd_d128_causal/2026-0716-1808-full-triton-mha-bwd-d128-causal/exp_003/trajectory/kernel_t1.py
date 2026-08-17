import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_kernel_dq(Q_desc, K_desc, V_desc, dO_desc, L_ptr, dQ_desc, S, stride_s):
    """Compute dQ by iterating over query tiles and their relevant key tiles."""
    i = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    row_idx = tl.arange(0, 128)
    row_mask_i = (i * 128 + row_idx) < S
    
    bh_offset = b * stride_s + h * S
    
    Q_0 = Q_desc.load((bh_offset + i * 128, 0))
    Q_1 = Q_desc.load((bh_offset + i * 128, 64))
    dO_0 = dO_desc.load((bh_offset + i * 128, 0))
    dO_1 = dO_desc.load((bh_offset + i * 128, 64))
    
    L_i = tl.load(L_ptr + bh_offset + i * 128 + row_idx, mask=row_mask_i, other=0.0)
    
    scale = 1.0 / math.sqrt(128)
    
    dQ_0 = tl.zeros((128, 64), dtype=tl.float32)
    dQ_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    for j in range(0, i + 1):  
        row_mask_j = (j * 128 + row_idx) < S
        
        K_0 = K_desc.load((bh_offset + j * 128, 0))
        K_1 = K_desc.load((bh_offset + j * 128, 64))
        V_0 = V_desc.load((bh_offset + j * 128, 0))
        V_1 = V_desc.load((bh_offset + j * 128, 64))
        
        QK_T = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        
        P = tl.exp(QK_T * scale - L_i[:, None])
        
        col_idx_128 = tl.arange(0, 128)
        mask_causal = ((i * 128 + row_idx[:, None]) >= (j * 128 + col_idx_128[None, :]))
        P = P * row_mask_i[:, None] * row_mask_j[None, :] * mask_causal
        
        dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        D_P = tl.sum(P * dP, axis=1, keepdims=True)
        dS = P * (dP - D_P) * scale
        
        dQ_0 = tl.dot(dS, K_0, dQ_0)
        dQ_1 = tl.dot(dS, K_1, dQ_1)
        
    dQ_desc.store((bh_offset + i * 128, 0), dQ_0)
    dQ_desc.store((bh_offset + i * 128, 64), dQ_1)


@triton.jit
def _bwd_kernel_dkv(Q_desc, K_desc, V_desc, dO_desc, L_ptr, dK_desc, dV_desc, S, stride_s):
    """Compute dK and dV by iterating over key tiles and accumulating over subsequent query tiles."""
    j = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    row_idx = tl.arange(0, 128)
    row_mask_j = (j * 128 + row_idx) < S
    
    bh_offset = b * stride_s + h * S
    
    K_0 = K_desc.load((bh_offset + j * 128, 0))
    K_1 = K_desc.load((bh_offset + j * 128, 64))
    V_0 = V_desc.load((bh_offset + j * 128, 0))
    V_1 = V_desc.load((bh_offset + j * 128, 64))
    
    scale = 1.0 / math.sqrt(128)
    num_tiles = tl.cdiv(S, 128)
    
    dV_0 = tl.zeros((128, 64), dtype=tl.float32)
    dV_1 = tl.zeros((128, 64), dtype=tl.float32)
    dK_0 = tl.zeros((128, 64), dtype=tl.float32)
    dK_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    for i in range(j, num_tiles):
        row_mask_i = (i * 128 + row_idx) < S
        
        Q_0 = Q_desc.load((bh_offset + i * 128, 0))
        Q_1 = Q_desc.load((bh_offset + i * 128, 64))
        dO_0 = dO_desc.load((bh_offset + i * 128, 0))
        dO_1 = dO_desc.load((bh_offset + i * 128, 64))
        
        L_i = tl.load(L_ptr + bh_offset + i * 128 + row_idx, mask=row_mask_i, other=0.0)
        
        QK_T = tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)
        
        P = tl.exp(QK_T * scale - L_i[:, None])
        
        col_idx_128 = tl.arange(0, 128)
        mask_causal = ((i * 128 + row_idx[:, None]) >= (j * 128 + col_idx_128[None, :]))
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
    
    dV_desc.store((bh_offset + j * 128, 0), dV_0)
    dV_desc.store((bh_offset + j * 128, 64), dV_1)
    dK_desc.store((bh_offset + j * 128, 0), dK_0)
    dK_desc.store((bh_offset + j * 128, 64), dK_1)


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
    
    stride_bh = H * S
    stride_h = S
    stride_s = d
    
    Q_desc = TensorDescriptor.from_tensor(Q.view(B * H * S, d), [128, 64])
    K_desc = TensorDescriptor.from_tensor(K.view(B * H * S, d), [128, 64])
    V_desc = TensorDescriptor.from_tensor(V.view(B * H * S, d), [128, 64])
    dO_desc = TensorDescriptor.from_tensor(dO.view(B * H * S, d), [128, 64])
    
    dQ_desc = TensorDescriptor.from_tensor(dQ.view(B * H * S, d), [128, 64])
    dK_desc = TensorDescriptor.from_tensor(dK.view(B * H * S, d), [128, 64])
    dV_desc = TensorDescriptor.from_tensor(dV.view(B * H * S, d), [128, 64])
    
    num_tiles = triton.cdiv(S, 128)
    grid = (num_tiles, H, B)
    
    _bwd_kernel_dq[grid](
        Q_desc, K_desc, V_desc, dO_desc, L, dQ_desc, S, stride_s,
        num_warps=8, num_stages=4
    )
    
    _bwd_kernel_dkv[grid](
        Q_desc, K_desc, V_desc, dO_desc, L, dK_desc, dV_desc, S, stride_s,
        num_warps=8, num_stages=4
    )