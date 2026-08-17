import math
import torch
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _bwd_dq_kernel(
    Q_desc, K_desc, V_desc, dO_desc, L_desc, dQ_desc,
    S, alpha: tl.constexpr,
):
    q_tile = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    seq_idx = q_tile * 128
    
    q0 = Q_desc.load([bh_idx, seq_idx, 0]).squeeze(0)
    q1 = Q_desc.load([bh_idx, seq_idx, 64]).squeeze(0)
    
    do0 = dO_desc.load([bh_idx, seq_idx, 0]).squeeze(0)
    do1 = dO_desc.load([bh_idx, seq_idx, 64]).squeeze(0)
    
    lse = L_desc.load([bh_idx, seq_idx]).squeeze(0)
    
    row_offsets = seq_idx + tl.arange(0, 128)
    
    acc_dQ0 = tl.zeros((128, 64), tl.float32)
    acc_dQ1 = tl.zeros((128, 64), tl.float32)
    
    num_tiles = triton.cdiv(S, 128)
    
    for kv_tile in range(num_tiles):
        kv_seq_idx = kv_tile * 128
        
        k0 = K_desc.load([bh_idx, kv_seq_idx, 0]).squeeze(0)
        k1 = K_desc.load([bh_idx, kv_seq_idx, 64]).squeeze(0)
        
        v0 = V_desc.load([bh_idx, kv_seq_idx, 0]).squeeze(0)
        v1 = V_desc.load([bh_idx, kv_seq_idx, 64]).squeeze(0)
        
        col_offsets = kv_seq_idx + tl.arange(0, 128)
        
        acc_S = tl.zeros((128, 128), tl.float32)
        acc_S = tl.dot(q0, k0.T, acc_S)
        acc_S = tl.dot(q1, k1.T, acc_S)
        
        s = acc_S * alpha
        
        p = tl.exp(s - lse[:, None])
        
        p *= (row_offsets[:, None] < S) & (col_offsets[None, :] < S)
        
        acc_dP = tl.zeros((128, 128), tl.float32)
        acc_dP = tl.dot(do0, v0.T, acc_dP)
        acc_dP = tl.dot(do1, v1.T, acc_dP)
        
        ds = p * acc_dP
        
        acc_dQ0 = tl.dot(ds, k0, acc_dQ0)
        acc_dQ1 = tl.dot(ds, k1, acc_dQ1)
    
    dQ_desc.store([bh_idx, seq_idx, 0], acc_dQ0.to(tl.bfloat16) * alpha)
    dQ_desc.store([bh_idx, seq_idx, 64], acc_dQ1.to(tl.bfloat16) * alpha)


@triton.jit
def _bwd_dkv_kernel(
    Q_desc, K_desc, V_desc, dO_desc, L_desc, dK_desc, dV_desc,
    S, alpha: tl.constexpr,
):
    kv_tile = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    kv_seq_idx = kv_tile * 128
    
    k0 = K_desc.load([bh_idx, kv_seq_idx, 0]).squeeze(0)
    k1 = K_desc.load([bh_idx, kv_seq_idx, 64]).squeeze(0)
    
    v0 = V_desc.load([bh_idx, kv_seq_idx, 0]).squeeze(0)
    v1 = V_desc.load([bh_idx, kv_seq_idx, 64]).squeeze(0)
    
    acc_dK0 = tl.zeros((128, 64), tl.float32)
    acc_dK1 = tl.zeros((128, 64), tl.float32)
    
    acc_dV0 = tl.zeros((128, 64), tl.float32)
    acc_dV1 = tl.zeros((128, 64), tl.float32)
    
    num_tiles = triton.cdiv(S, 128)
    
    for q_tile in range(num_tiles):
        seq_idx = q_tile * 128
        
        q0 = Q_desc.load([bh_idx, seq_idx, 0]).squeeze(0)
        q1 = Q_desc.load([bh_idx, seq_idx, 64]).squeeze(0)
        
        do0 = dO_desc.load([bh_idx, seq_idx, 0]).squeeze(0)
        do1 = dO_desc.load([bh_idx, seq_idx, 64]).squeeze(0)
        
        lse = L_desc.load([bh_idx, seq_idx]).squeeze(0)
        
        row_offsets = seq_idx + tl.arange(0, 128)
        col_offsets = kv_seq_idx + tl.arange(0, 128)
        
        acc_S = tl.zeros((128, 128), tl.float32)
        acc_S = tl.dot(q0, k0.T, acc_S)
        acc_S = tl.dot(q1, k1.T, acc_S)
        
        s = acc_S * alpha
        
        p = tl.exp(s - lse[:, None])
        
        p *= (row_offsets[:, None] < S) & (col_offsets[None, :] < S)
        
        acc_dP = tl.zeros((128, 128), tl.float32)
        acc_dP = tl.dot(do0, v0.T, acc_dP)
        acc_dP = tl.dot(do1, v1.T, acc_dP)
        
        ds = p * acc_dP
        
        ds_T = ds.T
        p_T = p.T
        
        acc_dV0 = tl.dot(p_T, do0, acc_dV0)
        acc_dV1 = tl.dot(p_T, do1, acc_dV1)
        
        acc_dK0 = tl.dot(ds_T, q0, acc_dK0)
        acc_dK1 = tl.dot(ds_T, q1, acc_dK1)
    
    dK_desc.store([bh_idx, kv_seq_idx, 0], acc_dK0.to(tl.bfloat16) * alpha)
    dK_desc.store([bh_idx, kv_seq_idx, 64], acc_dK1.to(tl.bfloat16) * alpha)
    
    dV_desc.store([bh_idx, kv_seq_idx, 0], acc_dV0.to(tl.bfloat16))
    dV_desc.store([bh_idx, kv_seq_idx, 64], acc_dV1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward pass of Multi-Head Attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    alpha = 1.0 / math.sqrt(d)
    
    Q_bh = Q.view(B * H, S, d)
    K_bh = K.view(B * H, S, d)
    V_bh = V.view(B * H, S, d)
    dO_bh = dO.view(B * H, S, d)
    dQ_bh = dQ.view(B * H, S, d)
    dK_bh = dK.view(B * H, S, d)
    dV_bh = dV.view(B * H, S, d)
    
    Q_desc = TensorDescriptor.from_tensor(Q_bh, block_shape=[1, 128, 64])
    K_desc = TensorDescriptor.from_tensor(K_bh, block_shape=[1, 128, 64])
    V_desc = TensorDescriptor.from_tensor(V_bh, block_shape=[1, 128, 64])
    dO_desc = TensorDescriptor.from_tensor(dO_bh, block_shape=[1, 128, 64])
    dQ_desc = TensorDescriptor.from_tensor(dQ_bh, block_shape=[1, 128, 64])
    dK_desc = TensorDescriptor.from_tensor(dK_bh, block_shape=[1, 128, 64])
    dV_desc = TensorDescriptor.from_tensor(dV_bh, block_shape=[1, 128, 64])
    
    L_bh = L.view(B * H, S)
    L_desc = TensorDescriptor.from_tensor(L_bh, block_shape=[1, 128])
    
    num_tiles = triton.cdiv(S, 128)
    grid_dq = (num_tiles, B * H)
    grid_dkv = (num_tiles, B * H)
    
    _bwd_dq_kernel[grid_dq](
        Q_desc, K_desc, V_desc, dO_desc, L_desc, dQ_desc,
        S, alpha=alpha,
        num_warps=4, num_stages=3,
    )
    
    _bwd_dkv_kernel[grid_dkv](
        Q_desc, K_desc, V_desc, dO_desc, L_desc, dK_desc, dV_desc,
        S, alpha=alpha,
        num_warps=4, num_stages=3,
    )