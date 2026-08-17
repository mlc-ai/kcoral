import math
import torch
import triton
import triton.language as tl


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr,
    S, alpha: tl.constexpr, d: tl.constexpr):
    
    q_tile = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    offset_m = q_tile * BLOCK_M
    
    desc_Q = tl.make_tensor_descriptor(Q_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    desc_K = tl.make_tensor_descriptor(K_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    desc_V = tl.make_tensor_descriptor(V_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    desc_dO = tl.make_tensor_descriptor(dO_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    desc_dQ = tl.make_tensor_descriptor(dQ_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    desc_L = tl.make_tensor_descriptor(L_ptr + bh_idx * S, shape=[S], strides=[1], block_shape=[BLOCK_M], padding_option="zero")
    
    acc_dQ0 = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    acc_dQ1 = tl.zeros((BLOCK_M, BLOCK_K), tl.float32)
    
    num_tiles = triton.cdiv(S, BLOCK_N)
    
    for kv_tile in range(num_tiles):
        offset_n = kv_tile * BLOCK_N
        
        q0 = desc_Q.load([offset_m, 0])
        q1 = desc_Q.load([offset_m, 64])
        
        do0 = desc_dO.load([offset_m, 0])
        do1 = desc_dO.load([offset_m, 64])
        
        lse = desc_L.load([offset_m])
        
        k0 = desc_K.load([offset_n, 0])
        k1 = desc_K.load([offset_n, 64])
        
        v0 = desc_V.load([offset_n, 0])
        v1 = desc_V.load([offset_n, 64])
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(q0, k0.T, s)
        s = tl.dot(q1, k1.T, s)
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(do0, v0.T, dp)
        dp = tl.dot(do1, v1.T, dp)
        
        p = tl.exp(s * alpha - lse[:, None])
        
        row_offsets_q = offset_m + tl.arange(0, BLOCK_M)
        col_offsets_kv = offset_n + tl.arange(0, BLOCK_N)
        mask = ((row_offsets_q[:, None] < S) & (col_offsets_kv[None, :] < S)).to(tl.float32)
        p = p * mask
        
        ds = p * dp
        
        acc_dQ0 = tl.dot(ds, k0, acc_dQ0)
        acc_dQ1 = tl.dot(ds, k1, acc_dQ1)
    
    dQ_desc = tl.make_tensor_descriptor(dQ_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    dQ_desc.store([bh_idx, offset_m, 0], acc_dQ0.to(tl.bfloat16) * alpha)
    dQ_desc.store([bh_idx, offset_m, 64], acc_dQ1.to(tl.bfloat16) * alpha)


@triton.jit
def _bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S, alpha: tl.constexpr, d: tl.constexpr):
    
    kv_tile = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    
    offset_n = kv_tile * BLOCK_N
    
    desc_Q = tl.make_tensor_descriptor(Q_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    desc_K = tl.make_tensor_descriptor(K_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    desc_V = tl.make_tensor_descriptor(V_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    desc_dO = tl.make_tensor_descriptor(dO_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    desc_dK = tl.make_tensor_descriptor(dK_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    desc_dV = tl.make_tensor_descriptor(dV_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    desc_L = tl.make_tensor_descriptor(L_ptr + bh_idx * S, shape=[S], strides=[1], block_shape=[BLOCK_M], padding_option="zero")
    
    k0 = desc_K.load([offset_n, 0])
    k1 = desc_K.load([offset_n, 64])
    
    v0 = desc_V.load([offset_n, 0])
    v1 = desc_V.load([offset_n, 64])
    
    acc_dK0 = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    acc_dK1 = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    
    acc_dV0 = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    acc_dV1 = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    
    num_tiles = triton.cdiv(S, BLOCK_M)
    
    for q_tile in range(num_tiles):
        offset_m = q_tile * BLOCK_M
        
        q0 = desc_Q.load([offset_m, 0])
        q1 = desc_Q.load([offset_m, 64])
        
        do0 = desc_dO.load([offset_m, 0])
        do1 = desc_dO.load([offset_m, 64])
        
        lse = desc_L.load([offset_m])
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(q0, k0.T, s)
        s = tl.dot(q1, k1.T, s)
        
        dp = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        dp = tl.dot(do0, v0.T, dp)
        dp = tl.dot(do1, v1.T, dp)
        
        p = tl.exp(s * alpha - lse[:, None])
        
        row_offsets_q = offset_m + tl.arange(0, BLOCK_M)
        col_offsets_kv = offset_n + tl.arange(0, BLOCK_N)
        mask = ((row_offsets_q[:, None] < S) & (col_offsets_kv[None, :] < S)).to(tl.float32)
        p = p * mask
        
        ds = p * dp
        
        ds_T = ds.T
        p_T = p.T
        
        acc_dV0 = tl.dot(p_T, do0, acc_dV0)
        acc_dV1 = tl.dot(p_T, do1, acc_dV1)
        
        acc_dK0 = tl.dot(ds_T, q0, acc_dK0)
        acc_dK1 = tl.dot(ds_T, q1, acc_dK1)
    
    dK_desc = tl.make_tensor_descriptor(dK_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    dV_desc = tl.make_tensor_descriptor(dV_ptr + bh_idx * S * d, shape=[S, d], strides=[d, 1], block_shape=[BLOCK_M, BLOCK_N])
    
    dK_desc.store([bh_idx, offset_n, 0], acc_dK0.to(tl.bfloat16) * alpha)
    dK_desc.store([bh_idx, offset_n, 64], acc_dK1.to(tl.bfloat16) * alpha)
    
    dV_desc.store([bh_idx, offset_n, 0], acc_dV0.to(tl.bfloat16))
    dV_desc.store([bh_idx, offset_n, 64], acc_dV1.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute backward pass of Multi-Head Attention."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    alpha = 1.0 / math.sqrt(d)
    
    Q_ptr = Q.data_ptr()
    K_ptr = K.data_ptr()
    V_ptr = V.data_ptr()
    dO_ptr = dO.data_ptr()
    L_ptr = L.data_ptr()
    dQ_ptr = dQ.data_ptr()
    dK_ptr = dK.data_ptr()
    dV_ptr = dV.data_ptr()
    
    num_tiles = triton.cdiv(S, 128)
    grid_dq = (num_tiles, B * H)
    grid_dkv = (num_tiles, B * H)
    
    _bwd_dq_kernel[grid_dq](
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dQ_ptr, S, alpha=alpha, d=d,
        num_stages=3,
    )
    
    _bwd_dkv_kernel[grid_dkv](
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr, S, alpha=alpha, d=d,
        num_stages=3,
    )