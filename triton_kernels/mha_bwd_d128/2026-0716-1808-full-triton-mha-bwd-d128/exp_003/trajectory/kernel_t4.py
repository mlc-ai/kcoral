import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor


def create_desc(ptr, shape_0):
    return TensorDescriptor(ptr, shape=[shape_0, 128], strides=[128, 1], block_shape=[128, 128], padding_option="zero")


@triton.jit
def load_fp32_slice(ptr, start_idx, limit):
    idx = tl.arange(0, 128)
    mask = idx < (limit - start_idx)
    return tl.load(ptr + start_idx + idx, mask=mask, other=0.0)


@triton.jit
def _compute_d_val(dO_desc, O_desc, d_val_ptr, S):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    do = dO_desc.load([bh * S + i * 128, 0])
    o = O_desc.load([bh * S + i * 128, 0])
    
    prod = do * o
    sum_val = prod.to(tl.float32).sum(axis=1)
    
    idx = tl.arange(0, 128)
    mask = idx < (S - i * 128)
    tl.store(d_val_ptr + bh * S + i * 128 + idx, sum_val, mask=mask)


@triton.jit
def _bwd_dq(Q_desc, K_desc, V_desc, dO_desc, d_val_ptr, L_ptr, dQ_desc, S, scale):
    i = tl.program_id(0)
    bh = tl.program_id(1)
    
    q = Q_desc.load([bh * S + i * 128, 0])
    do = dO_desc.load([bh * S + i * 128, 0])
    
    d_val = load_fp32_slice(d_val_ptr, bh * S + i * 128, S)
    l = load_fp32_slice(L_ptr, bh * S + i * 128, S)
    
    acc_dQ = tl.zeros((128, 128), tl.float32)
    
    row_idx = tl.arange(128)[:, None]
    col_idx = tl.arange(128)[None, :]
    
    for j in range(tl.cdiv(S, 128)):
        k = K_desc.load([bh * S + j * 128, 0])
        v = V_desc.load([bh * S + j * 128, 0])
        
        s_ij = tl.dot(q, k.T, acc=None)
        
        mask_ij = ((i * 128 + row_idx) < S) & ((j * 128 + col_idx) < S)
        p_ij = tl.exp(s_ij * scale - l[:, None])
        p_ij = p_ij * mask_ij
        
        acc_dP = tl.dot(do, v.T, acc=None)
        dp_ij = (acc_dP - d_val[:, None]) * p_ij
        
        acc_dQ = tl.dot(dp_ij, k, acc=acc_dQ)
        
    acc_dQ = acc_dQ * scale
    dQ_desc.store([bh * S + i * 128, 0], acc_dQ.to(tl.bfloat16))


@triton.jit
def _bwd_dk(Q_desc, K_desc, V_desc, dO_desc, d_val_ptr, L_ptr, dK_desc, S, scale):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    k = K_desc.load([bh * S + j * 128, 0])
    v = V_desc.load([bh * S + j * 128, 0])
    
    acc_dK = tl.zeros((128, 128), tl.float32)
    
    row_idx = tl.arange(128)[:, None]
    col_idx = tl.arange(128)[None, :]
    
    for i in range(tl.cdiv(S, 128)):
        q = Q_desc.load([bh * S + i * 128, 0])
        do = dO_desc.load([bh * S + i * 128, 0])
        
        d_val_i = load_fp32_slice(d_val_ptr, bh * S + i * 128, S)
        l_i = load_fp32_slice(L_ptr, bh * S + i * 128, S)
        
        s_ij = tl.dot(q, k.T, acc=None)
        
        mask_ij = ((i * 128 + row_idx) < S) & ((j * 128 + col_idx) < S)
        p_ij = tl.exp(s_ij * scale - l_i[:, None])
        p_ij = p_ij * mask_ij
        
        acc_dP = tl.dot(do, v.T, acc=None)
        dp_ij = (acc_dP - d_val_i[:, None]) * p_ij
        
        acc_dK = tl.dot(dp_ij.T, q, acc=acc_dK)
        
    acc_dK = acc_dK * scale
    dK_desc.store([bh * S + j * 128, 0], acc_dK.to(tl.bfloat16))


@triton.jit
def _bwd_dv(Q_desc, K_desc, V_desc, dO_desc, L_ptr, dV_desc, S, scale):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    
    k = K_desc.load([bh * S + j * 128, 0])
    
    acc_dV = tl.zeros((128, 128), tl.float32)
    
    row_idx = tl.arange(128)[:, None]
    col_idx = tl.arange(128)[None, :]
    
    for i in range(tl.cdiv(S, 128)):
        q = Q_desc.load([bh * S + i * 128, 0])
        do = dO_desc.load([bh * S + i * 128, 0])
        
        l_i = load_fp32_slice(L_ptr, bh * S + i * 128, S)
        
        s_ij = tl.dot(q, k.T, acc=None)
        
        mask_ij = ((i * 128 + row_idx) < S) & ((j * 128 + col_idx) < S)
        p_ij = tl.exp(s_ij * scale - l_i[:, None])
        p_ij = p_ij * mask_ij
        
        acc_dV = tl.dot(p_ij.T, do, acc=acc_dV)
        
    dV_desc.store([bh * S + j * 128, 0], acc_dV.to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    b, h, s, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    desc_Q = create_desc(Q.data_ptr(), b * h * s)
    desc_K = create_desc(K.data_ptr(), b * h * s)
    desc_V = create_desc(V.data_ptr(), b * h * s)
    desc_dO = create_desc(dO.data_ptr(), b * h * s)
    desc_O = create_desc(O.data_ptr(), b * h * s)
    
    desc_dQ = create_desc(dQ.data_ptr(), b * h * s)
    desc_dK = create_desc(dK.data_ptr(), b * h * s)
    desc_dV = create_desc(dV.data_ptr(), b * h * s)
    
    d_val = torch.empty(b * h, s, device=Q.device, dtype=torch.float32)
    
    grid_0 = (triton.cdiv(s, 128), b * h)
    _compute_d_val[grid_0](desc_dO, desc_O, d_val.data_ptr(), s, num_warps=4, num_stages=2)
    
    grid_1 = (triton.cdiv(s, 128), b * h)
    
    _bwd_dq[grid_1](
        desc_Q, desc_K, desc_V, desc_dO, d_val.data_ptr(), L.data_ptr(), desc_dQ,
        s, scale, num_warps=4, num_stages=3
    )
    
    _bwd_dk[grid_1](
        desc_Q, desc_K, desc_V, desc_dO, d_val.data_ptr(), L.data_ptr(), desc_dK,
        s, scale, num_warps=4, num_stages=3
    )
    
    _bwd_dv[grid_1](
        desc_Q, desc_K, desc_V, desc_dO, L.data_ptr(), desc_dV,
        s, scale, num_warps=4, num_stages=3
    )