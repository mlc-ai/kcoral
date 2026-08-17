import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


BLOCK = 64

@triton.jit
def bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dQ,
    L_ptr, S_len, scale
):
    i = tl.program_id(0)
    b_h_idx = tl.program_id(1)
    offset_m = i * BLOCK
    
    row = tl.arange(0, BLOCK)
    col = tl.arange(0, BLOCK)
    
    row_off = b_h_idx * S_len + offset_m
    
    q0 = desc_Q.load([row_off, 0])
    q1 = desc_Q.load([row_off, 64])
    
    o0 = desc_O.load([row_off, 0])
    o1 = desc_O.load([row_off, 64])
    
    do0 = desc_dO.load([row_off, 0])
    do1 = desc_dO.load([row_off, 64])
    
    D_i = tl.sum(o0 * do0, axis=-1, keep_dims=True) + tl.sum(o1 * do1, axis=-1, keep_dims=True)
    
    l_val = tl.load(L_ptr + b_h_idx * S_len + offset_m + row, mask=(offset_m + row < S_len), other=0.0)
    
    dQ_acc0 = tl.zeros((BLOCK, 64), tl.float32)
    dQ_acc1 = tl.zeros((BLOCK, 64), tl.float32)
    
    cur_k = [desc_K, desc_K]
    cur_v = [desc_V, desc_V]
    next_k = [desc_K, desc_K]
    next_v = [desc_V, desc_V]
    
    if S_len // BLOCK > 0:
        n_row_off = b_h_idx * S_len + 0
        k0 = cur_k[0].load([n_row_off, 0])
        k1 = cur_k[1].load([n_row_off, 64])
        v0 = cur_v[0].load([n_row_off, 0])
        v1 = cur_v[1].load([n_row_off, 64])

    for j in range(S_len // BLOCK):
        offset_n = j * BLOCK
        n_row_off = b_h_idx * S_len + offset_n
        
        if j + 1 < S_len // BLOCK:
            next_n_row_off = b_h_idx * S_len + (j + 1) * BLOCK
            k0 = next_k[0].load([next_n_row_off, 0])
            k1 = next_k[1].load([next_n_row_off, 64])
            v0 = next_v[0].load([next_n_row_off, 0])
            v1 = next_v[1].load([next_n_row_off, 64])
            
        row_exp = offset_m + row[:, None]
        col_exp = offset_n + col[None, :]
        mask_S = ((row_exp < S_len) & (col_exp < S_len)).to(tl.float32)
        
        s = tl.dot(q0, cur_k[0].T) + tl.dot(q1, cur_k[1].T)
        s = s * mask_S
        
        p = tl.exp(s * scale - l_val[:, None])
        p = p * mask_S
        
        dp = tl.dot(do0, cur_v[0].T) + tl.dot(do1, cur_v[1].T)
        
        ds = p * (dp - D_i) * scale
        
        dQ_acc0 = tl.dot(ds.to(tl.bfloat16), cur_k[0], acc=dQ_acc0)
        dQ_acc1 = tl.dot(ds.to(tl.bfloat16), cur_k[1], acc=dQ_acc1)
        
        cur_k, next_k = next_k, cur_k
        cur_v, next_v = next_v, cur_v
        
    desc_dQ.store([row_off, 0], dQ_acc0.to(tl.bfloat16))
    desc_dQ.store([row_off, 64], dQ_acc1.to(tl.bfloat16))


@triton.jit
def bwd_dkv_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
    L_ptr, S_len, scale
):
    j = tl.program_id(0)
    b_h_idx = tl.program_id(1)
    offset_n = j * BLOCK
    
    row = tl.arange(0, BLOCK)
    col = tl.arange(0, BLOCK)
    
    n_row_off = b_h_idx * S_len + offset_n
    
    k0 = desc_K.load([n_row_off, 0])
    k1 = desc_K.load([n_row_off, 64])
    
    v0 = desc_V.load([n_row_off, 0])
    v1 = desc_V.load([n_row_off, 64])
    
    dK_acc0 = tl.zeros((BLOCK, 64), tl.float32)
    dK_acc1 = tl.zeros((BLOCK, 64), tl.float32)
    dV_acc0 = tl.zeros((BLOCK, 64), tl.float32)
    dV_acc1 = tl.zeros((BLOCK, 64), tl.float32)
    
    cur_q = [desc_Q, desc_Q]
    cur_o = [desc_O, desc_O]
    cur_do = [desc_dO, desc_dO]
    next_q = [desc_Q, desc_Q]
    next_o = [desc_O, desc_O]
    next_do = [desc_dO, desc_dO]
    
    if S_len // BLOCK > 0:
        i_row_off = b_h_idx * S_len + 0
        q0 = cur_q[0].load([i_row_off, 0])
        q1 = cur_q[1].load([i_row_off, 64])
        o0 = cur_o[0].load([i_row_off, 0])
        o1 = cur_o[1].load([i_row_off, 64])
        do0 = cur_do[0].load([i_row_off, 0])
        do1 = cur_do[1].load([i_row_off, 64])

    for i in range(S_len // BLOCK):
        offset_m = i * BLOCK
        i_row_off = b_h_idx * S_len + offset_m
        
        if i + 1 < S_len // BLOCK:
            next_i_row_off = b_h_idx * S_len + (i + 1) * BLOCK
            q0 = next_q[0].load([next_i_row_off, 0])
            q1 = next_q[1].load([next_i_row_off, 64])
            o0 = next_o[0].load([next_i_row_off, 0])
            o1 = next_o[1].load([next_i_row_off, 64])
            do0 = next_do[0].load([next_i_row_off, 0])
            do1 = next_do[1].load([next_i_row_off, 64])
            
        D_i = tl.sum(cur_o[0] * cur_do[0], axis=-1, keep_dims=True) + tl.sum(cur_o[1] * cur_do[1], axis=-1, keep_dims=True)
        
        l_val = tl.load(L_ptr + b_h_idx * S_len + offset_m + row, mask=(offset_m + row < S_len), other=0.0)
        
        row_exp = offset_m + row[:, None]
        col_exp = offset_n + col[None, :]
        mask_S = ((row_exp < S_len) & (col_exp < S_len)).to(tl.float32)
        
        s = tl.dot(cur_q[0], k0.T) + tl.dot(cur_q[1], k1.T)
        s = s * mask_S
        
        p = tl.exp(s * scale - l_val[:, None])
        p = p * mask_S
        
        dp = tl.dot(cur_do[0], v0.T) + tl.dot(cur_do[1], v1.T)
        
        ds = p * (dp - D_i) * scale
        
        dK_acc0 = tl.dot(ds.to(tl.bfloat16).T, cur_q[0], acc=dK_acc0)
        dK_acc1 = tl.dot(ds.to(tl.bfloat16).T, cur_q[1], acc=dK_acc1)
        
        dV_acc0 = tl.dot(p.to(tl.bfloat16).T, cur_do[0], acc=dV_acc0)
        dV_acc1 = tl.dot(p.to(tl.bfloat16).T, cur_do[1], acc=dV_acc1)
        
        cur_q, next_q = next_q, cur_q
        cur_o, next_o = next_o, cur_o
        cur_do, next_do = next_do, cur_do
        
    desc_dK.store([n_row_off, 0], dK_acc0.to(tl.bfloat16))
    desc_dK.store([n_row_off, 64], dK_acc1.to(tl.bfloat16))
    
    desc_dV.store([n_row_off, 0], dV_acc0.to(tl.bfloat16))
    desc_dV.store([n_row_off, 64], dV_acc1.to(tl.bfloat16))


NUM_WARPS = 4
NUM_STAGES = 3

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    b, h, s_len, d = Q.shape
    device = Q.device
    torch.cuda.set_device(device)
    
    scale = 1.0 / math.sqrt(d)
    
    def make_desc_2d(tensor):
        t_2d = tensor.reshape(b * h * s_len, d)
        desc = TensorDescriptor.from_tensor(t_2d, [BLOCK, 64])
        return desc

    desc_Q = make_desc_2d(Q)
    desc_K = make_desc_2d(K)
    desc_V = make_desc_2d(V)
    desc_O = make_desc_2d(O)
    desc_dO = make_desc_2d(dO)
    desc_dQ = make_desc_2d(dQ)
    desc_dK = make_desc_2d(dK)
    desc_dV = make_desc_2d(dV)
    
    grid = (S_len // BLOCK, b * h)
    
    bwd_dq_kernel[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dQ,
        L, s_len, scale,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    
    bwd_dkv_kernel[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
        L, s_len, scale,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )