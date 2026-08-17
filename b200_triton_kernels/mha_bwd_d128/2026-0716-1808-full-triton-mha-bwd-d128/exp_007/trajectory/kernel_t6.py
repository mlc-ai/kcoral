import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


BLOCK = 64

@triton.jit(launch_bounds=min_ctas=1, max_ctas=4)
def bwd_dq_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dQ,
    L_ptr, S_len, scale
):
    i = tl.program_id(0)
    b_h_idx = tl.program_id(1)
    offset_m = i * BLOCK
    
    row = tl.arange(0, BLOCK)
    col = tl.arange(0, BLOCK)
    
    q0 = desc_Q.load((offset_m, 0))
    q1 = desc_Q.load((offset_m, 64))
    
    o0 = desc_O.load((offset_m, 0))
    o1 = desc_O.load((offset_m, 64))
    
    do0 = desc_dO.load((offset_m, 0))
    do1 = desc_dO.load((offset_m, 64))
    
    D_i = tl.sum(o0 * do0, axis=-1, keep_dims=True) + tl.sum(o1 * do1, axis=-1, keep_dims=True)
    
    l_val = tl.load(L_ptr + b_h_idx * S_len + offset_m + row, mask=(offset_m + row < S_len), other=0.0)
    
    dQ_acc0 = tl.zeros((BLOCK, BLOCK), tl.float32)
    dQ_acc1 = tl.zeros((BLOCK, BLOCK), tl.float32)
    
    num_tiles = tl.cdiv(S_len, BLOCK)
    
    for j in tl.range(0, num_tiles, warp_specialize=True, flatten=False):
        offset_n = j * BLOCK
        
        k0 = desc_K.load((offset_n, 0))
        k1 = desc_K.load((offset_n, 64))
        
        v0 = desc_V.load((offset_n, 0))
        v1 = desc_V.load((offset_n, 64))
        
        s = tl.dot(q0, k0) + tl.dot(q1, k1)
        
        row_exp = offset_m + row[:, None]
        col_exp = offset_n + col[None, :]
        mask_S = (row_exp < S_len) & (col_exp < S_len)
        s = s * mask_S.to(tl.float32)
        
        p = tl.exp(s * scale - l_val[:, None])
        p = p * mask_S.to(tl.float32)
        
        dp = tl.dot(do0, v0) + tl.dot(do1, v1)
        
        ds = p * (dp - D_i) * scale
        
        dQ_acc0 = tl.dot(ds, k0, acc=dQ_acc0)
        dQ_acc1 = tl.dot(ds, k1, acc=dQ_acc1)
        
    desc_dQ.store((offset_m, 0), dQ_acc0.to(tl.bfloat16))
    desc_dQ.store((offset_m, 64), dQ_acc1.to(tl.bfloat16))


@triton.jit(launch_bounds=min_ctas=1, max_ctas=4)
def bwd_dkv_kernel(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, desc_dK, desc_dV,
    L_ptr, S_len, scale
):
    j = tl.program_id(0)
    b_h_idx = tl.program_id(1)
    offset_n = j * BLOCK
    
    row = tl.arange(0, BLOCK)
    col = tl.arange(0, BLOCK)
    
    k0 = desc_K.load((offset_n, 0))
    k1 = desc_K.load((offset_n, 64))
    
    v0 = desc_V.load((offset_n, 0))
    v1 = desc_V.load((offset_n, 64))
    
    dK_acc0 = tl.zeros((BLOCK, BLOCK), tl.float32)
    dK_acc1 = tl.zeros((BLOCK, BLOCK), tl.float32)
    dV_acc0 = tl.zeros((BLOCK, BLOCK), tl.float32)
    dV_acc1 = tl.zeros((BLOCK, BLOCK), tl.float32)
    
    num_tiles = tl.cdiv(S_len, BLOCK)
    
    for i in tl.range(0, num_tiles, warp_specialize=True, flatten=False):
        offset_m = i * BLOCK
        
        q0 = desc_Q.load((offset_m, 0))
        q1 = desc_Q.load((offset_m, 64))
        
        o0 = desc_O.load((offset_m, 0))
        o1 = desc_O.load((offset_m, 64))
        
        do0 = desc_dO.load((offset_m, 0))
        do1 = desc_dO.load((offset_m, 64))
        
        D_i = tl.sum(o0 * do0, axis=-1, keep_dims=True) + tl.sum(o1 * do1, axis=-1, keep_dims=True)
        
        l_val = tl.load(L_ptr + b_h_idx * S_len + offset_m + row, mask=(offset_m + row < S_len), other=0.0)
        
        s = tl.dot(q0, k0) + tl.dot(q1, k1)
        
        row_exp = offset_m + row[:, None]
        col_exp = offset_n + col[None, :]
        mask_S = (row_exp < S_len) & (col_exp < S_len)
        s = s * mask_S.to(tl.float32)
        
        p = tl.exp(s * scale - l_val[:, None])
        p = p * mask_S.to(tl.float32)
        
        dp = tl.dot(do0, v0) + tl.dot(do1, v1)
        
        ds = p * (dp - D_i) * scale
        
        dK_acc0 = tl.dot(ds.T, q0, acc=dK_acc0)
        dK_acc1 = tl.dot(ds.T, q1, acc=dK_acc1)
        
        dV_acc0 = tl.dot(p.T, do0, acc=dV_acc0)
        dV_acc1 = tl.dot(p.T, do1, acc=dV_acc1)
        
    desc_dK.store((offset_n, 0), dK_acc0.to(tl.bfloat16))
    desc_dK.store((offset_n, 64), dK_acc1.to(tl.bfloat16))
    
    desc_dV.store((offset_n, 0), dV_acc0.to(tl.bfloat16))
    desc_dV.store((offset_n, 64), dV_acc1.to(tl.bfloat16))


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
        desc.block_shape = [BLOCK, BLOCK]
        return desc

    desc_Q = make_desc_2d(Q)
    desc_K = make_desc_2d(K)
    desc_V = make_desc_2d(V)
    desc_O = make_desc_2d(O)
    desc_dO = make_desc_2d(dO)
    desc_dQ = make_desc_2d(dQ)
    desc_dK = make_desc_2d(dK)
    desc_dV = make_desc_2d(dV)
    
    grid = (triton.cdiv(s_len, BLOCK), b * h)
    
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