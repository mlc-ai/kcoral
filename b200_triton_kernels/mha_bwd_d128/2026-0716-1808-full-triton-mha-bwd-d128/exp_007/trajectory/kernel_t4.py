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
    
    q = desc_Q.load([b_h_idx, offset_m, 0])
    o = desc_O.load([b_h_idx, offset_m, 0])
    do = desc_dO.load([b_h_idx, offset_m, 0])
    
    D_i = tl.sum(o * do, axis=-1, keep_dims=True)
    
    l_val = tl.load(L_ptr + b_h_idx * S_len + offset_m + row, mask=(offset_m + row < S_len), other=0.0)
    
    dQ_acc = tl.zeros((BLOCK, 128), tl.float32)
    
    for j in range(S_len // BLOCK):
        offset_n = j * BLOCK
        
        k = desc_K.load([b_h_idx, offset_n, 0])
        v = desc_V.load([b_h_idx, offset_n, 0])
        
        s = tl.dot(q, k.T, layout_nhwc=True)
        
        row_exp = offset_m + row[:, None]
        col_exp = offset_n + col[None, :]
        mask_S = (row_exp < S_len) & (col_exp < S_len)
        s = s * mask_S
        
        p = tl.exp(s * scale - l_val[:, None])
        
        dp = tl.dot(do, v.T, layout_nhwc=True)
        
        ds = p * (dp - D_i) * scale
        
        dQ_acc = tl.dot(ds, k, acc=dQ_acc, layout_nhwc=True)
        
    desc_dQ.store([b_h_idx, offset_m, 0], dQ_acc.to(tl.bfloat16))


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
    
    k = desc_K.load([b_h_idx, offset_n, 0])
    v = desc_V.load([b_h_idx, offset_n, 0])
    
    dK_acc = tl.zeros((BLOCK, 128), tl.float32)
    dV_acc = tl.zeros((BLOCK, 128), tl.float32)
    
    for i in range(S_len // BLOCK):
        offset_m = i * BLOCK
        
        q = desc_Q.load([b_h_idx, offset_m, 0])
        o = desc_O.load([b_h_idx, offset_m, 0])
        do = desc_dO.load([b_h_idx, offset_m, 0])
        
        D_i = tl.sum(o * do, axis=-1, keep_dims=True)
        
        l_val = tl.load(L_ptr + b_h_idx * S_len + offset_m + row, mask=(offset_m + row < S_len), other=0.0)
        
        s = tl.dot(q, k.T, layout_nhwc=True)
        
        row_exp = offset_m + row[:, None]
        col_exp = offset_n + col[None, :]
        mask_S = (row_exp < S_len) & (col_exp < S_len)
        s = s * mask_S
        
        p = tl.exp(s * scale - l_val[:, None])
        
        dp = tl.dot(do, v.T, layout_nhwc=True)
        
        ds = p * (dp - D_i) * scale
        
        dK_acc = tl.dot(ds.T, q, acc=dK_acc, layout_nhwc=True)
        dV_acc = tl.dot(p.T, do, acc=dV_acc, layout_nhwc=True)
        
    desc_dK.store([b_h_idx, offset_n, 0], dK_acc.to(tl.bfloat16))
    desc_dV.store([b_h_idx, offset_n, 0], dV_acc.to(tl.bfloat16))


NUM_WARPS = 4
NUM_STAGES = 3

def run(Q, K, V, O, dO, L, dQ, dK, dV):
    b, h, s_len, d = Q.shape
    device = Q.device
    torch.cuda.set_device(device)
    
    scale = 1.0 / math.sqrt(d)
    
    def make_desc(tensor):
        desc = TensorDescriptor.from_tensor(tensor, [B * H, S, 128])
        desc.block_shape = [1, BLOCK, 128]
        return desc

    desc_Q = make_desc(Q)
    desc_K = make_desc(K)
    desc_V = make_desc(V)
    desc_O = make_desc(O)
    desc_dO = make_desc(dO)
    desc_dQ = make_desc(dQ)
    desc_dK = make_desc(dK)
    desc_dV = make_desc(dV)
    
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