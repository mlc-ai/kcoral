import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

BLOCK_SIZE = 32

# =========================================================================================================
# FUSED KERNEL: Computes dQ, dK, dV
# =========================================================================================================
@triton.jit
def _kernel_mha_bwd(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr,
    dQ_ptr, dK_ptr, dV_ptr, S, scale
):
    s_i = tl.program_id(0)
    b_h = tl.program_id(1)
    
    row_idx = s_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    q = []
    do = []
    o = []
    for step in range(4):
        q.append(desc_Q.load([b_h, s_i * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
        do.append(desc_dO.load([b_h, s_i * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
        o.append(desc_O.load([b_h, s_i * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
    
    o_dot_do = (o[0] * do[0]).sum(axis=1, keep_dims=True)
    o_dot_do += (o[1] * do[1]).sum(axis=1, keep_dims=True)
    o_dot_do += (o[2] * do[2]).sum(axis=1, keep_dims=True)
    o_dot_do += (o[3] * do[3]).sum(axis=1, keep_dims=True)
    
    l_ptr = L_ptr + b_h * S
    l_val = tl.load(l_ptr + row_idx[:, None], mask=(row_idx[:, None] < S), other=0.0)
    
    dQ_0 = [tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32) for _ in range(4)]
    dQ_1 = [tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32) for _ in range(4)]
    dK = [tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32) for _ in range(4)]
    dV = [tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32) for _ in range(4)]
    
    num_S_tiles = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for j in range(num_S_tiles):
        col_idx_j = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        k = []
        v = []
        for step in range(4):
            k.append(desc_K.load([b_h, j * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
            v.append(desc_V.load([b_h, j * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
        
        acc_s = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        acc_d = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        for step in range(4):
            acc_s = tl.dot(q[step], k[step].T, acc_s)
            acc_d = tl.dot(do[step], v[step].T, acc_d)
            
        tile_mask = (row_idx[:, None] < S) & (col_idx_j[None, :] < S) & (row_idx[:, None] >= col_idx_j[None, :])
        p_st = tl.exp(acc_s * scale - l_val) * tile_mask
        
        dP_st = p_st * (acc_d - o_dot_do)
        
        if j < s_i:
            for step in range(4):
                dQ_0[step] = tl.dot(dP_st, k[step], dQ_0[step])
        
        if j >= s_i:
            for step in range(4):
                dQ_1[step] = tl.dot(dP_st, k[step], dQ_1[step])
                dK[step] = tl.dot(dP_st.T, q[step], dK[step])
                dV[step] = tl.dot(p_st.T, do[step], dV[step])
                
    for step in range(4):
        col_idx_k = step * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tile_mask = (row_idx[:, None] < S) & (col_idx_k[None, :] < 128)
        
        out_ptr_dQ = dQ_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx_k[None, :]
        tl.store(out_ptr_dQ, ((dQ_0[step] + dQ_1[step]) * scale).to(tl.bfloat16), mask=tile_mask)
        
        out_ptr_dK = dK_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx_k[None, :]
        tl.store(out_ptr_dK, (dK[step] * scale).to(tl.bfloat16), mask=tile_mask)
        
        out_ptr_dV = dV_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx_k[None, :]
        tl.store(out_ptr_dV, dV[step].to(tl.bfloat16), mask=tile_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    assert B == 4 and H == 48 and d == 128
    
    scale = 1.0 / math.sqrt(d)
    num_S_tiles = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    desc_Q = TensorDescriptor.from_tensor(Q.view(B * H, S, d), [1, BLOCK_SIZE, BLOCK_SIZE])
    desc_K = TensorDescriptor.from_tensor(K.view(B * H, S, d), [1, BLOCK_SIZE, BLOCK_SIZE])
    desc_V = TensorDescriptor.from_tensor(V.view(B * H, S, d), [1, BLOCK_SIZE, BLOCK_SIZE])
    desc_O = TensorDescriptor.from_tensor(O.view(B * H, S, d), [1, BLOCK_SIZE, BLOCK_SIZE])
    desc_dO = TensorDescriptor.from_tensor(dO.view(B * H, S, d), [1, BLOCK_SIZE, BLOCK_SIZE])
    
    grid = (num_S_tiles, B * H)
    _kernel_mha_bwd[grid](desc_Q, desc_K, desc_V, desc_O, desc_dO, L, dQ, dK, dV, S, scale, num_warps=8, num_stages=3)