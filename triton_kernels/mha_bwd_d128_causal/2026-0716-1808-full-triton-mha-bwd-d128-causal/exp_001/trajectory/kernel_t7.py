import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

BLOCK_SIZE = 64
HEAD_DIM = 128

# =========================================================================================================
# KERNEL 1: Computes dQ
# =========================================================================================================
@triton.jit
def _kernel_dQ(desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr, dQ_ptr, S, scale):
    s_i = tl.program_id(0)
    b_h = tl.program_id(1)
    
    row_idx = s_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    q_0 = desc_Q.load([b_h * S + s_i * BLOCK_SIZE, 0]).to(tl.float32)
    q_1 = desc_Q.load([b_h * S + s_i * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
    
    o_0 = desc_O.load([b_h * S + s_i * BLOCK_SIZE, 0]).to(tl.float32)
    o_1 = desc_O.load([b_h * S + s_i * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
    
    do_0 = desc_dO.load([b_h * S + s_i * BLOCK_SIZE, 0]).to(tl.float32)
    do_1 = desc_dO.load([b_h * S + s_i * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
    
    o_dot_do = (o_0 * do_0).sum(axis=1, keep_dims=True) + (o_1 * do_1).sum(axis=1, keep_dims=True)
    
    l_ptr = L_ptr + b_h * S
    l_val = tl.load(l_ptr + row_idx, mask=(row_idx < S), other=0.0)
    
    dQ_0 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
    dQ_1 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
    
    for s_k in range(0, s_i + 1):
        col_idx = s_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        k_0 = desc_K.load([b_h * S + s_k * BLOCK_SIZE, 0]).to(tl.float32)
        k_1 = desc_K.load([b_h * S + s_k * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
        
        v_0 = desc_V.load([b_h * S + s_k * BLOCK_SIZE, 0]).to(tl.float32)
        v_1 = desc_V.load([b_h * S + s_k * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
            
        S_st = tl.dot(q_0, k_0.T) + tl.dot(q_1, k_1.T)
        D_st = tl.dot(do_0, v_0.T) + tl.dot(do_1, v_1.T)
            
        tile_mask = (row_idx[:, None] >= col_idx[None, :]) & (row_idx[:, None] < S) & (col_idx[None, :] < S)
        
        P_st = tl.exp(S_st * scale - l_val[:, None]) * tile_mask
        
        dS_st = P_st * (D_st - o_dot_do)
        
        dQ_0 += tl.dot(dS_st, k_0)
        dQ_1 += tl.dot(dS_st, k_1)
            
    col_idx_0 = tl.arange(0, BLOCK_SIZE)
    out_ptr_dQ_0 = dQ_ptr + b_h * S * HEAD_DIM + row_idx[:, None] * HEAD_DIM + col_idx_0[None, :]
    tile_mask = (row_idx[:, None] < S) & (col_idx_0[None, :] < HEAD_DIM)
    tl.store(out_ptr_dQ_0, (dQ_0 * scale).to(tl.bfloat16), mask=tile_mask)
    
    col_idx_1 = BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_ptr_dQ_1 = dQ_ptr + b_h * S * HEAD_DIM + row_idx[:, None] * HEAD_DIM + col_idx_1[None, :]
    tl.store(out_ptr_dQ_1, (dQ_1 * scale).to(tl.bfloat16), mask=tile_mask)


# =========================================================================================================
# KERNEL 2: Computes dK and dV
# =========================================================================================================
@triton.jit
def _kernel_dK_dV(desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr, dK_ptr, dV_ptr, S, scale):
    s_j = tl.program_id(0)
    b_h = tl.program_id(1)
    
    row_idx = s_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    k_0 = desc_K.load([b_h * S + s_j * BLOCK_SIZE, 0]).to(tl.float32)
    k_1 = desc_K.load([b_h * S + s_j * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
    
    v_0 = desc_V.load([b_h * S + s_j * BLOCK_SIZE, 0]).to(tl.float32)
    v_1 = desc_V.load([b_h * S + s_j * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
    
    dK_0 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
    dK_1 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
    dV_0 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
    dV_1 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
    
    num_S_tiles = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for s_l in range(s_j, num_S_tiles):
        row_idx_l = s_l * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        q_0 = desc_Q.load([b_h * S + s_l * BLOCK_SIZE, 0]).to(tl.float32)
        q_1 = desc_Q.load([b_h * S + s_l * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
        
        o_0 = desc_O.load([b_h * S + s_l * BLOCK_SIZE, 0]).to(tl.float32)
        o_1 = desc_O.load([b_h * S + s_l * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
        
        do_0 = desc_dO.load([b_h * S + s_l * BLOCK_SIZE, 0]).to(tl.float32)
        do_1 = desc_dO.load([b_h * S + s_l * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
            
        o_dot_do = (o_0 * do_0).sum(axis=1, keep_dims=True) + (o_1 * do_1).sum(axis=1, keep_dims=True)
        
        l_ptr = L_ptr + b_h * S
        l_val = tl.load(l_ptr + row_idx_l, mask=(row_idx_l < S), other=0.0)
            
        S_st = tl.dot(q_0, k_0.T) + tl.dot(q_1, k_1.T)
        D_st = tl.dot(do_0, v_0.T) + tl.dot(do_1, v_1.T)
        
        tile_mask = (row_idx_l[:, None] >= row_idx[None, :]) & (row_idx_l[:, None] < S) & (row_idx[None, :] < S)
        
        P_st = tl.exp(S_st * scale - l_val[:, None]) * tile_mask
        
        dS_st = P_st * (D_st - o_dot_do)
            
        dK_0 += tl.dot(dS_st.T, q_0)
        dK_1 += tl.dot(dS_st.T, q_1)
        
        dV_0 += tl.dot(P_st.T, do_0)
        dV_1 += tl.dot(P_st.T, do_1)
            
    col_idx_0 = tl.arange(0, BLOCK_SIZE)
    out_ptr_dK_0 = dK_ptr + b_h * S * HEAD_DIM + row_idx[:, None] * HEAD_DIM + col_idx_0[None, :]
    out_ptr_dV_0 = dV_ptr + b_h * S * HEAD_DIM + row_idx[:, None] * HEAD_DIM + col_idx_0[None, :]
    tile_mask = (row_idx[:, None] < S) & (col_idx_0[None, :] < HEAD_DIM)
    tl.store(out_ptr_dK_0, (dK_0 * scale).to(tl.bfloat16), mask=tile_mask)
    tl.store(out_ptr_dV_0, dV_0.to(tl.bfloat16), mask=tile_mask)
    
    col_idx_1 = BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_ptr_dK_1 = dK_ptr + b_h * S * HEAD_DIM + row_idx[:, None] * HEAD_DIM + col_idx_1[None, :]
    out_ptr_dV_1 = dV_ptr + b_h * S * HEAD_DIM + row_idx[:, None] * HEAD_DIM + col_idx_1[None, :]
    tl.store(out_ptr_dK_1, (dK_1 * scale).to(tl.bfloat16), mask=tile_mask)
    tl.store(out_ptr_dV_1, dV_1.to(tl.bfloat16), mask=tile_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    assert B == 4 and H == 48 and d == 128
    
    scale = 1.0 / math.sqrt(d)
    num_S_tiles = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    desc_Q = TensorDescriptor.from_tensor(Q.view(B * H * S, HEAD_DIM), [BLOCK_SIZE, BLOCK_SIZE])
    desc_K = TensorDescriptor.from_tensor(K.view(B * H * S, HEAD_DIM), [BLOCK_SIZE, BLOCK_SIZE])
    desc_V = TensorDescriptor.from_tensor(V.view(B * H * S, HEAD_DIM), [BLOCK_SIZE, BLOCK_SIZE])
    desc_O = TensorDescriptor.from_tensor(O.view(B * H * S, HEAD_DIM), [BLOCK_SIZE, BLOCK_SIZE])
    desc_dO = TensorDescriptor.from_tensor(dO.view(B * H * S, HEAD_DIM), [BLOCK_SIZE, BLOCK_SIZE])
    
    grid_out = (num_S_tiles, B * H)
    
    _kernel_dQ[grid_out](desc_Q, desc_K, desc_V, desc_O, desc_dO, L.data_ptr(), dQ.data_ptr(), S, scale, num_warps=4, num_stages=4)
    _kernel_dK_dV[grid_out](desc_Q, desc_K, desc_V, desc_O, desc_dO, L.data_ptr(), dK.data_ptr(), dV.data_ptr(), S, scale, num_warps=4, num_stages=4)