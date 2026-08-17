import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

BLOCK_SIZE = 64

# =========================================================================================================
# KERNEL 1: Computes dQ
# =========================================================================================================
@triton.jit
def _kernel_dQ(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr, dQ_ptr, S, scale
):
    s_i = tl.program_id(0)
    b_h = tl.program_id(1)
    
    row_idx = s_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    o_0 = desc_O.load([b_h * S + s_i * BLOCK_SIZE, 0]).to(tl.float32)
    o_1 = desc_O.load([b_h * S + s_i * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
    
    do_0 = desc_dO.load([b_h * S + s_i * BLOCK_SIZE, 0]).to(tl.float32)
    do_1 = desc_dO.load([b_h * S + s_i * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
    
    q_0 = desc_Q.load([b_h * S + s_i * BLOCK_SIZE, 0]).to(tl.float32)
    q_1 = desc_Q.load([b_h * S + s_i * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
    
    o_dot_do = (o_0 * do_0).sum(axis=1, keep_dims=True) + (o_1 * do_1).sum(axis=1, keep_dims=True)
    
    l_ptr = L_ptr + b_h * S
    l_val = tl.load(l_ptr + row_idx[:, None], mask=(row_idx[:, None] < S), other=0.0)
    
    dQ_acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
    
    for k in range(s_i + 1):
        col_idx_k = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        k_0 = desc_K.load([b_h * S + k * BLOCK_SIZE, 0]).to(tl.float32)
        k_1 = desc_K.load([b_h * S + k * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
        
        v_0 = desc_V.load([b_h * S + k * BLOCK_SIZE, 0]).to(tl.float32)
        v_1 = desc_V.load([b_h * S + k * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
        
        acc_s = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        acc_s = tl.dot(q_0, k_0.T, acc_s)
        acc_s = tl.dot(q_1, k_1.T, acc_s)
        
        acc_d = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        acc_d = tl.dot(do_0, v_0.T, acc_d)
        acc_d = tl.dot(do_1, v_1.T, acc_d)
        
        tile_mask = (row_idx[:, None] >= col_idx_k[None, :]) & (row_idx[:, None] < S) & (col_idx_k[None, :] < S)
        
        p_st = tl.exp(acc_s * scale - l_val) * tile_mask
        dp_st = p_st * (acc_d - o_dot_do)
        
        dQ_acc = tl.dot(dp_st, k_0, dQ_acc)
        dQ_acc = tl.dot(dp_st, k_1, dQ_acc)
        
    out_ptr = dQ_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx_k[None, :]
    tile_mask = (row_idx[:, None] < S) & (col_idx_k[None, :] < 128)
    tl.store(out_ptr, (dQ_acc * scale).to(tl.bfloat16), mask=tile_mask)


# =========================================================================================================
# KERNEL 2: Computes dK and dV
# =========================================================================================================
@triton.jit
def _kernel_dK_dV(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr, dK_ptr, dV_ptr, S, scale
):
    s_j = tl.program_id(0)
    b_h = tl.program_id(1)
    
    row_idx = s_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    k_0 = desc_K.load([b_h * S + s_j * BLOCK_SIZE, 0]).to(tl.float32)
    k_1 = desc_K.load([b_h * S + s_j * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
    
    v_0 = desc_V.load([b_h * S + s_j * BLOCK_SIZE, 0]).to(tl.float32)
    v_1 = desc_V.load([b_h * S + s_j * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
    
    dK_acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
    dV_acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
    
    num_S_tiles = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for i in range(s_j, num_S_tiles):
        row_idx_i = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        o_0 = desc_O.load([b_h * S + i * BLOCK_SIZE, 0]).to(tl.float32)
        o_1 = desc_O.load([b_h * S + i * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
        
        do_0 = desc_dO.load([b_h * S + i * BLOCK_SIZE, 0]).to(tl.float32)
        do_1 = desc_dO.load([b_h * S + i * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
        
        q_0 = desc_Q.load([b_h * S + i * BLOCK_SIZE, 0]).to(tl.float32)
        q_1 = desc_Q.load([b_h * S + i * BLOCK_SIZE, BLOCK_SIZE]).to(tl.float32)
        
        o_dot_do = (o_0 * do_0).sum(axis=1, keep_dims=True) + (o_1 * do_1).sum(axis=1, keep_dims=True)
        
        l_ptr = L_ptr + b_h * S
        l_val = tl.load(l_ptr + row_idx_i[:, None], mask=(row_idx_i[:, None] < S), other=0.0)
        
        acc_s = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        acc_s = tl.dot(q_0, k_0.T, acc_s)
        acc_s = tl.dot(q_1, k_1.T, acc_s)
        
        acc_d = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        acc_d = tl.dot(do_0, v_0.T, acc_d)
        acc_d = tl.dot(do_1, v_1.T, acc_d)
        
        tile_mask = (row_idx_i[:, None] >= row_idx[None, :]) & (row_idx_i[:, None] < S) & (row_idx[None, :] < S)
        
        p_st = tl.exp(acc_s * scale - l_val) * tile_mask
        dp_st = p_st * (acc_d - o_dot_do)
        
        dK_acc = tl.dot(dp_st.T, q_0, dK_acc)
        dK_acc = tl.dot(dp_st.T, q_1, dK_acc)
        
        dV_acc = tl.dot(p_st.T, do_0, dV_acc)
        dV_acc = tl.dot(p_st.T, do_1, dV_acc)
        
    col_idx_k = tl.arange(0, BLOCK_SIZE)
    out_ptr_dK = dK_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx_k[None, :]
    out_ptr_dV = dV_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx_k[None, :]
    tile_mask = (row_idx[:, None] < S) & (col_idx_k[None, :] < 128)
    tl.store(out_ptr_dK, (dK_acc * scale).to(tl.bfloat16), mask=tile_mask)
    tl.store(out_ptr_dV, dV_acc.to(tl.bfloat16), mask=tile_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    assert B == 4 and H == 48 and d == 128
    
    scale = 1.0 / math.sqrt(d)
    num_S_tiles = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    desc_Q = TensorDescriptor.from_tensor(Q.view(B * H * S, d), [BLOCK_SIZE, BLOCK_SIZE])
    desc_K = TensorDescriptor.from_tensor(K.view(B * H * S, d), [BLOCK_SIZE, BLOCK_SIZE])
    desc_V = TensorDescriptor.from_tensor(V.view(B * H * S, d), [BLOCK_SIZE, BLOCK_SIZE])
    desc_O = TensorDescriptor.from_tensor(O.view(B * H * S, d), [BLOCK_SIZE, BLOCK_SIZE])
    desc_dO = TensorDescriptor.from_tensor(dO.view(B * H * S, d), [BLOCK_SIZE, BLOCK_SIZE])
    
    grid_out = (num_S_tiles, B * H)
    _kernel_dQ[grid_out](desc_Q, desc_K, desc_V, desc_O, desc_dO, L, dQ, S, scale, num_warps=4, num_stages=3)
    _kernel_dK_dV[grid_out](desc_Q, desc_K, desc_V, desc_O, desc_dO, L, dK, dV, S, scale, num_warps=4, num_stages=3)