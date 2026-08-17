import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

BLOCK_SIZE = 32

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
    
    for k in range(0, 4):
        col_idx = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tile_mask = (row_idx[:, None] < S) & (col_idx[None, :] < 128)
        
        q = desc_Q.load([b_h * S + s_i * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
        do = desc_dO.load([b_h * S + s_i * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
        o = desc_O.load([b_h * S + s_i * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
        
        o_dot_do = (o * do).sum(axis=1, keep_dims=True)
        
        acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        for j in range(0, s_i + 1):
            col_idx_j = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            tile_mask_j = (row_idx[:, None] < S) & (col_idx_j[None, :] < S) & (row_idx[:, None] >= col_idx_j[None, :])
            
            k_tile = desc_K.load([b_h * S + j * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
            v_tile = desc_V.load([b_h * S + j * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
            
            s_st = q @ k_tile.T
            d_st = do @ v_tile.T
            
            l_ptr = L_ptr + b_h * S
            l_val = tl.load(l_ptr + row_idx[:, None], mask=(row_idx[:, None] < S), other=0.0)
            
            p_st = tl.exp(s_st * scale - l_val) * tile_mask_j
            ds_st = p_st * (d_st - o_dot_do)
            
            acc += ds_st @ k_tile * scale
        
        out_ptr = dQ_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx[None, :]
        tl.store(out_ptr, acc.to(tl.bfloat16), mask=tile_mask)


# =========================================================================================================
# KERNEL 2: Computes dK
# =========================================================================================================
@triton.jit
def _kernel_dK(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr, dK_ptr, S, scale
):
    s_j = tl.program_id(0)
    b_h = tl.program_id(1)
    
    row_idx = s_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    for k in range(0, 4):
        col_idx = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tile_mask = (row_idx[:, None] < S) & (col_idx[None, :] < 128)
        
        k_tile = desc_K.load([b_h * S + s_j * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
        v_tile = desc_V.load([b_h * S + s_j * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
        
        acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        for i in range(s_j, (S + BLOCK_SIZE - 1) // BLOCK_SIZE):
            row_idx_i = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            tile_mask_i = (row_idx_i[:, None] < S) & (row_idx_i[:, None] >= row_idx[None, :])
            
            q_tile = desc_Q.load([b_h * S + i * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
            do_tile = desc_dO.load([b_h * S + i * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
            o_tile = desc_O.load([b_h * S + i * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
            
            o_dot_do = (o_tile * do_tile).sum(axis=1, keep_dims=True)
            
            s_st = q_tile @ k_tile.T
            d_st = do_tile @ v_tile.T
            
            l_ptr = L_ptr + b_h * S
            l_val = tl.load(l_ptr + row_idx_i[:, None], mask=(row_idx_i[:, None] < S), other=0.0)
            
            p_st = tl.exp(s_st * scale - l_val) * tile_mask_i
            ds_st = p_st * (d_st - o_dot_do)
            
            ds_T = ds_st.T
            acc += ds_T @ q_tile * scale
        
        out_ptr = dK_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx[None, :]
        tl.store(out_ptr, acc.to(tl.bfloat16), mask=tile_mask)


# =========================================================================================================
# KERNEL 3: Computes dV
# =========================================================================================================
@triton.jit
def _kernel_dV(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr, dV_ptr, S, scale
):
    s_j = tl.program_id(0)
    b_h = tl.program_id(1)
    
    row_idx = s_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    for k in range(0, 4):
        col_idx = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tile_mask = (row_idx[:, None] < S) & (col_idx[None, :] < 128)
        
        k_tile = desc_K.load([b_h * S + s_j * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
        v_tile = desc_V.load([b_h * S + s_j * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
        
        acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        for i in range(s_j, (S + BLOCK_SIZE - 1) // BLOCK_SIZE):
            row_idx_i = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            tile_mask_i = (row_idx_i[:, None] < S) & (row_idx_i[:, None] >= row_idx[None, :])
            
            q_tile = desc_Q.load([b_h * S + i * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
            do_tile = desc_dO.load([b_h * S + i * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
            o_tile = desc_O.load([b_h * S + i * BLOCK_SIZE, k * BLOCK_SIZE]).to(tl.float32)
            
            o_dot_do = (o_tile * do_tile).sum(axis=1, keep_dims=True)
            
            s_st = q_tile @ k_tile.T
            d_st = do_tile @ v_tile.T
            
            l_ptr = L_ptr + b_h * S
            l_val = tl.load(l_ptr + row_idx_i[:, None], mask=(row_idx_i[:, None] < S), other=0.0)
            
            p_st = tl.exp(s_st * scale - l_val) * tile_mask_i
            
            pt = p_st.T
            acc += pt @ do_tile
        
        out_ptr = dV_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx[None, :]
        tl.store(out_ptr, acc.to(tl.bfloat16), mask=tile_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    assert B == 4 and H == 48 and d == 128
    
    scale = 1.0 / math.sqrt(d)
    num_S_tiles = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    desc_Q = TensorDescriptor.from_tensor(Q, [BLOCK_SIZE, BLOCK_SIZE])
    desc_K = TensorDescriptor.from_tensor(K, [BLOCK_SIZE, BLOCK_SIZE])
    desc_V = TensorDescriptor.from_tensor(V, [BLOCK_SIZE, BLOCK_SIZE])
    desc_O = TensorDescriptor.from_tensor(O, [BLOCK_SIZE, BLOCK_SIZE])
    desc_dO = TensorDescriptor.from_tensor(dO, [BLOCK_SIZE, BLOCK_SIZE])
    desc_dQ = TensorDescriptor.from_tensor(dQ, [BLOCK_SIZE, BLOCK_SIZE])
    desc_dK = TensorDescriptor.from_tensor(dK, [BLOCK_SIZE, BLOCK_SIZE])
    desc_dV = TensorDescriptor.from_tensor(dV, [BLOCK_SIZE, BLOCK_SIZE])
    
    grid_out = (num_S_tiles, B * H)
    _kernel_dQ[grid_out](desc_Q, desc_K, desc_V, desc_O, desc_dO, L, dQ, S, scale, num_warps=8, num_stages=3)
    _kernel_dK[grid_out](desc_Q, desc_K, desc_V, desc_O, desc_dO, L, dK, S, scale, num_warps=8, num_stages=3)
    _kernel_dV[grid_out](desc_Q, desc_K, desc_V, desc_O, desc_dO, L, dV, S, scale, num_warps=8, num_stages=3)