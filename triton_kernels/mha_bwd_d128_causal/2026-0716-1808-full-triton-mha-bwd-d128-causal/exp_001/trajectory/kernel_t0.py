import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

BLOCK_SIZE = 16

# =========================================================================================================
# KERNEL 1: Computes the intermediate attention gradients and softmax probabilities
# =========================================================================================================
@triton.jit
def _kernel_1(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_ptr, dP_ptr, P_T_ptr, S, scale
):
    s_i = tl.program_id(0)
    s_j = tl.program_id(1)
    b_h = tl.program_id(2)

    if s_i < s_j:
        return  # Causal mask skip
    
    row_idx = s_i * 16 + tl.arange(0, BLOCK_SIZE)
    col_idx = s_j * 16 + tl.arange(0, BLOCK_SIZE)
    tile_mask = (row_idx[:, None] < S) & (col_idx[None, :] < S) & (row_idx[:, None] >= col_idx[None, :])
    
    l_ptr = L_ptr + b_h * S
    l_val = tl.load(l_ptr + row_idx[:, None], mask=(row_idx[:, None] < S), other=0.0)
    
    S_st = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
    D_st = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
    O_dot_dO = tl.zeros((BLOCK_SIZE, 1), tl.float32)
    
    for k in range(0, 128, BLOCK_SIZE):
        Q_tile = desc_Q.load([b_h * S + s_i * 16, k])
        K_tile = desc_K.load([b_h * S + s_j * 16, k])
        dO_tile = desc_dO.load([b_h * S + s_i * 16, k])
        V_tile = desc_V.load([b_h * S + s_j * 16, k])
        O_tile = desc_O.load([b_h * S + s_i * 16, k])
        
        Q_f32 = Q_tile.to(tl.float32)
        K_f32 = K_tile.to(tl.float32)
        dO_f32 = dO_tile.to(tl.float32)
        V_f32 = V_tile.to(tl.float32)
        O_f32 = O_tile.to(tl.float32)
        
        S_st += Q_f32 @ K_f32.T
        D_st += dO_f32 @ V_f32.T
        O_dot_dO += (O_f32 * dO_f32).sum(axis=1, keep_dims=True)
        
    S_st *= scale
    
    P_st = tl.exp(S_st - l_val) * tile_mask
    
    dS_st = P_st * (D_st - O_dot_dO)
    
    P_T_st = P_st.T
    
    base_idx = b_h * S * S + s_i * S * 16 + s_j * 16
    ptr_dS = dP_ptr + base_idx + row_idx[:, None] * S + col_idx[None, :]
    tl.store(ptr_dS, dS_st, mask=tile_mask)
    
    base_idx_T = b_h * S * S + s_j * S * 16 + s_i * 16
    ptr_PT = P_T_ptr + base_idx_T + col_idx[:, None] * S + row_idx[None, :]
    tl.store(ptr_PT, P_T_st, mask=tile_mask)


# =========================================================================================================
# KERNEL 2: Iterates along Keys to calculate dQ
# =========================================================================================================
@triton.jit
def _kernel_dQ(desc_K, desc_dP, dQ_ptr, S, scale):
    s_i = tl.program_id(0)
    b_h = tl.program_id(1)
    
    for k in range(0, 128, BLOCK_SIZE):
        acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        for j in range(0, s_i + 1):
            ds = desc_dP.load([b_h * S + s_i * 16, j * 16])
            kk = desc_K.load([b_h * S + j * 16, k])
            acc += ds @ kk * scale
        
        row_idx = s_i * 16 + tl.arange(0, BLOCK_SIZE)
        col_idx = k + tl.arange(0, BLOCK_SIZE)
        tile_mask = (row_idx[:, None] < S) & (col_idx[None, :] < 128)
        out_ptr = dQ_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx[None, :]
        tl.store(out_ptr, acc, mask=tile_mask)


# =========================================================================================================
# KERNEL 3: Iterates along Queries to calculate dK
# =========================================================================================================
@triton.jit
def _kernel_dK(desc_Q, desc_dP, dK_ptr, S, scale):
    s_j = tl.program_id(0)
    b_h = tl.program_id(1)
    
    for k in range(0, 128, BLOCK_SIZE):
        acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        for i in range(s_j, (S + 15) // 16):
            ds = desc_dP.load([b_h * S + i * 16, s_j * 16])
            ds_T = ds.T
            qq = desc_Q.load([b_h * S + i * 16, k])
            acc += ds_T @ qq * scale
        
        row_idx = s_j * 16 + tl.arange(0, BLOCK_SIZE)
        col_idx = k + tl.arange(0, BLOCK_SIZE)
        tile_mask = (row_idx[:, None] < S) & (col_idx[None, :] < 128)
        out_ptr = dK_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx[None, :]
        tl.store(out_ptr, acc, mask=tile_mask)


# =========================================================================================================
# KERNEL 4: Iterates along Queries to calculate dV
# =========================================================================================================
@triton.jit
def _kernel_dV(desc_dO, desc_P_T, dV_ptr, S):
    s_j = tl.program_id(0)
    b_h = tl.program_id(1)
    
    for k in range(0, 128, BLOCK_SIZE):
        acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        for i in range(s_j, (S + 15) // 16):
            pt = desc_P_T.load([b_h * S + s_j * 16, i * 16])
            do = desc_dO.load([b_h * S + i * 16, k])
            acc += pt @ do
        
        row_idx = s_j * 16 + tl.arange(0, BLOCK_SIZE)
        col_idx = k + tl.arange(0, BLOCK_SIZE)
        tile_mask = (row_idx[:, None] < S) & (col_idx[None, :] < 128)
        out_ptr = dV_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx[None, :]
        tl.store(out_ptr, acc, mask=tile_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    assert B == 4 and H == 48 and d == 128
    
    scale = 1.0 / math.sqrt(d)
    num_S_tiles = (S + 15) // 16
    
    dP_ptr = torch.empty(B * H * S * S, dtype=torch.bfloat16, device=Q.device)
    P_T_ptr = torch.empty(B * H * S * S, dtype=torch.bfloat16, device=Q.device)
    
    desc_Q = TensorDescriptor.from_tensor(Q, [BLOCK_SIZE, BLOCK_SIZE])
    desc_K = TensorDescriptor.from_tensor(K, [BLOCK_SIZE, BLOCK_SIZE])
    desc_V = TensorDescriptor.from_tensor(V, [BLOCK_SIZE, BLOCK_SIZE])
    desc_O = TensorDescriptor.from_tensor(O, [BLOCK_SIZE, BLOCK_SIZE])
    desc_dO = TensorDescriptor.from_tensor(dO, [BLOCK_SIZE, BLOCK_SIZE])
    desc_dP = TensorDescriptor.from_tensor(dP_ptr.view(B * H * S, S), [BLOCK_SIZE, BLOCK_SIZE])
    desc_P_T = TensorDescriptor.from_tensor(P_T_ptr.view(B * H * S, S), [BLOCK_SIZE, BLOCK_SIZE])
    
    grid_1 = (num_S_tiles, num_S_tiles, B * H)
    _kernel_1[grid_1](desc_Q, desc_K, desc_V, desc_O, desc_dO, L, dP_ptr, P_T_ptr, S, scale)
    
    grid_out = (num_S_tiles, B * H)
    _kernel_dQ[grid_out](desc_K, desc_dP, dQ, S, scale)
    _kernel_dK[grid_out](desc_Q, desc_dP, dK, S, scale)
    _kernel_dV[grid_out](desc_dO, desc_P_T, dV, S)