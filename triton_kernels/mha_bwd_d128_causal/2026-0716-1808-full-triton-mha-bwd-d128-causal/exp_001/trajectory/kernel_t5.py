import torch
import triton
import triton.language as tl
import math

BLOCK_SIZE = 32

# =========================================================================================================
# KERNEL 1: Computes dQ
# =========================================================================================================
@triton.jit
def _kernel_dQ(Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr, S, scale):
    s_i = tl.program_id(0)
    b_h = tl.program_id(1)
    
    row_idx = s_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    Q = []
    O = []
    dO = []
    for step in range(4):
        col_idx = step * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        base_ptr = Q_ptr + b_h * S * 128
        ptr = base_ptr + row_idx[:, None] * 128 + col_idx[None, :]
        mask = (row_idx[:, None] < S) & (col_idx[None, :] < 128)
        Q.append(tl.load(ptr, mask=mask, other=0.0))
        
        base_ptr_o = O_ptr + b_h * S * 128
        ptr_o = base_ptr_o + row_idx[:, None] * 128 + col_idx[None, :]
        O.append(tl.load(ptr_o, mask=mask, other=0.0))
        
        base_ptr_do = dO_ptr + b_h * S * 128
        ptr_do = base_ptr_do + row_idx[:, None] * 128 + col_idx[None, :]
        dO.append(tl.load(ptr_do, mask=mask, other=0.0))
    
    o_dot_do = tl.zeros((BLOCK_SIZE, 1), tl.float32)
    for step in range(4):
        o_dot_do += (O[step] * dO[step]).sum(axis=1, keep_dims=True)
        
    l_ptr = L_ptr + b_h * S
    l_val = tl.load(l_ptr + row_idx, mask=(row_idx < S), other=0.0)
    
    dQ_acc = [tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32) for _ in range(4)]
    
    for k in range(s_i + 1):
        col_idx_k = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        K = []
        V = []
        for step in range(4):
            col_idx = step * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            base_ptr = K_ptr + b_h * S * 128
            ptr = base_ptr + col_idx_k[:, None] * 128 + col_idx[None, :]
            mask = (col_idx_k[:, None] < S) & (col_idx[None, :] < 128)
            K.append(tl.load(ptr, mask=mask, other=0.0))
            
            base_ptr_v = V_ptr + b_h * S * 128
            ptr_v = base_ptr_v + col_idx_k[:, None] * 128 + col_idx[None, :]
            V.append(tl.load(ptr_v, mask=mask, other=0.0))
            
        acc_s = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        acc_d = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        
        for step in range(4):
            acc_s = tl.dot(Q[step], K[step].T, acc_s)
            acc_d = tl.dot(dO[step], V[step].T, acc_d)
            
        tile_mask = (row_idx[:, None] >= col_idx_k[None, :]) & (row_idx[:, None] < S) & (col_idx_k[None, :] < S)
        p_st = tl.exp(acc_s * scale - l_val) * tile_mask
        
        dp_st = p_st * (acc_d - o_dot_do)
        
        for step in range(4):
            dQ_acc[step] = tl.dot(dp_st, K[step], dQ_acc[step])
            
    for step in range(4):
        col_idx = step * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tile_mask = (row_idx[:, None] < S) & (col_idx[None, :] < 128)
        base_ptr = dQ_ptr + b_h * S * 128
        out_ptr = base_ptr + row_idx[:, None] * 128 + col_idx[None, :]
        tl.store(out_ptr, (dQ_acc[step] * scale).to(tl.bfloat16), mask=tile_mask)


# =========================================================================================================
# KERNEL 2: Computes dK and dV
# =========================================================================================================
@triton.jit
def _kernel_dK_dV(Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr, S, scale):
    s_j = tl.program_id(0)
    b_h = tl.program_id(1)
    
    row_idx = s_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    K = []
    V = []
    for step in range(4):
        col_idx = step * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        base_ptr = K_ptr + b_h * S * 128
        ptr = base_ptr + row_idx[:, None] * 128 + col_idx[None, :]
        mask = (row_idx[:, None] < S) & (col_idx[None, :] < 128)
        K.append(tl.load(ptr, mask=mask, other=0.0))
        
        base_ptr_v = V_ptr + b_h * S * 128
        ptr_v = base_ptr_v + row_idx[:, None] * 128 + col_idx[None, :]
        V.append(tl.load(ptr_v, mask=mask, other=0.0))
    
    dK_acc = [tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32) for _ in range(4)]
    dV_acc = [tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32) for _ in range(4)]
    
    num_S_tiles = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for i in range(s_j, num_S_tiles):
        row_idx_i = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        Q = []
        O = []
        dO = []
        for step in range(4):
            col_idx = step * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            base_ptr = Q_ptr + b_h * S * 128
            ptr = base_ptr + row_idx_i[:, None] * 128 + col_idx[None, :]
            mask = (row_idx_i[:, None] < S) & (col_idx[None, :] < 128)
            Q.append(tl.load(ptr, mask=mask, other=0.0))
            
            base_ptr_o = O_ptr + b_h * S * 128
            ptr_o = base_ptr_o + row_idx_i[:, None] * 128 + col_idx[None, :]
            O.append(tl.load(ptr_o, mask=mask, other=0.0))
            
            base_ptr_do = dO_ptr + b_h * S * 128
            ptr_do = base_ptr_do + row_idx_i[:, None] * 128 + col_idx[None, :]
            dO.append(tl.load(ptr_do, mask=mask, other=0.0))
            
        o_dot_do = tl.zeros((BLOCK_SIZE, 1), tl.float32)
        for step in range(4):
            o_dot_do += (O[step] * dO[step]).sum(axis=1, keep_dims=True)
            
        l_ptr = L_ptr + b_h * S
        l_val = tl.load(l_ptr + row_idx_i, mask=(row_idx_i < S), other=0.0)
        
        acc_s = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        acc_d = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        
        for step in range(4):
            acc_s = tl.dot(Q[step], K[step].T, acc_s)
            acc_d = tl.dot(dO[step], V[step].T, acc_d)
            
        tile_mask = (row_idx_i[:, None] >= row_idx[None, :]) & (row_idx_i[:, None] < S) & (row_idx[None, :] < S)
        p_st = tl.exp(acc_s * scale - l_val) * tile_mask
        
        dp_st = p_st * (acc_d - o_dot_do)
        
        for step in range(4):
            dK_acc[step] = tl.dot(dp_st.T, Q[step], dK_acc[step])
            dV_acc[step] = tl.dot(p_st.T, dO[step], dV_acc[step])
            
    for step in range(4):
        col_idx = step * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tile_mask = (row_idx[:, None] < S) & (col_idx[None, :] < 128)
        
        base_ptr_dK = dK_ptr + b_h * S * 128
        out_ptr_dK = base_ptr_dK + row_idx[:, None] * 128 + col_idx[None, :]
        tl.store(out_ptr_dK, (dK_acc[step] * scale).to(tl.bfloat16), mask=tile_mask)
        
        base_ptr_dV = dV_ptr + b_h * S * 128
        out_ptr_dV = base_ptr_dV + row_idx[:, None] * 128 + col_idx[None, :]
        tl.store(out_ptr_dV, dV_acc[step].to(tl.bfloat16), mask=tile_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    assert B == 4 and H == 48 and d == 128
    
    scale = 1.0 / math.sqrt(d)
    num_S_tiles = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    Q_ptr = Q.__tensor_base__().__array_interface__["data"][0]
    K_ptr = K.__tensor_base__().__array_interface__["data"][0]
    V_ptr = V.__tensor_base__().__array_interface__["data"][0]
    O_ptr = O.__tensor_base__().__array_interface__["data"][0]
    dO_ptr = dO.__tensor_base__().__array_interface__["data"][0]
    L_ptr = L.__tensor_base__().__array_interface__["data"][0]
    
    dQ_ptr = dQ.__tensor_base__().__array_interface__["data"][0]
    dK_ptr = dK.__tensor_base__().__array_interface__["data"][0]
    dV_ptr = dV.__tensor_base__().__array_interface__["data"][0]
    
    grid_out = (num_S_tiles, B * H)
    _kernel_dQ[grid_out](Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr, S, scale, num_warps=4, num_stages=4)
    _kernel_dK_dV[grid_out](Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr, S, scale, num_warps=4, num_stages=4)