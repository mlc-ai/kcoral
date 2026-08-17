import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math

BLOCK_SIZE = 128

def pre_kernel_launch(kernel, desc_Q, desc_K, desc_V, desc_O, desc_dO):
    for desc in [desc_Q, desc_K, desc_V, desc_O, desc_dO]:
        desc.block_shape = [BLOCK_SIZE, BLOCK_SIZE]

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
    
    q = []
    do = []
    o = []
    for step in range(4):
        q.append(desc_Q.load([b_h * S + s_i * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
        do.append(desc_dO.load([b_h * S + s_i * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
        o.append(desc_O.load([b_h * S + s_i * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
    
    o_dot_do = tl.zeros((BLOCK_SIZE, 1), tl.float32)
    for step in range(4):
        o_dot_do += (o[step] * do[step]).sum(axis=1, keep_dims=True)
        
    l_ptr = L_ptr + b_h * S
    l_val = tl.load(l_ptr + row_idx[:, None], mask=(row_idx[:, None] < S), other=0.0)
    
    dQ_acc = [tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32) for _ in range(4)]
    
    for k in range(s_i + 1):
        col_idx_k = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        k = []
        v = []
        for step in range(4):
            k.append(desc_K.load([b_h * S + k * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
            v.append(desc_V.load([b_h * S + k * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
            
        acc_s = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        acc_d = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        
        for step in range(4):
            acc_s = tl.dot(q[step], k[step].T, acc_s)
            acc_d = tl.dot(do[step], v[step].T, acc_d)
            
        tile_mask = (row_idx[:, None] >= col_idx_k[None, :]) & (row_idx[:, None] < S) & (col_idx_k[None, :] < S)
        p_st = tl.exp(acc_s * scale - l_val) * tile_mask
        
        dP_st = p_st * (acc_d - o_dot_do)
        
        for step in range(4):
            dQ_acc[step] = tl.dot(dP_st, k[step], dQ_acc[step])
            
    for step in range(4):
        col_idx_step = step * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tile_mask = (row_idx[:, None] < S) & (col_idx_step[None, :] < 128)
        out_ptr_dQ = dQ_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx_step[None, :]
        tl.store(out_ptr_dQ, (dQ_acc[step] * scale).to(tl.bfloat16), mask=tile_mask)


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
    
    k = []
    v = []
    for step in range(4):
        k.append(desc_K.load([b_h * S + s_j * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
        v.append(desc_V.load([b_h * S + s_j * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
        
    dK_acc = [tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32) for _ in range(4)]
    dV_acc = [tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32) for _ in range(4)]
    
    num_S_tiles = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    for i in range(s_j, num_S_tiles):
        row_idx_i = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        q = []
        do = []
        o = []
        for step in range(4):
            q.append(desc_Q.load([b_h * S + i * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
            do.append(desc_dO.load([b_h * S + i * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
            o.append(desc_O.load([b_h * S + i * BLOCK_SIZE, step * BLOCK_SIZE]).squeeze(0))
            
        o_dot_do = tl.zeros((BLOCK_SIZE, 1), tl.float32)
        for step in range(4):
            o_dot_do += (o[step] * do[step]).sum(axis=1, keep_dims=True)
            
        l_ptr = L_ptr + b_h * S
        l_val = tl.load(l_ptr + row_idx_i[:, None], mask=(row_idx_i[:, None] < S), other=0.0)
        
        acc_s = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        acc_d = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), tl.float32)
        
        for step in range(4):
            acc_s = tl.dot(q[step], k[step].T, acc_s)
            acc_d = tl.dot(do[step], v[step].T, acc_d)
            
        tile_mask = (row_idx_i[:, None] >= row_idx[None, :]) & (row_idx_i[:, None] < S) & (row_idx[None, :] < S)
        p_st = tl.exp(acc_s * scale - l_val) * tile_mask
        
        dP_st = p_st * (acc_d - o_dot_do)
        
        for step in range(4):
            dK_acc[step] = tl.dot(dP_st.T, q[step], dK_acc[step])
            dV_acc[step] = tl.dot(p_st.T, do[step], dV_acc[step])
            
    for step in range(4):
        col_idx_step = step * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tile_mask = (row_idx[:, None] < S) & (col_idx_step[None, :] < 128)
        
        out_ptr_dK = dK_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx_step[None, :]
        tl.store(out_ptr_dK, (dK_acc[step] * scale).to(tl.bfloat16), mask=tile_mask)
        
        out_ptr_dV = dV_ptr + b_h * S * 128 + row_idx[:, None] * 128 + col_idx_step[None, :]
        tl.store(out_ptr_dV, dV_acc[step].to(tl.bfloat16), mask=tile_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    assert B == 4 and H == 48 and d == 128
    
    scale = 1.0 / math.sqrt(d)
    num_S_tiles = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    desc_Q = TensorDescriptor.from_tensor(Q.view(B * H * S, d), [BLOCK_SIZE, BLOCK_SIZE], contiguous=True)
    desc_K = TensorDescriptor.from_tensor(K.view(B * H * S, d), [BLOCK_SIZE, BLOCK_SIZE], contiguous=True)
    desc_V = TensorDescriptor.from_tensor(V.view(B * H * S, d), [BLOCK_SIZE, BLOCK_SIZE], contiguous=True)
    desc_O = TensorDescriptor.from_tensor(O.view(B * H * S, d), [BLOCK_SIZE, BLOCK_SIZE], contiguous=True)
    desc_dO = TensorDescriptor.from_tensor(dO.view(B * H * S, d), [BLOCK_SIZE, BLOCK_SIZE], contiguous=True)
    
    grid_out = (num_S_tiles, B * H)
    pre_kernel_launch(_kernel_dQ, desc_Q, desc_K, desc_V, desc_O, desc_dO)
    _kernel_dQ[grid_out](desc_Q, desc_K, desc_V, desc_O, desc_dO, L, dQ, S, scale, num_warps=4, num_stages=3)
    
    pre_kernel_launch(_kernel_dK_dV, desc_Q, desc_K, desc_V, desc_O, desc_dO)
    _kernel_dK_dV[grid_out](desc_Q, desc_K, desc_V, desc_O, desc_dO, L, dK, dV, S, scale, num_warps=4, num_stages=3)