import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _bwd_dk_dv(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr, dK_ptr, dV_ptr,
    S_len, scale, H, B,
    s_stride, l_h_stride, l_b_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0) 
    h = tl.program_id(1)
    b = tl.program_id(2)
    half = tl.program_id(3)
    
    j_block = j * BLOCK_M
    row_off_j = j_block + tl.arange(0, BLOCK_M)
    
    bh_idx = b * H + h
    row_mask_j = (row_off_j < S_len)
    
    K0 = K_desc.load([bh_idx * S_len + j_block, 0])
    K1 = K_desc.load([bh_idx * S_len + j_block, BLOCK_N])
    
    V0 = V_desc.load([bh_idx * S_len + j_block, 0])
    V1 = V_desc.load([bh_idx * S_len + j_block, BLOCK_N])
    
    dK_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dV_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_q_blocks = tl.cdiv(S_len, BLOCK_M)
    for i in range(num_q_blocks):
        i_block = i * BLOCK_M
        row_off_i = i_block + tl.arange(0, BLOCK_M)
        row_mask_i = (row_off_i < S_len)
        
        Q0 = Q_desc.load([bh_idx * S_len + i_block, 0])
        Q1 = Q_desc.load([bh_idx * S_len + i_block, BLOCK_N])
        
        dO0 = dO_desc.load([bh_idx * S_len + i_block, 0])
        dO1 = dO_desc.load([bh_idx * S_len + i_block, BLOCK_N])
        
        O0 = O_desc.load([bh_idx * S_len + i_block, 0])
        O1 = O_desc.load([bh_idx * S_len + i_block, BLOCK_N])
        
        D = tl.sum(dO0 * O0, axis=1) + tl.sum(dO1 * O1, axis=1)
        
        l_base_idx = b * l_b_stride + h * l_h_stride
        L_tile = tl.load(L_ptr + l_base_idx + row_off_i, mask=row_mask_i, other=0.0)
        
        S_acc = tl.dot(Q0, K0.T)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        dP_acc = tl.dot(dO0, V0.T)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        P = tl.exp(S_acc * scale - L_tile[:, None])
        
        col_idx = tl.arange(0, BLOCK_M)
        key_mask = ((j_block + col_idx) < S_len)[None, :]
        P = P * key_mask
        
        dS = P * (dP_acc - D[:, None]) * scale
        
        dK_acc = tl.dot(dS.T, Q0, dK_acc)
        dK_acc = tl.dot(dS.T, Q1, dK_acc)
        
        dV_acc = tl.dot(P.T, dO0, dV_acc)
        dV_acc = tl.dot(P.T, dO1, dV_acc)
        
    col_off = tl.arange(0, BLOCK_N)
    store_mask = row_mask_j[:, None]
    
    j_base_addr = dK_ptr + bh_idx * s_stride + j_block * s_stride
    curr_addr = j_base_addr + row_off_j[:, None] * s_stride + half * BLOCK_N + col_off[None, :]
    tl.store(curr_addr, dK_acc.to(tl.bfloat16), mask=store_mask)
    
    j_base_addr = dV_ptr + bh_idx * s_stride + j_block * s_stride
    curr_addr = j_base_addr + row_off_j[:, None] * s_stride + half * BLOCK_N + col_off[None, :]
    tl.store(curr_addr, dV_acc.to(tl.bfloat16), mask=store_mask)


@triton.jit
def _bwd_dq(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L_ptr, dQ_ptr,
    S_len, scale, H, B,
    s_stride, l_h_stride, l_b_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0) 
    h = tl.program_id(1)
    b = tl.program_id(2)
    half = tl.program_id(3)
    
    i_block = i * BLOCK_M
    row_off_i = i_block + tl.arange(0, BLOCK_M)
    
    bh_idx = b * H + h
    row_mask_i = (row_off_i < S_len)
    
    Q0 = Q_desc.load([bh_idx * S_len + i_block, 0])
    Q1 = Q_desc.load([bh_idx * S_len + i_block, BLOCK_N])
    
    dO0 = dO_desc.load([bh_idx * S_len + i_block, 0])
    dO1 = dO_desc.load([bh_idx * S_len + i_block, BLOCK_N])
    
    O0 = O_desc.load([bh_idx * S_len + i_block, 0])
    O1 = O_desc.load([bh_idx * S_len + i_block, BLOCK_N])
    
    D = tl.sum(dO0 * O0, axis=1) + tl.sum(dO1 * O1, axis=1)
    
    l_base_idx = b * l_b_stride + h * l_h_stride
    L_tile = tl.load(L_ptr + l_base_idx + row_off_i, mask=row_mask_i, other=0.0)
    
    dQ_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_kv_blocks = tl.cdiv(S_len, BLOCK_M)
    for j in range(num_kv_blocks):
        j_block = j * BLOCK_M
        
        K0 = K_desc.load([bh_idx * S_len + j_block, 0])
        K1 = K_desc.load([bh_idx * S_len + j_block, BLOCK_N])
        
        V0 = V_desc.load([bh_idx * S_len + j_block, 0])
        V1 = V_desc.load([bh_idx * S_len + j_block, BLOCK_N])
        
        S_acc = tl.dot(Q0, K0.T)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        dP_acc = tl.dot(dO0, V0.T)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        P = tl.exp(S_acc * scale - L_tile[:, None])
        
        col_idx = tl.arange(0, BLOCK_M)
        key_mask = ((j_block + col_idx) < S_len)[None, :]
        P = P * key_mask
        
        dS = P * (dP_acc - D[:, None]) * scale
        
        dQ_acc = tl.dot(dS, K0, dQ_acc)
        dQ_acc = tl.dot(dS, K1, dQ_acc)
    
    col_off = tl.arange(0, BLOCK_N)
    store_mask = row_mask_i[:, None]
    
    i_base_addr = dQ_ptr + bh_idx * s_stride + i_block * s_stride
    curr_addr = i_base_addr + row_off_i[:, None] * s_stride + half * BLOCK_N + col_off[None, :]
    tl.store(curr_addr, dQ_acc.to(tl.bfloat16), mask=store_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward gradients into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    O = O.contiguous()
    dO = dO.contiguous()
    dQ = dQ.contiguous()
    dK = dK.contiguous()
    dV = dV.contiguous()
    
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    s_stride = d
    l_h_stride = S
    l_b_stride = H * S
    
    block_m = 64
    block_n = 64 
    
    Q_2d = Q.view(B * H * S, d)
    K_2d = K.view(B * H * S, d)
    V_2d = V.view(B * H * S, d)
    O_2d = O.view(B * H * S, d)
    dO_2d = dO.view(B * H * S, d)
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [block_m, block_n])
    K_desc = TensorDescriptor.from_tensor(K_2d, [block_m, block_n])
    V_desc = TensorDescriptor.from_tensor(V_2d, [block_m, block_n])
    O_desc = TensorDescriptor.from_tensor(O_2d, [block_m, block_n])
    dO_desc = TensorDescriptor.from_tensor(dO_2d, [block_m, block_n])
    
    num_blocks = triton.cdiv(S, block_m)
    grid = (num_blocks, H, B, 2)
    
    _bwd_dk_dv[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK, dV,
        S, scale, H, B,
        s_stride, l_h_stride, l_b_stride,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=8, num_stages=3,
    )
    
    _bwd_dq[grid](
        Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ,
        S, scale, H, B,
        s_stride, l_h_stride, l_b_stride,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=8, num_stages=3,
    )