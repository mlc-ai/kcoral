import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _bwd_dk_dv(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK_desc, dV_desc,
    S_len, scale, H, B,
    l_h_stride, l_b_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_SMS: tl.constexpr, launch_idx: tl.constexpr,
):
    j = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    h_idx = tl.program_id(3)
    
    flat_idx = (((j * H + h) * B + b) * 2 + h_idx)
    if flat_idx // NUM_SMS != launch_idx:
        return
    
    j_block = j * BLOCK_M
    row_off_j = j_block + tl.arange(0, BLOCK_M)
    
    base_idx = b * H * S_len + h * S_len + j_block
    K0 = K_desc.load([base_idx, h_idx * BLOCK_N])
    K1 = K_desc.load([base_idx, h_idx * BLOCK_N + BLOCK_N])
    V0 = V_desc.load([base_idx, h_idx * BLOCK_N])
    V1 = V_desc.load([base_idx, h_idx * BLOCK_N + BLOCK_N])
    
    dK_acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dK_acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dV_acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dV_acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_q_blocks = tl.cdiv(S_len, BLOCK_M)
    for i in range(num_q_blocks, num_stages=2):
        i_block = i * BLOCK_M
        row_off_q = i_block + tl.arange(0, BLOCK_M)
        
        q_base_idx = b * H * S_len + h * S_len + i_block
        
        Q0 = Q_desc.load([q_base_idx, 0])
        Q1 = Q_desc.load([q_base_idx, BLOCK_N])
        
        dO0 = dO_desc.load([q_base_idx, 0])
        dO1 = dO_desc.load([q_base_idx, BLOCK_N])
        
        O0 = O_desc.load([q_base_idx, 0])
        O1 = O_desc.load([q_base_idx, BLOCK_N])
        
        D = tl.sum(dO0 * O0, axis=1) + tl.sum(dO1 * O1, axis=1)
        
        l_base_idx = b * l_b_stride + h * l_h_stride
        L_tile = tl.load(L + l_base_idx + row_off_q,
                          mask=(row_off_q < S_len), other=0.0)
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_M), tl.float32)
        S_acc = tl.dot(Q0, K0.T, S_acc)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_M), tl.float32)
        dP_acc = tl.dot(dO0, V0.T, dP_acc)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        P = tl.exp(S_acc * scale - L_tile[:, None])
        
        col_idx = tl.arange(0, BLOCK_M)
        key_mask = ((j_block + col_idx) < S_len)[None, :]
        P = P * key_mask
        
        D_expanded = D[:, None]
        
        dS = P * (dP_acc - D_expanded) * scale
        
        P_T = P.T
        dS_T = dS.T
        
        dV_acc0 = tl.dot(P_T, dO0, dV_acc0)
        dK_acc0 = tl.dot(dS_T, Q0, dK_acc0)
        
        dV_acc1 = tl.dot(P_T, dO1, dV_acc1)
        dK_acc1 = tl.dot(dS_T, Q1, dK_acc1)
        
    store_mask = (row_off_j[:, None] < S_len)
    
    out_base_idx = b * H * S_len + h * S_len + j_block
    dK_h0 = dK_desc.store([out_base_idx, h_idx * BLOCK_N], dK_acc0.to(tl.bfloat16))
    tl.store(dK_h0, dK_acc0.to(tl.bfloat16), mask=store_mask)
    
    dK_h1 = dK_desc.store([out_base_idx, h_idx * BLOCK_N + BLOCK_N], dK_acc1.to(tl.bfloat16))
    tl.store(dK_h1, dK_acc1.to(tl.bfloat16), mask=store_mask)
    
    dV_h0 = dV_desc.store([out_base_idx, h_idx * BLOCK_N], dV_acc0.to(tl.bfloat16))
    tl.store(dV_h0, dV_acc0.to(tl.bfloat16), mask=store_mask)
    
    dV_h1 = dV_desc.store([out_base_idx, h_idx * BLOCK_N + BLOCK_N], dV_acc1.to(tl.bfloat16))
    tl.store(dV_h1, dV_acc1.to(tl.bfloat16), mask=store_mask)


@triton.jit
def _bwd_dq(
    Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ_desc,
    S_len, scale, H, B,
    l_h_stride, l_b_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_SMS: tl.constexpr, launch_idx: tl.constexpr,
):
    i = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    h_idx = tl.program_id(3)
    
    flat_idx = (((i * H + h) * B + b) * 2 + h_idx)
    if flat_idx // NUM_SMS != launch_idx:
        return
    
    i_block = i * BLOCK_M
    row_off_q = i_block + tl.arange(0, BLOCK_M)
    
    base_idx = b * H * S_len + h * S_len + i_block
    
    Q0 = Q_desc.load([base_idx, 0])
    Q1 = Q_desc.load([base_idx, BLOCK_N])
    
    dO0 = dO_desc.load([base_idx, 0])
    dO1 = dO_desc.load([base_idx, BLOCK_N])
    
    O0 = O_desc.load([base_idx, 0])
    O1 = O_desc.load([base_idx, BLOCK_N])
    
    D = tl.sum(dO0 * O0, axis=1) + tl.sum(dO1 * O1, axis=1)
    
    l_base_idx = b * l_b_stride + h * l_h_stride
    L_tile = tl.load(L + l_base_idx + row_off_q,
                      mask=(row_off_q < S_len), other=0.0)
    
    D_expanded = D[:, None]
    
    dQ_acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dQ_acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_kv_blocks = tl.cdiv(S_len, BLOCK_M)
    for j in range(num_kv_blocks, num_stages=2):
        j_block = j * BLOCK_M
        row_off_k = j_block + tl.arange(0, BLOCK_M)
        
        k_base_idx = b * H * S_len + h * S_len + j_block
        
        K0 = K_desc.load([k_base_idx, 0])
        K1 = K_desc.load([k_base_idx, BLOCK_N])
        
        V0 = V_desc.load([k_base_idx, 0])
        V1 = V_desc.load([k_base_idx, BLOCK_N])
        
        S_acc = tl.zeros((BLOCK_M, BLOCK_M), tl.float32)
        S_acc = tl.dot(Q0, K0.T, S_acc)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        dP_acc = tl.zeros((BLOCK_M, BLOCK_M), tl.float32)
        dP_acc = tl.dot(dO0, V0.T, dP_acc)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        P = tl.exp(S_acc * scale - L_tile[:, None])
        
        col_idx = tl.arange(0, BLOCK_M)
        key_mask = ((j_block + col_idx) < S_len)[None, :]
        P = P * key_mask
        
        dS = P * (dP_acc - D_expanded) * scale
        
        dQ_acc0 = tl.dot(dS, K0, dQ_acc0)
        dQ_acc1 = tl.dot(dS, K1, dQ_acc1)
    
    store_mask = (row_off_q[:, None] < S_len)
    
    out_base_idx = b * H * S_len + h * S_len + i_block
    dQ_h0 = dQ_desc.store([out_base_idx, h_idx * BLOCK_N], dQ_acc0.to(tl.bfloat16))
    tl.store(dQ_h0, dQ_acc0.to(tl.bfloat16), mask=store_mask)
    
    dQ_h1 = dQ_desc.store([out_base_idx, h_idx * BLOCK_N + BLOCK_N], dQ_acc1.to(tl.bfloat16))
    tl.store(dQ_h1, dQ_acc1.to(tl.bfloat16), mask=store_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward gradients into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    l_h_stride = S
    l_b_stride = H * S
    
    block_m = 64
    block_n = 64
    
    Q_2d = Q.view(B * H * S, d)
    K_2d = K.view(B * H * S, d)
    V_2d = V.view(B * H * S, d)
    O_2d = O.view(B * H * S, d)
    dO_2d = dO.view(B * H * S, d)
    dQ_2d = dQ.view(B * H * S, d)
    dK_2d = dK.view(B * H * S, d)
    dV_2d = dV.view(B * H * S, d)
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [block_m, block_n])
    K_desc = TensorDescriptor.from_tensor(K_2d, [block_m, block_n])
    V_desc = TensorDescriptor.from_tensor(V_2d, [block_m, block_n])
    O_desc = TensorDescriptor.from_tensor(O_2d, [block_m, block_n])
    dO_desc = TensorDescriptor.from_tensor(dO_2d, [block_m, block_n])
    dQ_desc = TensorDescriptor.from_tensor(dQ_2d, [block_m, block_n])
    dK_desc = TensorDescriptor.from_tensor(dK_2d, [block_m, block_n])
    dV_desc = TensorDescriptor.from_tensor(dV_2d, [block_m, block_n])
    
    NUM_SMS = 132
    
    num_kv_blocks = triton.cdiv(S, block_m)
    total_tiles = num_kv_blocks * H * B * 2
    num_launches = triton.cdiv(total_tiles, NUM_SMS)
    
    for launch_idx in range(num_launches):
        grid = (num_kv_blocks, H, B, 2)
        _bwd_dk_dv[grid](
            Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dK_desc, dV_desc,
            S, scale, H, B,
            l_h_stride, l_b_stride,
            BLOCK_M=block_m, BLOCK_N=block_n,
            NUM_SMS=NUM_SMS, launch_idx=launch_idx,
            num_warps=4,
        )
    
    num_q_blocks = triton.cdiv(S, block_m)
    total_tiles = num_q_blocks * H * B * 2
    num_launches = triton.cdiv(total_tiles, NUM_SMS)
    
    for launch_idx in range(num_launches):
        grid = (num_q_blocks, H, B, 2)
        _bwd_dq[grid](
            Q_desc, K_desc, V_desc, O_desc, dO_desc, L, dQ_desc,
            S, scale, H, B,
            l_h_stride, l_b_stride,
            BLOCK_M=block_m, BLOCK_N=block_n,
            NUM_SMS=NUM_SMS, launch_idx=launch_idx,
            num_warps=4,
        )