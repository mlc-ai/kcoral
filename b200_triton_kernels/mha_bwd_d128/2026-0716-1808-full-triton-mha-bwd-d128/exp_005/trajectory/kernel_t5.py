import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _bwd_dk_dv(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, scale, H, B,
    s_stride, l_h_stride, l_b_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    Q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[B * H * S_len, BLOCK_M * BLOCK_N], strides=[BLOCK_M * BLOCK_N, 1], block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(K_ptr, shape=[B * H * S_len, BLOCK_M * BLOCK_N], strides=[BLOCK_M * BLOCK_N, 1], block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(V_ptr, shape=[B * H * S_len, BLOCK_M * BLOCK_N], strides=[BLOCK_M * BLOCK_N, 1], block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")
    O_desc = tl.make_tensor_descriptor(O_ptr, shape=[B * H * S_len, BLOCK_M * BLOCK_N], strides=[BLOCK_M * BLOCK_N, 1], block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")
    dO_desc = tl.make_tensor_descriptor(dO_ptr, shape=[B * H * S_len, BLOCK_M * BLOCK_N], strides=[BLOCK_M * BLOCK_N, 1], block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")

    j = tl.program_id(0) 
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    j_block = j * BLOCK_M
    row_off_j = j_block + tl.arange(0, BLOCK_M)
    
    bh_idx = b * H + h
    j_base = bh_idx * s_stride + j_block * s_stride
    
    K0 = K_desc.load([bh_idx * S_len + j_block, 0]).to(tl.float32)
    K1 = K_desc.load([bh_idx * S_len + j_block, BLOCK_N]).to(tl.float32)
    V0 = V_desc.load([bh_idx * S_len + j_block, 0]).to(tl.float32)
    V1 = V_desc.load([bh_idx * S_len + j_block, BLOCK_N]).to(tl.float32)
    
    dK_acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dK_acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dV_acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dV_acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_q_blocks = tl.cdiv(S_len, BLOCK_M)
    for i in range(num_q_blocks):
        i_block = i * BLOCK_M
        row_off_q = i_block + tl.arange(0, BLOCK_M)
        
        Q0 = Q_desc.load([bh_idx * S_len + i_block, 0]).to(tl.float32)
        Q1 = Q_desc.load([bh_idx * S_len + i_block, BLOCK_N]).to(tl.float32)
        
        dO0 = dO_desc.load([bh_idx * S_len + i_block, 0]).to(tl.float32)
        dO1 = dO_desc.load([bh_idx * S_len + i_block, BLOCK_N]).to(tl.float32)
        
        O0 = O_desc.load([bh_idx * S_len + i_block, 0]).to(tl.float32)
        O1 = O_desc.load([bh_idx * S_len + i_block, BLOCK_N]).to(tl.float32)
        
        D = tl.sum(dO0 * O0, axis=1) + tl.sum(dO1 * O1, axis=1)
        
        l_base_idx = b * l_b_stride + h * l_h_stride
        L_tile = tl.load(L_ptr + l_base_idx + row_off_q,
                          mask=(row_off_q < S_len), other=0.0)
        
        S_acc = tl.dot(Q0, K0.T)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        dP_acc = tl.dot(dO0, V0.T)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        P = tl.exp(S_acc * scale - L_tile[:, None])
        
        col_idx = tl.arange(0, BLOCK_M)
        key_mask = ((j_block + col_idx) < S_len)[None, :]
        P = P * key_mask
        
        dS = P * (dP_acc - D[:, None]) * scale
        
        dK_acc0 = tl.dot(dS.T, Q0, dK_acc0)
        dK_acc1 = tl.dot(dS.T, Q1, dK_acc1)
        
        dV_acc0 = tl.dot(P.T, dO0, dV_acc0)
        dV_acc1 = tl.dot(P.T, dO1, dV_acc1)
        
    col_off0 = tl.arange(0, BLOCK_N)
    col_off1 = BLOCK_N + tl.arange(0, BLOCK_N)
    store_mask = (row_off_j < S_len)[:, None]
    
    tl.store(dK_ptr + j_base + row_off_j[:, None] * s_stride + col_off0[None, :],
             dK_acc0.to(tl.bfloat16), mask=store_mask)
    tl.store(dK_ptr + j_base + row_off_j[:, None] * s_stride + col_off1[None, :],
             dK_acc1.to(tl.bfloat16), mask=store_mask)
             
    tl.store(dV_ptr + j_base + row_off_j[:, None] * s_stride + col_off0[None, :],
             dV_acc0.to(tl.bfloat16), mask=store_mask)
    tl.store(dV_ptr + j_base + row_off_j[:, None] * s_stride + col_off1[None, :],
             dV_acc1.to(tl.bfloat16), mask=store_mask)


@triton.jit
def _bwd_dq(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, scale, H, B,
    s_stride, l_h_stride, l_b_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    Q_desc = tl.make_tensor_descriptor(Q_ptr, shape=[B * H * S_len, BLOCK_M * BLOCK_N], strides=[BLOCK_M * BLOCK_N, 1], block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")
    K_desc = tl.make_tensor_descriptor(K_ptr, shape=[B * H * S_len, BLOCK_M * BLOCK_N], strides=[BLOCK_M * BLOCK_N, 1], block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")
    V_desc = tl.make_tensor_descriptor(V_ptr, shape=[B * H * S_len, BLOCK_M * BLOCK_N], strides=[BLOCK_M * BLOCK_N, 1], block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")
    O_desc = tl.make_tensor_descriptor(O_ptr, shape=[B * H * S_len, BLOCK_M * BLOCK_N], strides=[BLOCK_M * BLOCK_N, 1], block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")
    dO_desc = tl.make_tensor_descriptor(dO_ptr, shape=[B * H * S_len, BLOCK_M * BLOCK_N], strides=[BLOCK_M * BLOCK_N, 1], block_shape=[BLOCK_M, BLOCK_N], padding_option="zero")

    i = tl.program_id(0) 
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    i_block = i * BLOCK_M
    row_off_q = i_block + tl.arange(0, BLOCK_M)
    
    bh_idx = b * H + h
    q_base = bh_idx * s_stride + i_block * s_stride
    
    Q0 = Q_desc.load([bh_idx * S_len + i_block, 0]).to(tl.float32)
    Q1 = Q_desc.load([bh_idx * S_len + i_block, BLOCK_N]).to(tl.float32)
    
    dO0 = dO_desc.load([bh_idx * S_len + i_block, 0]).to(tl.float32)
    dO1 = dO_desc.load([bh_idx * S_len + i_block, BLOCK_N]).to(tl.float32)
    
    O0 = O_desc.load([bh_idx * S_len + i_block, 0]).to(tl.float32)
    O1 = O_desc.load([bh_idx * S_len + i_block, BLOCK_N]).to(tl.float32)
    
    D = tl.sum(dO0 * O0, axis=1) + tl.sum(dO1 * O1, axis=1)
    
    l_base_idx = b * l_b_stride + h * l_h_stride
    L_tile = tl.load(L_ptr + l_base_idx + row_off_q,
                      mask=(row_off_q < S_len), other=0.0)
    
    dQ_acc0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    dQ_acc1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    num_kv_blocks = tl.cdiv(S_len, BLOCK_M)
    for j in range(num_kv_blocks):
        j_block = j * BLOCK_M
        
        K0 = K_desc.load([bh_idx * S_len + j_block, 0]).to(tl.float32)
        K1 = K_desc.load([bh_idx * S_len + j_block, BLOCK_N]).to(tl.float32)
        
        V0 = V_desc.load([bh_idx * S_len + j_block, 0]).to(tl.float32)
        V1 = V_desc.load([bh_idx * S_len + j_block, BLOCK_N]).to(tl.float32)
        
        S_acc = tl.dot(Q0, K0.T)
        S_acc = tl.dot(Q1, K1.T, S_acc)
        
        dP_acc = tl.dot(dO0, V0.T)
        dP_acc = tl.dot(dO1, V1.T, dP_acc)
        
        P = tl.exp(S_acc * scale - L_tile[:, None])
        
        col_idx = tl.arange(0, BLOCK_M)
        key_mask = ((j_block + col_idx) < S_len)[None, :]
        P = P * key_mask
        
        dS = P * (dP_acc - D[:, None]) * scale
        
        dQ_acc0 = tl.dot(dS, K0, dQ_acc0)
        dQ_acc1 = tl.dot(dS, K1, dQ_acc1)
    
    col_off0 = tl.arange(0, BLOCK_N)
    col_off1 = BLOCK_N + tl.arange(0, BLOCK_N)
    store_mask = (row_off_q < S_len)[:, None]
    
    tl.store(dQ_ptr + q_base + row_off_q[:, None] * s_stride + col_off0[None, :],
             dQ_acc0.to(tl.bfloat16), mask=store_mask)
    tl.store(dQ_ptr + q_base + row_off_q[:, None] * s_stride + col_off1[None, :],
             dQ_acc1.to(tl.bfloat16), mask=store_mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    """Compute attention backward gradients into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    
    # Ensure tensors are contiguous to guarantee uniform and predictable element strides.
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
    
    num_blocks = triton.cdiv(S, block_m)
    grid = (num_blocks, H, B)
    
    _bwd_dk_dv[grid](
        Q, K, V, O, dO, L, dK, dV,
        S, scale, H, B,
        s_stride, l_h_stride, l_b_stride,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4, num_stages=2,
    )
    
    _bwd_dq[grid](
        Q, K, V, O, dO, L, dQ,
        S, scale, H, B,
        s_stride, l_h_stride, l_b_stride,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4, num_stages=2,
    )