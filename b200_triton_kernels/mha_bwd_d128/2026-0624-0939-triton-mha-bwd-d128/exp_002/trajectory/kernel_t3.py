import torch
import triton
import triton.language as tl
import math


@triton.jit
def backward_dK_dV_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    seq_len,
    q_stride_b, q_stride_h, q_stride_s, q_stride_d,
    k_stride_b, k_stride_h, k_stride_s, k_stride_d,
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,
    o_stride_b, o_stride_h, o_stride_s, o_stride_d,
    do_stride_b, do_stride_h, do_stride_s, do_stride_d,
    l_stride_b, l_stride_h, l_stride_s,
    dk_stride_b, dk_stride_h, dk_stride_s, dk_stride_d,
    dv_stride_b, dv_stride_h, dv_stride_s, dv_stride_d,
    inv_sqrt_d: tl.constexpr,
    BLOCK: tl.constexpr,
    D_HEAD_DIM: tl.constexpr,
):
    bid_j = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    j_start = bid_j * BLOCK
    k_row_indices = j_start + tl.arange(0, BLOCK)
    d_indices = tl.arange(0, D_HEAD_DIM)
    
    valid_k_rows = k_row_indices[:, None] < seq_len
    valid_d_cols = d_indices[None, :] < D_HEAD_DIM
    mask_K = valid_k_rows & valid_d_cols
    
    k_ptrs = K_ptr + bid_b * k_stride_b + bid_h * k_stride_h + k_row_indices[:, None] * k_stride_s + d_indices[None, :] * k_stride_d
    K_tile = tl.load(k_ptrs, mask=mask_K, other=0.0)
    K_fp32 = K_tile.to(tl.float32)
    
    v_ptrs = V_ptr + bid_b * v_stride_b + bid_h * v_stride_h + k_row_indices[:, None] * v_stride_s + d_indices[None, :] * v_stride_d
    V_tile = tl.load(v_ptrs, mask=mask_K, other=0.0)
    V_fp32 = V_tile.to(tl.float32)
    
    dK_acc = tl.zeros((BLOCK, D_HEAD_DIM), tl.float32)
    dV_acc = tl.zeros((BLOCK, D_HEAD_DIM), tl.float32)
    
    for i in range(0, seq_len, BLOCK):
        q_row_indices = i + tl.arange(0, BLOCK)
        valid_q_rows = q_row_indices[:, None] < seq_len
        mask_Q = valid_q_rows & valid_d_cols
        
        q_ptrs = Q_ptr + bid_b * q_stride_b + bid_h * q_stride_h + q_row_indices[:, None] * q_stride_s + d_indices[None, :] * q_stride_d
        Q_tile = tl.load(q_ptrs, mask=mask_Q, other=0.0)
        Q_fp32 = Q_tile.to(tl.float32)
        
        do_ptrs = dO_ptr + bid_b * do_stride_b + bid_h * do_stride_h + q_row_indices[:, None] * do_stride_s + d_indices[None, :] * do_stride_d
        dO_tile = tl.load(do_ptrs, mask=mask_Q, other=0.0)
        dO_fp32 = dO_tile.to(tl.float32)
        
        o_ptrs = O_ptr + bid_b * o_stride_b + bid_h * o_stride_h + q_row_indices[:, None] * o_stride_s + d_indices[None, :] * o_stride_d
        O_tile = tl.load(o_ptrs, mask=mask_Q, other=0.0)
        O_fp32 = O_tile.to(tl.float32)
        
        l_ptrs = L_ptr + bid_b * l_stride_b + bid_h * l_stride_h + q_row_indices * l_stride_s
        L_i = tl.load(l_ptrs, mask=(q_row_indices < seq_len), other=0.0)
        
        D_i = tl.sum(dO_fp32 * O_fp32, axis=1)
        
        S = tl.dot(Q_fp32, K_fp32.T)
        P = tl.exp(S * inv_sqrt_d - L_i[:, None])
        
        dP = tl.dot(dO_fp32, V_fp32.T)
        dS = P * (dP - D_i[:, None])
        
        dK_acc = tl.dot(dS.T, Q_fp32, dK_acc)
        dV_acc = tl.dot(P.T, dO_fp32, dV_acc)
        
    dk_ptrs = dK_ptr + bid_b * dk_stride_b + bid_h * dk_stride_h + k_row_indices[:, None] * dk_stride_s + d_indices[None, :] * dk_stride_d
    tl.store(dk_ptrs, (dK_acc * inv_sqrt_d).to(tl.bfloat16), mask=mask_K)
    
    dv_ptrs = dV_ptr + bid_b * dv_stride_b + bid_h * dv_stride_h + k_row_indices[:, None] * dv_stride_s + d_indices[None, :] * dv_stride_d
    tl.store(dv_ptrs, dV_acc.to(tl.bfloat16), mask=mask_K)


@triton.jit
def backward_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    seq_len,
    q_stride_b, q_stride_h, q_stride_s, q_stride_d,
    k_stride_b, k_stride_h, k_stride_s, k_stride_d,
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,
    o_stride_b, o_stride_h, o_stride_s, o_stride_d,
    do_stride_b, do_stride_h, do_stride_s, do_stride_d,
    l_stride_b, l_stride_h, l_stride_s,
    dq_stride_b, dq_stride_h, dq_stride_s, dq_stride_d,
    inv_sqrt_d: tl.constexpr,
    BLOCK: tl.constexpr,
    D_HEAD_DIM: tl.constexpr,
):
    bid_i = tl.program_id(0)
    bid_h = tl.program_id(1)
    bid_b = tl.program_id(2)
    
    i_start = bid_i * BLOCK
    q_row_indices = i_start + tl.arange(0, BLOCK)
    d_indices = tl.arange(0, D_HEAD_DIM)
    
    valid_q_rows = q_row_indices[:, None] < seq_len
    valid_d_cols = d_indices[None, :] < D_HEAD_DIM
    mask_Q = valid_q_rows & valid_d_cols
    
    q_ptrs = Q_ptr + bid_b * q_stride_b + bid_h * q_stride_h + q_row_indices[:, None] * q_stride_s + d_indices[None, :] * q_stride_d
    Q_tile = tl.load(q_ptrs, mask=mask_Q, other=0.0)
    Q_fp32 = Q_tile.to(tl.float32)
    
    do_ptrs = dO_ptr + bid_b * do_stride_b + bid_h * do_stride_h + q_row_indices[:, None] * do_stride_s + d_indices[None, :] * do_stride_d
    dO_tile = tl.load(do_ptrs, mask=mask_Q, other=0.0)
    dO_fp32 = dO_tile.to(tl.float32)
    
    o_ptrs = O_ptr + bid_b * o_stride_b + bid_h * o_stride_h + q_row_indices[:, None] * o_stride_s + d_indices[None, :] * o_stride_d
    O_tile = tl.load(o_ptrs, mask=mask_Q, other=0.0)
    O_fp32 = O_tile.to(tl.float32)
    
    l_ptrs = L_ptr + bid_b * l_stride_b + bid_h * l_stride_h + q_row_indices * l_stride_s
    L_i = tl.load(l_ptrs, mask=(q_row_indices < seq_len), other=0.0)
    
    D_i = tl.sum(dO_fp32 * O_fp32, axis=1)
    
    dQ_acc = tl.zeros((BLOCK, D_HEAD_DIM), tl.float32)
    
    for j in range(0, seq_len, BLOCK):
        k_row_indices = j + tl.arange(0, BLOCK)
        valid_k_rows = k_row_indices[:, None] < seq_len
        mask_K = valid_k_rows & valid_d_cols
        
        k_ptrs = K_ptr + bid_b * k_stride_b + bid_h * k_stride_h + k_row_indices[:, None] * k_stride_s + d_indices[None, :] * k_stride_d
        K_tile = tl.load(k_ptrs, mask=mask_K, other=0.0)
        K_fp32 = K_tile.to(tl.float32)
        
        v_ptrs = V_ptr + bid_b * v_stride_b + bid_h * v_stride_h + k_row_indices[:, None] * v_stride_s + d_indices[None, :] * v_stride_d
        V_tile = tl.load(v_ptrs, mask=mask_K, other=0.0)
        V_fp32 = V_tile.to(tl.float32)
        
        S = tl.dot(Q_fp32, K_fp32.T)
        P = tl.exp(S * inv_sqrt_d - L_i[:, None])
        
        dP = tl.dot(dO_fp32, V_fp32.T)
        dS = P * (dP - D_i[:, None])
        
        dQ_acc = tl.dot(dS, K_fp32, dQ_acc)
        
    dq_ptrs = dQ_ptr + bid_b * dq_stride_b + bid_h * dq_stride_h + q_row_indices[:, None] * dq_stride_s + d_indices[None, :] * dq_stride_d
    tl.store(dq_ptrs, (dQ_acc * inv_sqrt_d).to(tl.bfloat16), mask=mask_Q)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    
    B, H, S, d = Q.shape
    inv_sqrt_d = 1.0 / math.sqrt(d)
    BLOCK = 128
    
    dQ.zero_()
    dK.zero_()
    dV.zero_()
    
    q_strides = Q.stride()
    k_strides = K.stride()
    v_strides = V.stride()
    o_strides = O.stride()
    do_strides = dO.stride()
    l_strides = L.stride()
    dq_strides = dQ.stride()
    dk_strides = dK.stride()
    dv_strides = dV.stride()
    
    grid = (triton.cdiv(S, BLOCK), H, B)
    
    backward_dK_dV_kernel[grid](
        Q, K, V, O, dO, L, dK, dV, S,
        q_strides[0], q_strides[1], q_strides[2], q_strides[3],
        k_strides[0], k_strides[1], k_strides[2], k_strides[3],
        v_strides[0], v_strides[1], v_strides[2], v_strides[3],
        o_strides[0], o_strides[1], o_strides[2], o_strides[3],
        do_strides[0], do_strides[1], do_strides[2], do_strides[3],
        l_strides[0], l_strides[1], l_strides[2],
        dk_strides[0], dk_strides[1], dk_strides[2], dk_strides[3],
        dv_strides[0], dv_strides[1], dv_strides[2], dv_strides[3],
        inv_sqrt_d=inv_sqrt_d, BLOCK=BLOCK, D_HEAD_DIM=d, num_warps=8, num_stages=2
    )
    
    backward_dQ_kernel[grid](
        Q, K, V, O, dO, L, dQ, S,
        q_strides[0], q_strides[1], q_strides[2], q_strides[3],
        k_strides[0], k_strides[1], k_strides[2], k_strides[3],
        v_strides[0], v_strides[1], v_strides[2], v_strides[3],
        o_strides[0], o_strides[1], o_strides[2], o_strides[3],
        do_strides[0], do_strides[1], do_strides[2], do_strides[3],
        l_strides[0], l_strides[1], l_strides[2],
        dq_strides[0], dq_strides[1], dq_strides[2], dq_strides[3],
        inv_sqrt_d=inv_sqrt_d, BLOCK=BLOCK, D_HEAD_DIM=d, num_warps=8, num_stages=2
    )