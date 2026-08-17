import math
import torch
import triton
import triton.language as tl


@triton.jit
def load_3d(base_ptr, stride_row):
    row_off = tl.arange(0, 64)[:, None]
    col_off = tl.arange(0, 64)[None, :]
    ptr = base_ptr + row_off * stride_row + col_off * 1
    return tl.load(ptr, mask=mask, other=0.0)


@triton.jit
def _mha_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, d, scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    b = b_h_idx // H
    h = b_h_idx % H
    
    offset_m = pid_m * BLOCK_M
    mask_m = (offset_m + tl.arange(0, BLOCK_M)) < S_len
    
    B_stride = H * S_len * d
    H_stride = S_len * d
    
    Q_base_0 = Q_ptr + b * B_stride + h * H_stride + offset_m * d + 0
    Q_m_0 = load_3d(Q_base_0, stride_row=d)
    Q_base_1 = Q_ptr + b * B_stride + h * H_stride + offset_m * d + 64
    Q_m_1 = load_3d(Q_base_1, stride_row=d)
    
    dO_base_0 = dO_ptr + b * B_stride + h * H_stride + offset_m * d + 0
    dO_m_0 = load_3d(dO_base_0, stride_row=d)
    dO_base_1 = dO_ptr + b * B_stride + h * H_stride + offset_m * d + 64
    dO_m_1 = load_3d(dO_base_1, stride_row=d)
    
    O_base_0 = O_ptr + b * B_stride + h * H_stride + offset_m * d + 0
    O_m_0 = load_3d(O_base_0, stride_row=d)
    O_base_1 = O_ptr + b * B_stride + h * H_stride + offset_m * d + 64
    O_m_1 = load_3d(O_base_1, stride_row=d)
    
    Q_m_0 = Q_m_0.to(tl.float32)
    Q_m_1 = Q_m_1.to(tl.float32)
    dO_m_0 = dO_m_0.to(tl.float32)
    dO_m_1 = dO_m_1.to(tl.float32)
    O_m_0 = O_m_0.to(tl.float32)
    O_m_1 = O_m_1.to(tl.float32)
    
    D_m = tl.sum(dO_m_0 * O_m_0, axis=1) + tl.sum(dO_m_1 * O_m_1, axis=1)
    D_m = D_m[:, None]
    
    L_base = L_ptr + b * (H * S_len) + h * S_len + offset_m
    L_m = tl.load(L_base + tl.arange(0, BLOCK_M), mask=mask_m, other=0.0)
    L_m = L_m[:, None]
    
    acc_dQ_0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    acc_dQ_1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    for n_idx in range(0, S_len, BLOCK_N):
        offset_n = n_idx * BLOCK_N
        mask_n = (offset_n + tl.arange(0, BLOCK_N)) < S_len
        
        K_base_0 = K_ptr + b * B_stride + h * H_stride + offset_n * d + 0
        K_n_0 = load_3d(K_base_0, stride_row=d)
        K_base_1 = K_ptr + b * B_stride + h * H_stride + offset_n * d + 64
        K_n_1 = load_3d(K_base_1, stride_row=d)
        
        V_base_0 = V_ptr + b * B_stride + h * H_stride + offset_n * d + 0
        V_n_0 = load_3d(V_base_0, stride_row=d)
        V_base_1 = V_ptr + b * B_stride + h * H_stride + offset_n * d + 64
        V_n_1 = load_3d(V_base_1, stride_row=d)
        
        K_n_0 = K_n_0.to(tl.float32)
        K_n_1 = K_n_1.to(tl.float32)
        V_n_0 = V_n_0.to(tl.float32)
        V_n_1 = V_n_1.to(tl.float32)
        
        S_acc = tl.dot(Q_m_0, K_n_0.T) + tl.dot(Q_m_1, K_n_1.T)
        S = S_acc * scale
        
        P = tl.exp(S - L_m)
        
        dP_acc = tl.dot(dO_m_0, V_n_0.T) + tl.dot(dO_m_1, V_n_1.T)
        dS = P * (dP_acc - D_m) * scale
        
        acc_dQ_0 = tl.dot(dS, K_n_0, acc_dQ_0)
        acc_dQ_1 = tl.dot(dS, K_n_1, acc_dQ_1)
    
    dq_out_0 = acc_dQ_0.to(tl.bfloat16)
    dq_out_1 = acc_dQ_1.to(tl.bfloat16)
    
    mask_2d = mask_m[:, None]
    ptr_0 = dQ_ptr + b * B_stride + h * H_stride + offset_m * d + 0
    row_off = tl.arange(0, 64)[:, None]
    col_off = tl.arange(0, 64)[None, :]
    final_ptr_0 = ptr_0 + row_off * d + col_off * 1
    tl.store(final_ptr_0, dq_out_0, mask=mask_2d)
    
    ptr_1 = dQ_ptr + b * B_stride + h * H_stride + offset_m * d + 64
    final_ptr_1 = ptr_1 + row_off * d + col_off * 1
    tl.store(final_ptr_1, dq_out_1, mask=mask_2d)


@triton.jit
def _mha_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, d, scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    b = b_h_idx // H
    h = b_h_idx % H
    
    offset_n = pid_n * BLOCK_N
    mask_n = (offset_n + tl.arange(0, BLOCK_N)) < S_len
    
    B_stride = H * S_len * d
    H_stride = S_len * d
    
    K_base_0 = K_ptr + b * B_stride + h * H_stride + offset_n * d + 0
    K_n_0 = load_3d(K_base_0, stride_row=d)
    K_base_1 = K_ptr + b * B_stride + h * H_stride + offset_n * d + 64
    K_n_1 = load_3d(K_base_1, stride_row=d)
    
    V_base_0 = V_ptr + b * B_stride + h * H_stride + offset_n * d + 0
    V_n_0 = load_3d(V_base_0, stride_row=d)
    V_base_1 = V_ptr + b * B_stride + h * H_stride + offset_n * d + 64
    V_n_1 = load_3d(V_base_1, stride_row=d)
    
    K_n_0 = K_n_0.to(tl.float32)
    K_n_1 = K_n_1.to(tl.float32)
    V_n_0 = V_n_0.to(tl.float32)
    V_n_1 = V_n_1.to(tl.float32)
    
    acc_dK_0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    acc_dK_1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    acc_dV_0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    acc_dV_1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
    
    for m_idx in range(0, S_len, BLOCK_M):
        offset_m = m_idx * BLOCK_M
        mask_m = (offset_m + tl.arange(0, BLOCK_M)) < S_len
        
        Q_base_0 = Q_ptr + b * B_stride + h * H_stride + offset_m * d + 0
        Q_m_0 = load_3d(Q_base_0, stride_row=d)
        Q_base_1 = Q_ptr + b * B_stride + h * H_stride + offset_m * d + 64
        Q_m_1 = load_3d(Q_base_1, stride_row=d)
        
        dO_base_0 = dO_ptr + b * B_stride + h * H_stride + offset_m * d + 0
        dO_m_0 = load_3d(dO_base_0, stride_row=d)
        dO_base_1 = dO_ptr + b * B_stride + h * H_stride + offset_m * d + 64
        dO_m_1 = load_3d(dO_base_1, stride_row=d)
        
        O_base_0 = O_ptr + b * B_stride + h * H_stride + offset_m * d + 0
        O_m_0 = load_3d(O_base_0, stride_row=d)
        O_base_1 = O_ptr + b * B_stride + h * H_stride + offset_m * d + 64
        O_m_1 = load_3d(O_base_1, stride_row=d)
        
        Q_m_0 = Q_m_0.to(tl.float32)
        Q_m_1 = Q_m_1.to(tl.float32)
        dO_m_0 = dO_m_0.to(tl.float32)
        dO_m_1 = dO_m_1.to(tl.float32)
        O_m_0 = O_m_0.to(tl.float32)
        O_m_1 = O_m_1.to(tl.float32)
        
        D_m = tl.sum(dO_m_0 * O_m_0, axis=1) + tl.sum(dO_m_1 * O_m_1, axis=1)
        D_m = D_m[:, None]
        
        L_base = L_ptr + b * (H * S_len) + h * S_len + offset_m
        L_m = tl.load(L_base + tl.arange(0, BLOCK_M), mask=mask_m, other=0.0)
        L_m = L_m[:, None]
        
        S_acc = tl.dot(Q_m_0, K_n_0.T) + tl.dot(Q_m_1, K_n_1.T)
        S = S_acc * scale
        
        P = tl.exp(S - L_m)
        
        dP_acc = tl.dot(dO_m_0, V_n_0.T) + tl.dot(dO_m_1, V_n_1.T)
        dS = P * (dP_acc - D_m) * scale
        
        acc_dV_0 = tl.dot(P.T, dO_m_0, acc_dV_0)
        acc_dV_1 = tl.dot(P.T, dO_m_1, acc_dV_1)
        
        acc_dK_0 = tl.dot(dS.T, Q_m_0, acc_dK_0)
        acc_dK_1 = tl.dot(dS.T, Q_m_1, acc_dK_1)
    
    dk_out_0 = acc_dK_0.to(tl.bfloat16)
    dk_out_1 = acc_dK_1.to(tl.bfloat16)
    dv_out_0 = acc_dV_0.to(tl.bfloat16)
    dv_out_1 = acc_dV_1.to(tl.bfloat16)
    
    mask_2d = mask_n[:, None]
    row_off = tl.arange(0, 64)[:, None]
    col_off = tl.arange(0, 64)[None, :]
    
    ptr_k0 = dK_ptr + b * B_stride + h * H_stride + offset_n * d + 0
    final_ptr_k0 = ptr_k0 + row_off * d + col_off * 1
    tl.store(final_ptr_k0, dk_out_0, mask=mask_2d)
    
    ptr_k1 = dK_ptr + b * B_stride + h * H_stride + offset_n * d + 64
    final_ptr_k1 = ptr_k1 + row_off * d + col_off * 1
    tl.store(final_ptr_k1, dk_out_1, mask=mask_2d)
    
    ptr_v0 = dV_ptr + b * B_stride + h * H_stride + offset_n * d + 0
    final_ptr_v0 = ptr_v0 + row_off * d + col_off * 1
    tl.store(final_ptr_v0, dv_out_0, mask=mask_2d)
    
    ptr_v1 = dV_ptr + b * B_stride + h * H_stride + offset_n * d + 64
    final_ptr_v1 = ptr_v1 + row_off * d + col_off * 1
    tl.store(final_ptr_v1, dv_out_1, mask=mask_2d)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    grid_dq = (B * H, triton.cdiv(S, 64))
    _mha_bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        S, d, scale, H,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=8, num_stages=3,
    )
    
    grid_dkv = (B * H, triton.cdiv(S, 64))
    _mha_bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        S, d, scale, H,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=8, num_stages=3,
    )