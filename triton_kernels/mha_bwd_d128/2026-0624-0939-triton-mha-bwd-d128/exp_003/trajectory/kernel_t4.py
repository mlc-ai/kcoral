import math
import torch
import triton
import triton.language as tl


@triton.jit
def load_2d(base_ptr, stride_row, mask_row, mask_col):
    row_off = tl.arange(0, 64)[:, None]
    col_off = tl.arange(0, 64)[None, :]
    ptr = base_ptr + row_off * stride_row + col_off * 1
    mask = mask_row[:, None] & mask_col[None, :]
    return tl.load(ptr, mask=mask, other=0.0)


@triton.jit
def store_2d(base_ptr, val, stride_row, mask_row, mask_col):
    row_off = tl.arange(0, 64)[:, None]
    col_off = tl.arange(0, 64)[None, :]
    ptr = base_ptr + row_off * stride_row + col_off * 1
    mask = mask_row[:, None] & mask_col[None, :]
    tl.store(ptr, val, mask=mask)


@triton.jit
def compute_D_kernel(dO_ptr, O_ptr, D_ptr, S_len, d):
    b_h_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    if row_idx < S_len:
        off = tl.arange(0, 64)
        base = dO_ptr + b_h_idx * S_len * d + row_idx * d
        dO_0 = tl.load(base + off, mask=(row_idx < S_len), other=0.0)
        dO_1 = tl.load(base + 64 + off, mask=(row_idx < S_len), other=0.0)
        
        base_o = O_ptr + b_h_idx * S_len * d + row_idx * d
        O_0 = tl.load(base_o + off, mask=(row_idx < S_len), other=0.0)
        O_1 = tl.load(base_o + 64 + off, mask=(row_idx < S_len), other=0.0)
        
        D_val = tl.sum(dO_0 * O_0) + tl.sum(dO_1 * O_1)
        tl.store(D_ptr + b_h_idx * S_len + row_idx, D_val)


@triton.jit
def _mha_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    S_len, d, scale, H,
    BLOCK: tl.constexpr,
):
    b_h_idx = tl.program_id(0)
    m_idx = tl.program_id(1)
    
    offset_m = m_idx * BLOCK
    mask_m = (offset_m + tl.arange(0, BLOCK)) < S_len
    
    # =====================================================
    # Phase 1: Compute dQ
    # =====================================================
    Q_m_0 = load_2d(Q_ptr + b_h_idx * S_len * d + offset_m * d, stride_row=d, mask_row=mask_m, mask_col=(0 + tl.arange(0, 64)) < 64)
    Q_m_1 = load_2d(Q_ptr + b_h_idx * S_len * d + offset_m * d + 64, stride_row=d, mask_row=mask_m, mask_col=(64 + tl.arange(0, 64)) < 64)
    
    dO_m_0 = load_2d(dO_ptr + b_h_idx * S_len * d + offset_m * d, stride_row=d, mask_row=mask_m, mask_col=(0 + tl.arange(0, 64)) < 64)
    dO_m_1 = load_2d(dO_ptr + b_h_idx * S_len * d + offset_m * d + 64, stride_row=d, mask_row=mask_m, mask_col=(64 + tl.arange(0, 64)) < 64)
    
    Q_m_0 = Q_m_0.to(tl.float32)
    Q_m_1 = Q_m_1.to(tl.float32)
    dO_m_0 = dO_m_0.to(tl.float32)
    dO_m_1 = dO_m_1.to(tl.float32)
    
    L_m = tl.load(L_ptr + b_h_idx * S_len + offset_m + tl.arange(0, 64), mask=mask_m, other=0.0)
    L_m = L_m[:, None]
    
    D_m = tl.load(D_ptr + b_h_idx * S_len + offset_m + tl.arange(0, 64), mask=mask_m, other=0.0)
    D_m = D_m[:, None]
    
    acc_dQ_0 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    acc_dQ_1 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    
    for n_idx in range(0, S_len, BLOCK):
        offset_n = n_idx * BLOCK
        mask_n = (offset_n + tl.arange(0, BLOCK)) < S_len
        
        K_n_0 = load_2d(K_ptr + b_h_idx * S_len * d + offset_n * d, stride_row=d, mask_row=mask_n, mask_col=(0 + tl.arange(0, 64)) < 64)
        K_n_1 = load_2d(K_ptr + b_h_idx * S_len * d + offset_n * d + 64, stride_row=d, mask_row=mask_n, mask_col=(64 + tl.arange(0, 64)) < 64)
        V_n_0 = load_2d(V_ptr + b_h_idx * S_len * d + offset_n * d, stride_row=d, mask_row=mask_n, mask_col=(0 + tl.arange(0, 64)) < 64)
        V_n_1 = load_2d(V_ptr + b_h_idx * S_len * d + offset_n * d + 64, stride_row=d, mask_row=mask_n, mask_col=(64 + tl.arange(0, 64)) < 64)
        
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
    
    store_2d(dQ_ptr + b_h_idx * S_len * d + offset_m * d, dq_out_0, stride_row=d, mask_row=mask_m, mask_col=(0 + tl.arange(0, 64)) < 64)
    store_2d(dQ_ptr + b_h_idx * S_len * d + offset_m * d + 64, dq_out_1, stride_row=d, mask_row=mask_m, mask_col=(64 + tl.arange(0, 64)) < 64)
    
    # =====================================================
    # Phase 2: Compute dK and dV
    # =====================================================
    n_idx = m_idx 
    offset_n = n_idx * BLOCK
    mask_n = (offset_n + tl.arange(0, BLOCK)) < S_len
    
    K_n_0 = load_2d(K_ptr + b_h_idx * S_len * d + offset_n * d, stride_row=d, mask_row=mask_n, mask_col=(0 + tl.arange(0, 64)) < 64)
    K_n_1 = load_2d(K_ptr + b_h_idx * S_len * d + offset_n * d + 64, stride_row=d, mask_row=mask_n, mask_col=(64 + tl.arange(0, 64)) < 64)
    V_n_0 = load_2d(V_ptr + b_h_idx * S_len * d + offset_n * d, stride_row=d, mask_row=mask_n, mask_col=(0 + tl.arange(0, 64)) < 64)
    V_n_1 = load_2d(V_ptr + b_h_idx * S_len * d + offset_n * d + 64, stride_row=d, mask_row=mask_n, mask_col=(64 + tl.arange(0, 64)) < 64)
    
    K_n_0 = K_n_0.to(tl.float32)
    K_n_1 = K_n_1.to(tl.float32)
    V_n_0 = V_n_0.to(tl.float32)
    V_n_1 = V_n_1.to(tl.float32)
    
    acc_dK_0 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    acc_dK_1 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    acc_dV_0 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    acc_dV_1 = tl.zeros((BLOCK, 64), dtype=tl.float32)
    
    for m_loop_idx in range(0, S_len, BLOCK):
        offset_m = m_loop_idx * BLOCK
        mask_m = (offset_m + tl.arange(0, BLOCK)) < S_len
        
        Q_m_0 = load_2d(Q_ptr + b_h_idx * S_len * d + offset_m * d, stride_row=d, mask_row=mask_m, mask_col=(0 + tl.arange(0, 64)) < 64)
        Q_m_1 = load_2d(Q_ptr + b_h_idx * S_len * d + offset_m * d + 64, stride_row=d, mask_row=mask_m, mask_col=(64 + tl.arange(0, 64)) < 64)
        dO_m_0 = load_2d(dO_ptr + b_h_idx * S_len * d + offset_m * d, stride_row=d, mask_row=mask_m, mask_col=(0 + tl.arange(0, 64)) < 64)
        dO_m_1 = load_2d(dO_ptr + b_h_idx * S_len * d + offset_m * d + 64, stride_row=d, mask_row=mask_m, mask_col=(64 + tl.arange(0, 64)) < 64)
        
        Q_m_0 = Q_m_0.to(tl.float32)
        Q_m_1 = Q_m_1.to(tl.float32)
        dO_m_0 = dO_m_0.to(tl.float32)
        dO_m_1 = dO_m_1.to(tl.float32)
        
        L_m = tl.load(L_ptr + b_h_idx * S_len + offset_m + tl.arange(0, 64), mask=mask_m, other=0.0)
        L_m = L_m[:, None]
        
        D_m = tl.load(D_ptr + b_h_idx * S_len + offset_m + tl.arange(0, 64), mask=mask_m, other=0.0)
        D_m = D_m[:, None]
        
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
    
    store_2d(dK_ptr + b_h_idx * S_len * d + offset_n * d, dk_out_0, stride_row=d, mask_row=mask_n, mask_col=(0 + tl.arange(0, 64)) < 64)
    store_2d(dK_ptr + b_h_idx * S_len * d + offset_n * d + 64, dk_out_1, stride_row=d, mask_row=mask_n, mask_col=(64 + tl.arange(0, 64)) < 64)
    store_2d(dV_ptr + b_h_idx * S_len * d + offset_n * d, dv_out_0, stride_row=d, mask_row=mask_n, mask_col=(0 + tl.arange(0, 64)) < 64)
    store_2d(dV_ptr + b_h_idx * S_len * d + offset_n * d + 64, dv_out_1, stride_row=d, mask_row=mask_n, mask_col=(64 + tl.arange(0, 64)) < 64)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    L_float = L.float()
    D_buffer = torch.empty((B, H, S), dtype=torch.float32, device=Q.device)
    
    grid_D = (B * H, S)
    compute_D_kernel[grid_D](dO, O, D_buffer, S, d)
    
    BLOCK = 64
    grid_main = (B * H, triton.cdiv(S, BLOCK))
    _mha_bwd_kernel[grid_main](
        Q, K, V, dO, L_float, D_buffer,
        dQ, dK, dV,
        S, d, scale, H,
        BLOCK=BLOCK,
        num_warps=4, num_stages=3,
    )