import math
import torch
import triton
import triton.language as tl


@triton.jit
def load_2d(base_ptr, stride_row, mask_row, mask_col, col_major: tl.constexpr):
    row_off = tl.arange(0, 64)[:, None]
    col_off = tl.arange(0, 64)[None, :]
    if col_major:
        ptr = base_ptr + row_off * stride_row + col_off * 1
    else:
        ptr = base_ptr + row_off * 1 + col_off * stride_row
    mask = mask_row[:, None] & mask_col[None, :]
    return tl.load(ptr, mask=mask, other=0.0)


@triton.jit
def load_1d(base_ptr, mask):
    off = tl.arange(0, 64)
    return tl.load(base_ptr + off, mask=mask, other=0.0)


@triton.jit
def store_2d(base_ptr, val, stride_row, mask_row, mask_col, col_major: tl.constexpr):
    row_off = tl.arange(0, 64)[:, None]
    col_off = tl.arange(0, 64)[None, :]
    if col_major:
        ptr = base_ptr + row_off * stride_row + col_off * 1
    else:
        ptr = base_ptr + row_off * 1 + col_off * stride_row
    mask = mask_row[:, None] & mask_col[None, :]
    tl.store(ptr, val, mask=mask)


@triton.jit
def _mha_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, d, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b_h = tl.program_id(0)
    num_pid_m = tl.cdiv(S_len, BLOCK_M)
    start_pid_m = tl.program_id(1)
    
    for m_idx in tl.range(start_pid_m, num_pid_m, 1, flatten=True):
        offset_m = m_idx * BLOCK_M
        mask_r_m = (offset_m + tl.arange(0, BLOCK_M)) < S_len
        
        q_base = Q_ptr + pid_b_h * S_len * d + offset_m * d
        mask_col_0 = (0 + tl.arange(0, 64)) < 64
        mask_col_1 = (64 + tl.arange(0, 64)) < 64
        
        Q_m_0 = load_2d(q_base, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_0, col_major=True)
        Q_m_1 = load_2d(q_base + 64, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_1, col_major=True)
        
        do_base = dO_ptr + pid_b_h * S_len * d + offset_m * d
        dO_m_0 = load_2d(do_base, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_0, col_major=True)
        dO_m_1 = load_2d(do_base + 64, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_1, col_major=True)
        
        o_base = O_ptr + pid_b_h * S_len * d + offset_m * d
        O_m_0 = load_2d(o_base, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_0, col_major=True)
        O_m_1 = load_2d(o_base + 64, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_1, col_major=True)
        
        Q_m_0 = Q_m_0.to(tl.float32)
        Q_m_1 = Q_m_1.to(tl.float32)
        dO_m_0 = dO_m_0.to(tl.float32)
        dO_m_1 = dO_m_1.to(tl.float32)
        O_m_0 = O_m_0.to(tl.float32)
        O_m_1 = O_m_1.to(tl.float32)
        
        D_m = tl.sum(dO_m_0 * O_m_0, axis=1) + tl.sum(dO_m_1 * O_m_1, axis=1)
        D_m = D_m[:, None]
        
        L_base = L_ptr + pid_b_h * S_len + offset_m
        L_m = load_1d(L_base, mask_r_m)
        L_m = L_m[:, None]
        
        dq_acc_0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
        dq_acc_1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
        
        for n_idx in range(0, S_len, BLOCK_N):
            mask_r_n = (n_idx + tl.arange(0, BLOCK_N)) < S_len
            
            k_base = K_ptr + pid_b_h * S_len * d + n_idx * d
            K_n_0 = load_2d(k_base, stride_row=d, mask_row=mask_r_n, mask_col=mask_col_0, col_major=True)
            K_n_1 = load_2d(k_base + 64, stride_row=d, mask_row=mask_r_n, mask_col=mask_col_1, col_major=True)
            
            v_base = V_ptr + pid_b_h * S_len * d + n_idx * d
            V_n_0 = load_2d(v_base, stride_row=d, mask_row=mask_r_n, mask_col=mask_col_0, col_major=True)
            V_n_1 = load_2d(v_base + 64, stride_row=d, mask_row=mask_r_n, mask_col=mask_col_1, col_major=True)
            
            K_n_0 = K_n_0.to(tl.float32)
            K_n_1 = K_n_1.to(tl.float32)
            V_n_0 = V_n_0.to(tl.float32)
            V_n_1 = V_n_1.to(tl.float32)
            
            S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            S_acc = tl.dot(Q_m_0, K_n_0.T, S_acc)
            S_acc = tl.dot(Q_m_1, K_n_1.T, S_acc)
            
            S = S_acc * scale
            P = tl.exp(S - L_m)
            
            dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            dP_acc = tl.dot(dO_m_0, V_n_0.T, dP_acc)
            dP_acc = tl.dot(dO_m_1, V_n_1.T, dP_acc)
            
            dS = P * (dP_acc - D_m) * scale
            
            dq_acc_0 = tl.dot(dS, K_n_0, dq_acc_0)
            dq_acc_1 = tl.dot(dS, K_n_1, dq_acc_1)
        
        store_2d(dQ_ptr + pid_b_h * S_len * d + offset_m * d, dq_acc_0, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_0, col_major=True)
        store_2d(dQ_ptr + pid_b_h * S_len * d + offset_m * d + 64, dq_acc_1, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_1, col_major=True)


@triton.jit
def _mha_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, d, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b_h = tl.program_id(0)
    num_pid_n = tl.cdiv(S_len, BLOCK_N)
    start_pid_n = tl.program_id(1)
    
    for n_idx in tl.range(start_pid_n, num_pid_n, 1, flatten=True):
        offset_n = n_idx * BLOCK_N
        mask_r_n = (offset_n + tl.arange(0, BLOCK_N)) < S_len
        
        k_base = K_ptr + pid_b_h * S_len * d + offset_n * d
        mask_col_0 = (0 + tl.arange(0, 64)) < 64
        mask_col_1 = (64 + tl.arange(0, 64)) < 64
        
        K_n_0 = load_2d(k_base, stride_row=d, mask_row=mask_r_n, mask_col=mask_col_0, col_major=True)
        K_n_1 = load_2d(k_base + 64, stride_row=d, mask_row=mask_r_n, mask_col=mask_col_1, col_major=True)
        
        v_base = V_ptr + pid_b_h * S_len * d + offset_n * d
        V_n_0 = load_2d(v_base, stride_row=d, mask_row=mask_r_n, mask_col=mask_col_0, col_major=True)
        V_n_1 = load_2d(v_base + 64, stride_row=d, mask_row=mask_r_n, mask_col=mask_col_1, col_major=True)
        
        K_n_0 = K_n_0.to(tl.float32)
        K_n_1 = K_n_1.to(tl.float32)
        V_n_0 = V_n_0.to(tl.float32)
        V_n_1 = V_n_1.to(tl.float32)
        
        dk_acc_0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
        dk_acc_1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
        dv_acc_0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
        dv_acc_1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
        
        for m_idx in range(0, S_len, BLOCK_M):
            mask_r_m = (m_idx + tl.arange(0, BLOCK_M)) < S_len
            
            q_base = Q_ptr + pid_b_h * S_len * d + m_idx * d
            Q_m_0 = load_2d(q_base, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_0, col_major=True)
            Q_m_1 = load_2d(q_base + 64, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_1, col_major=True)
            
            do_base = dO_ptr + pid_b_h * S_len * d + m_idx * d
            dO_m_0 = load_2d(do_base, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_0, col_major=True)
            dO_m_1 = load_2d(do_base + 64, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_1, col_major=True)
            
            o_base = O_ptr + pid_b_h * S_len * d + m_idx * d
            O_m_0 = load_2d(o_base, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_0, col_major=True)
            O_m_1 = load_2d(o_base + 64, stride_row=d, mask_row=mask_r_m, mask_col=mask_col_1, col_major=True)
            
            Q_m_0 = Q_m_0.to(tl.float32)
            Q_m_1 = Q_m_1.to(tl.float32)
            dO_m_0 = dO_m_0.to(tl.float32)
            dO_m_1 = dO_m_1.to(tl.float32)
            O_m_0 = O_m_0.to(tl.float32)
            O_m_1 = O_m_1.to(tl.float32)
            
            D_m = tl.sum(dO_m_0 * O_m_0, axis=1) + tl.sum(dO_m_1 * O_m_1, axis=1)
            D_m = D_m[:, None]
            
            L_base = L_ptr + pid_b_h * S_len + m_idx
            L_m = load_1d(L_base, mask_r_m)
            L_m = L_m[:, None]
            
            S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            S_acc = tl.dot(Q_m_0, K_n_0.T, S_acc)
            S_acc = tl.dot(Q_m_1, K_n_1.T, S_acc)
            
            S = S_acc * scale
            P = tl.exp(S - L_m)
            
            dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            dP_acc = tl.dot(dO_m_0, V_n_0.T, dP_acc)
            dP_acc = tl.dot(dO_m_1, V_n_1.T, dP_acc)
            
            dS = P * (dP_acc - D_m) * scale
            
            pt = P.T
            dv_acc_0 = tl.dot(pt, dO_m_0, dv_acc_0)
            dv_acc_1 = tl.dot(pt, dO_m_1, dv_acc_1)
            
            ds_t = dS.T
            dk_acc_0 = tl.dot(ds_t, Q_m_0, dk_acc_0)
            dk_acc_1 = tl.dot(ds_t, Q_m_1, dk_acc_1)
        
        store_2d(dK_ptr + pid_b_h * S_len * d + offset_n * d, dk_acc_0, stride_row=d, mask_row=mask_r_n, mask_col=mask_col_0, col_major=True)
        store_2d(dK_ptr + pid_b_h * S_len * d + offset_n * d + 64, dk_acc_1, stride_row=d, mask_row=mask_r_n, mask_col=mask_col_1, col_major=True)
        store_2d(dV_ptr + pid_b_h * S_len * d + offset_n * d, dv_acc_0, stride_row=d, mask_row=mask_r_n, mask_col=mask_col_0, col_major=True)
        store_2d(dV_ptr + pid_b_h * S_len * d + offset_n * d + 64, dv_acc_1, stride_row=d, mask_row=mask_r_n, mask_col=mask_col_1, col_major=True)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    grid_dq = (B * H, triton.cdiv(S, 64))
    _mha_bwd_dq_kernel[grid_dq](
        Q, K, V, O, dO, L, dQ,
        S, d, scale,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=2,
    )
    
    grid_dkv = (B * H, triton.cdiv(S, 64))
    _mha_bwd_dkv_kernel[grid_dkv](
        Q, K, V, O, dO, L, dK, dV,
        S, d, scale,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=2,
    )