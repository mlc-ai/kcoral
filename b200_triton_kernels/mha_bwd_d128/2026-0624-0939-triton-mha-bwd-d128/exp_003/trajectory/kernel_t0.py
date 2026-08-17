import math
import torch
import triton
import triton.language as tl


@triton.jit
def load_2d(base_ptr, stride_row, stride_col, mask_row):
    row_off = tl.arange(0, 64)[:, None]
    col_off = tl.arange(0, 64)[None, :]
    ptr = base_ptr + row_off * stride_row + col_off * stride_col
    return tl.load(ptr, mask=mask_row[:, None], other=0.0)


@triton.jit
def load_1d(base_ptr, mask):
    off = tl.arange(0, 64)
    return tl.load(base_ptr + off, mask=mask, other=0.0)


@triton.jit
def store_2d(base_ptr, val, stride_row, stride_col, mask_row):
    row_off = tl.arange(0, 64)[:, None]
    col_off = tl.arange(0, 64)[None, :]
    ptr = base_ptr + row_off * stride_row + col_off * stride_col
    tl.store(ptr, val, mask=mask_row[:, None])


@triton.jit
def mha_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    S_len, d, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_b_h = tl.program_id(0)
    
    L_base = L_ptr + pid_b_h * S_len
    
    # Phase 1: Compute dQ
    for m_idx in range(0, S_len, BLOCK_M):
        mask_r_m = (m_idx + tl.arange(0, BLOCK_M)) < S_len
        
        Q_m_0 = load_2d(Q_ptr + pid_b_h * S_len * d + m_idx * d, stride_row=d, stride_col=1, mask_row=mask_r_m)
        Q_m_1 = load_2d(Q_ptr + pid_b_h * S_len * d + m_idx * d + 64, stride_row=d, stride_col=1, mask_row=mask_r_m)
        dO_m_0 = load_2d(dO_ptr + pid_b_h * S_len * d + m_idx * d, stride_row=d, stride_col=1, mask_row=mask_r_m)
        dO_m_1 = load_2d(dO_ptr + pid_b_h * S_len * d + m_idx * d + 64, stride_row=d, stride_col=1, mask_row=mask_r_m)
        O_m_0 = load_2d(O_ptr + pid_b_h * S_len * d + m_idx * d, stride_row=d, stride_col=1, mask_row=mask_r_m)
        O_m_1 = load_2d(O_ptr + pid_b_h * S_len * d + m_idx * d + 64, stride_row=d, stride_col=1, mask_row=mask_r_m)
        
        D_m = tl.sum(dO_m_0 * O_m_0, axis=1) + tl.sum(dO_m_1 * O_m_1, axis=1)
        D_m = D_m[:, None]
        
        L_m = load_1d(L_base + m_idx, mask_r_m)
        L_m = L_m[:, None]
        
        dq_acc_0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
        dq_acc_1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
        
        for n_idx in range(0, S_len, BLOCK_N):
            mask_r_n = (n_idx + tl.arange(0, BLOCK_N)) < S_len
            
            K_n_0 = load_2d(K_ptr + pid_b_h * S_len * d + n_idx * d, stride_row=d, stride_col=1, mask_row=mask_r_n)
            K_n_1 = load_2d(K_ptr + pid_b_h * S_len * d + n_idx * d + 64, stride_row=d, stride_col=1, mask_row=mask_r_n)
            V_n_0 = load_2d(V_ptr + pid_b_h * S_len * d + n_idx * d, stride_row=d, stride_col=1, mask_row=mask_r_n)
            V_n_1 = load_2d(V_ptr + pid_b_h * S_len * d + n_idx * d + 64, stride_row=d, stride_col=1, mask_row=mask_r_n)
            
            S_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            S_acc = tl.dot(Q_m_0, K_n_0.T, S_acc)
            S_acc = tl.dot(Q_m_1, K_n_1.T, S_acc)
            S = S_acc * scale
            
            P = tl.exp(S - L_m)
            
            dP_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            dP_acc = tl.dot(dO_m_0, V_n_0.T, dP_acc)
            dP_acc = tl.dot(dO_m_1, V_n_1.T, dP_acc)
            
            dS = P * (dP_acc - D_m) * scale
            
            for k in range(0, BLOCK_N, 16):
                ds_chunk = dS[:, k:k+16]
                dq_acc_0 = tl.dot(ds_chunk, K_n_0[k:k+16, :], dq_acc_0)
                dq_acc_1 = tl.dot(ds_chunk, K_n_1[k:k+16, :], dq_acc_1)
        
        store_2d(dQ_ptr + pid_b_h * S_len * d + m_idx * d, dq_acc_0, stride_row=d, stride_col=1, mask_row=mask_r_m)
        store_2d(dQ_ptr + pid_b_h * S_len * d + m_idx * d + 64, dq_acc_1, stride_row=d, stride_col=1, mask_row=mask_r_m)

    # Phase 2: Compute dK, dV
    for n_idx in range(0, S_len, BLOCK_N):
        mask_r_n = (n_idx + tl.arange(0, BLOCK_N)) < S_len
        
        K_n_0 = load_2d(K_ptr + pid_b_h * S_len * d + n_idx * d, stride_row=d, stride_col=1, mask_row=mask_r_n)
        K_n_1 = load_2d(K_ptr + pid_b_h * S_len * d + n_idx * d + 64, stride_row=d, stride_col=1, mask_row=mask_r_n)
        V_n_0 = load_2d(V_ptr + pid_b_h * S_len * d + n_idx * d, stride_row=d, stride_col=1, mask_row=mask_r_n)
        V_n_1 = load_2d(V_ptr + pid_b_h * S_len * d + n_idx * d + 64, stride_row=d, stride_col=1, mask_row=mask_r_n)
        
        dk_acc_0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
        dk_acc_1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
        dv_acc_0 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
        dv_acc_1 = tl.zeros((BLOCK_N, 64), dtype=tl.float32)
        
        for m_idx in range(0, S_len, BLOCK_M):
            mask_r_m = (m_idx + tl.arange(0, BLOCK_M)) < S_len
            
            Q_m_0 = load_2d(Q_ptr + pid_b_h * S_len * d + m_idx * d, stride_row=d, stride_col=1, mask_row=mask_r_m)
            Q_m_1 = load_2d(Q_ptr + pid_b_h * S_len * d + m_idx * d + 64, stride_row=d, stride_col=1, mask_row=mask_r_m)
            dO_m_0 = load_2d(dO_ptr + pid_b_h * S_len * d + m_idx * d, stride_row=d, stride_col=1, mask_row=mask_r_m)
            dO_m_1 = load_2d(dO_ptr + pid_b_h * S_len * d + m_idx * d + 64, stride_row=d, stride_col=1, mask_row=mask_r_m)
            O_m_0 = load_2d(O_ptr + pid_b_h * S_len * d + m_idx * d, stride_row=d, stride_col=1, mask_row=mask_r_m)
            O_m_1 = load_2d(O_ptr + pid_b_h * S_len * d + m_idx * d + 64, stride_row=d, stride_col=1, mask_row=mask_r_m)
            
            D_m = tl.sum(dO_m_0 * O_m_0, axis=1) + tl.sum(dO_m_1 * O_m_1, axis=1)
            D_m = D_m[:, None]
            
            L_m = load_1d(L_base + m_idx, mask_r_m)
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
            
            for k in range(0, BLOCK_M, 16):
                pt_chunk = P.T[:, k:k+16]
                dv_acc_0 = tl.dot(pt_chunk, dO_m_0[k:k+16, :], dv_acc_0)
                dv_acc_1 = tl.dot(pt_chunk, dO_m_1[k:k+16, :], dv_acc_1)
            
            for k in range(0, BLOCK_M, 16):
                ds_t_chunk = dS.T[:, k:k+16]
                dk_acc_0 = tl.dot(ds_t_chunk, Q_m_0[k:k+16, :], dk_acc_0)
                dk_acc_1 = tl.dot(ds_t_chunk, Q_m_1[k:k+16, :], dk_acc_1)
        
        store_2d(dK_ptr + pid_b_h * S_len * d + n_idx * d, dk_acc_0, stride_row=d, stride_col=1, mask_row=mask_r_n)
        store_2d(dK_ptr + pid_b_h * S_len * d + n_idx * d + 64, dk_acc_1, stride_row=d, stride_col=1, mask_row=mask_r_n)
        store_2d(dV_ptr + pid_b_h * S_len * d + n_idx * d, dv_acc_0, stride_row=d, stride_col=1, mask_row=mask_r_n)
        store_2d(dV_ptr + pid_b_h * S_len * d + n_idx * d + 64, dv_acc_1, stride_row=d, stride_col=1, mask_row=mask_r_n)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    grid = (B * H,)
    mha_bwd_kernel[grid](
        Q, K, V, O, dO, L,
        dQ, dK, dV,
        S, d, scale,
        BLOCK_M=64, BLOCK_N=64,
        num_warps=4, num_stages=2,
    )