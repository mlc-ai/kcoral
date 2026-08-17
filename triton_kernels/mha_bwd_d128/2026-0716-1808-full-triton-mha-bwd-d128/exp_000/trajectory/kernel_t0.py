import torch
import triton
import triton.language as tl
import math


@triton.jit
def load_tile_128x128(base_ptr, b_h, row_start, col_start, S, ptr_shared):
    row_idx = tl.program_id(1)
    col_idx = tl.program_id(2)
    idx = row_idx * 128 + col_idx
    
    s_idx = row_start + row_idx
    d_idx = col_start + col_idx
    
    if s_idx < S and d_idx < 128:
        g_idx = (b_h * S + s_idx) * 128 + d_idx
        val = tl.load(base_ptr + g_idx)
        ptr_shared[idx] = val.to(tl.float32)
    else:
        ptr_shared[idx] = 0.0
    
    tl.mem_fence()


@triton.jit
def mha_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, sqrt_d, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_q = tl.program_id(0)
    b_h = tl.program_id(1)
    q_rows = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
    d_cols = tl.arange(0, 128)

    q_ptr = Q_ptr + (b_h * S_len + q_rows[:, None]) * 128 + d_cols[None, :]
    do_ptr = dO_ptr + (b_h * S_len + q_rows[:, None]) * 128 + d_cols[None, :]
    q_ptr_shared = ptr_shared("q_ptr", (BLOCK_M, 128))
    do_ptr_shared = ptr_shared("do_ptr", (BLOCK_M, 128))
    load_tile_128x128(Q_ptr, b_h, pid_q * BLOCK_M, 0, S_len, q_ptr_shared)
    load_tile_128x128(dO_ptr, b_h, pid_q * BLOCK_M, 0, S_len, do_ptr_shared)

    l_ptr_q = L_ptr + b_h * S_len + q_rows
    l_q = tl.load(l_ptr_q, mask=(q_rows < S_len), other=0.0)

    acc_dQ = tl.zeros((BLOCK_M, 128), tl.float32)
    num_k_tiles = tl.cdiv(S_len, BLOCK_N)
    
    for k_tile in range(num_k_tiles):
        k_rows = k_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        k_ptr = K_ptr + (b_h * S_len + k_rows[:, None]) * 128 + d_cols[None, :]
        v_ptr = V_ptr + (b_h * S_len + k_rows[:, None]) * 128 + d_cols[None, :]
        k_ptr_shared = ptr_shared("k_ptr", (BLOCK_N, 128))
        v_ptr_shared = ptr_shared("v_ptr", (BLOCK_N, 128))
        load_tile_128x128(K_ptr, b_h, k_tile * BLOCK_N, 0, S_len, k_ptr_shared)
        load_tile_128x128(V_ptr, b_h, k_tile * BLOCK_N, 0, S_len, v_ptr_shared)

        l_ptr_k = L_ptr + b_h * S_len + k_rows
        l_k = tl.load(l_ptr_k, mask=(k_rows < S_len), other=0.0)

        temp_s_ptr = ptr_shared("temp_s_ptr", (BLOCK_M, BLOCK_N))
        temp_ds_ptr = ptr_shared("temp_ds_ptr", (BLOCK_M, BLOCK_N))

        for idx_s in range(BLOCK_M * BLOCK_N):
            row_idx = idx_s // BLOCK_N
            col_idx = idx_s % BLOCK_N
            s_val = 0.0
            for i in range(128):
                s_val += q_ptr_shared[row_idx, i] * k_ptr_shared[col_idx, i]
            s_val /= sqrt_d
            
            p_val = math.exp(s_val - l_q[row_idx])
            
            ds_val = 0.0
            for i in range(128):
                ds_val += do_ptr_shared[row_idx, i] * v_ptr_shared[col_idx, i]
            ds_val *= p_val
            
            temp_s_ptr[idx_s] = s_val
            temp_ds_ptr[idx_s] = ds_val
        
        for idx_acc in range(BLOCK_M * 128):
            row_idx = idx_acc // 128
            col_idx = idx_acc % 128
            acc_val = 0.0
            for j in range(BLOCK_N):
                acc_val += temp_ds_ptr[row_idx, j] * k_ptr_shared[j, col_idx]
            acc_dQ[row_idx, col_idx] += acc_val
        
        sync()

    for idx_out in range(BLOCK_M * 128):
        row_idx = idx_out // 128
        col_idx = idx_out % 128
        if q_rows[row_idx] < S_len:
            out_ptr = dQ_ptr + ((b_h * S_len + q_rows[row_idx]) * 128 + col_idx)
            tl.store(out_ptr, (acc_dQ[row_idx, col_idx] / sqrt_d).to(tl.bfloat16))


@triton.jit
def mha_bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, sqrt_d, BLOCK_K: tl.constexpr, BLOCK_Q: tl.constexpr
):
    pid_k = tl.program_id(0)
    b_h = tl.program_id(1)
    k_rows = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    d_cols = tl.arange(0, 128)

    k_ptr = K_ptr + (b_h * S_len + k_rows[:, None]) * 128 + d_cols[None, :]
    v_ptr = V_ptr + (b_h * S_len + k_rows[:, None]) * 128 + d_cols[None, :]
    k_ptr_shared = ptr_shared("k_ptr", (BLOCK_K, 128))
    v_ptr_shared = ptr_shared("v_ptr", (BLOCK_K, 128))
    load_tile_128x128(K_ptr, b_h, pid_k * BLOCK_K, 0, S_len, k_ptr_shared)
    load_tile_128x128(V_ptr, b_h, pid_k * BLOCK_K, 0, S_len, v_ptr_shared)

    l_ptr_k = L_ptr + b_h * S_len + k_rows
    l_k = tl.load(l_ptr_k, mask=(k_rows < S_len), other=0.0)

    acc_dK = tl.zeros((BLOCK_K, 128), tl.float32)
    acc_dV = tl.zeros((BLOCK_K, 128), tl.float32)
    
    num_q_tiles = tl.cdiv(S_len, BLOCK_Q)
    for q_tile in range(num_q_tiles):
        q_rows = q_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_ptr = Q_ptr + (b_h * S_len + q_rows[:, None]) * 128 + d_cols[None, :]
        do_ptr = dO_ptr + (b_h * S_len + q_rows[:, None]) * 128 + d_cols[None, :]
        q_ptr_shared = ptr_shared("q_ptr", (BLOCK_Q, 128))
        do_ptr_shared = ptr_shared("do_ptr", (BLOCK_Q, 128))
        load_tile_128x128(Q_ptr, b_h, q_tile * BLOCK_Q, 0, S_len, q_ptr_shared)
        load_tile_128x128(dO_ptr, b_h, q_tile * BLOCK_Q, 0, S_len, do_ptr_shared)

        l_ptr_q = L_ptr + b_h * S_len + q_rows
        l_q = tl.load(l_ptr_q, mask=(q_rows < S_len), other=0.0)

        temp_s_T_ptr = ptr_shared("temp_s_T_ptr", (BLOCK_K, BLOCK_Q))
        temp_p_T_ptr = ptr_shared("temp_p_T_ptr", (BLOCK_K, BLOCK_Q))

        for idx_s in range(BLOCK_K * BLOCK_Q):
            row_idx = idx_s // BLOCK_Q
            col_idx = idx_s % BLOCK_Q
            s_val = 0.0
            for i in range(128):
                s_val += q_ptr_shared[col_idx, i] * k_ptr_shared[row_idx, i]
            s_val /= sqrt_d
            
            p_val = math.exp(s_val - l_q[col_idx])
            
            ds_val = 0.0
            for i in range(128):
                ds_val += do_ptr_shared[col_idx, i] * v_ptr_shared[row_idx, i]
            ds_val *= p_val
            
            temp_s_T_ptr[idx_s] = s_val
            temp_p_T_ptr[idx_s] = p_val
            temp_ds_T_ptr = ptr_shared("temp_ds_T_ptr", (BLOCK_K, BLOCK_Q))
            temp_ds_T_ptr[idx_s] = ds_val

        for idx_acc in range(BLOCK_K * 128):
            row_idx = idx_acc // 128
            col_idx = idx_acc % 128
            acc_k_val = 0.0
            acc_v_val = 0.0
            for j in range(BLOCK_Q):
                acc_k_val += temp_ds_T_ptr[row_idx, j] * q_ptr_shared[j, col_idx]
                acc_v_val += temp_p_T_ptr[row_idx, j] * do_ptr_shared[j, col_idx]
            acc_dK[row_idx, col_idx] += acc_k_val
            acc_dV[row_idx, col_idx] += acc_v_val
        
        sync()

    for idx_out in range(BLOCK_K * 128):
        row_idx = idx_out // 128
        col_idx = idx_out % 128
        if k_rows[row_idx] < S_len:
            out_ptr_k = dK_ptr + ((b_h * S_len + k_rows[row_idx]) * 128 + col_idx)
            out_ptr_v = dV_ptr + ((b_h * S_len + k_rows[row_idx]) * 128 + col_idx)
            tl.store(out_ptr_k, (acc_dK[row_idx, col_idx] / sqrt_d).to(tl.bfloat16))
            tl.store(out_ptr_v, acc_dV[row_idx, col_idx].to(tl.bfloat16))


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, d = Q.shape
    assert d == 128
    sqrt_d = 1.0 / math.sqrt(d)
    
    Q_ptr = Q.flatten(0, 2)
    K_ptr = K.flatten(0, 2)
    V_ptr = V.flatten(0, 2)
    O_ptr = O.flatten(0, 2)
    dO_ptr = dO.flatten(0, 2)
    dQ_ptr = dQ.flatten(0, 2)
    dK_ptr = dK.flatten(0, 2)
    dV_ptr = dV.flatten(0, 2)
    
    grid_dq = (triton.cdiv(S_len, 128), B * H, 128, 128)
    grid_dk_dv = (triton.cdiv(S_len, 128), B * H, 128, 128)

    mha_bwd_dq_kernel[grid_dq](
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
        S_len, sqrt_d, BLOCK_M=128, BLOCK_N=128,
        num_warps=4)
    
    torch.cuda.synchronize()
    
    mha_bwd_dk_dv_kernel[grid_dk_dv](
        Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
        S_len, sqrt_d, BLOCK_K=128, BLOCK_Q=128,
        num_warps=4)