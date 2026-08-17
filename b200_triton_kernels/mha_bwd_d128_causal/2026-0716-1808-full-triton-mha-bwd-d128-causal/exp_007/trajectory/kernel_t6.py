import math
import torch
import triton
import triton.language as tl


@triton.jit
def load_chunk(base_ptr, rows, cols, b_off, s_off, S_len):
    ptr = base_ptr + b_off + (s_off + rows[:, None]) * 128 + cols[None, :]
    mask = (s_off + rows[:, None]) < S_len
    return tl.load(ptr, mask=mask, other=0.0)


@triton.jit
def _bwd_dk_dv(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dK_ptr, dV_ptr,
    S_len, HEAD_DIM, scale, H):
    
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    j_start = pid_s * 128
    bh = pid_b * H + pid_h
    
    b_off_Q = bh * S_len * HEAD_DIM
    b_off_K = bh * S_len * HEAD_DIM
    b_off_V = bh * S_len * HEAD_DIM
    b_off_O = bh * S_len * HEAD_DIM
    b_off_dO = bh * S_len * HEAD_DIM
    
    rows_k = tl.arange(0, 128)
    cols_c = tl.arange(0, 64)
    
    K_0 = load_chunk(K_ptr, rows_k, cols_c, b_off_K, j_start, S_len)
    K_1 = load_chunk(K_ptr, rows_k, cols_c + 64, b_off_K, j_start, S_len)
    
    V_0 = load_chunk(V_ptr, rows_k, cols_c, b_off_V, j_start, S_len)
    V_1 = load_chunk(V_ptr, rows_k, cols_c + 64, b_off_V, j_start, S_len)
    
    dK_acc = [tl.zeros((128, 64), tl.float32) for _ in range(2)]
    dV_acc = [tl.zeros((128, 64), tl.float32) for _ in range(2)]
            
    for i in range(j_start // 128, triton.cdiv(S_len, 128)):
        i_start = i * 128
        rows_q = tl.arange(0, 128)
        
        Q_0 = load_chunk(Q_ptr, rows_q, cols_c, b_off_Q, i_start, S_len)
        Q_1 = load_chunk(Q_ptr, rows_q, cols_c + 64, b_off_Q, i_start, S_len)
        
        O_0 = load_chunk(O_ptr, rows_q, cols_c, b_off_O, i_start, S_len)
        O_1 = load_chunk(O_ptr, rows_q, cols_c + 64, b_off_O, i_start, S_len)
        
        dO_0 = load_chunk(dO_ptr, rows_q, cols_c, b_off_dO, i_start, S_len)
        dO_1 = load_chunk(dO_ptr, rows_q, cols_c + 64, b_off_dO, i_start, S_len)
        
        D_i = (tl.sum(O_0 * dO_0, axis=1) + 
               tl.sum(O_1 * dO_1, axis=1))
        
        l_idx = bh * S_len + i_start + rows_q
        L_i = tl.load(L_ptr + l_idx, mask=(i_start + rows_q < S_len), other=0.0)
        
        S = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        P = tl.exp(S - L_i[:, None])
        
        valid = ((i_start + rows_q[:, None]) >= (j_start + rows_k[None, :])) & \
                ((i_start + rows_q[:, None]) < S_len) & \
                ((j_start + rows_k[None, :]) < S_len)
        P = tl.where(valid, P, 0.0)
        
        dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        dS = P * (dP - D_i[:, None]) * scale
        dS = tl.where(valid, dS, 0.0)
        
        dK_acc[0] += tl.dot(dS.T, Q_0)
        dK_acc[1] += tl.dot(dS.T, Q_1)
        
        dV_acc[0] += tl.dot(P.T, dO_0)
        dV_acc[1] += tl.dot(P.T, dO_1)
            
    for idx in range(2):
        k_ptr = dK_ptr + b_off_K + j_start * HEAD_DIM + idx * 64
        ptrs = k_ptr + rows_k[:, None] * HEAD_DIM + cols_c[None, :]
        mask = (j_start + rows_k[:, None]) < S_len
        tl.store(ptrs, dK_acc[idx].to(tl.bfloat16), mask=mask)
        
        v_ptr = dV_ptr + b_off_V + j_start * HEAD_DIM + idx * 64
        ptrs = v_ptr + rows_k[:, None] * HEAD_DIM + cols_c[None, :]
        mask = (j_start + rows_k[:, None]) < S_len
        tl.store(ptrs, dV_acc[idx].to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd_dq(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr, dQ_ptr,
    S_len, HEAD_DIM, scale, H):
    
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    i_start = pid_s * 128
    bh = pid_b * H + pid_h
    
    b_off = bh * S_len * HEAD_DIM
    
    rows_q = tl.arange(0, 128)
    cols_c = tl.arange(0, 64)
    
    Q_0 = load_chunk(Q_ptr, rows_q, cols_c, b_off, i_start, S_len)
    Q_1 = load_chunk(Q_ptr, rows_q, cols_c + 64, b_off, i_start, S_len)
    
    dO_0 = load_chunk(dO_ptr, rows_q, cols_c, b_off, i_start, S_len)
    dO_1 = load_chunk(dO_ptr, rows_q, cols_c + 64, b_off, i_start, S_len)
    
    O_0 = load_chunk(O_ptr, rows_q, cols_c, b_off, i_start, S_len)
    O_1 = load_chunk(O_ptr, rows_q, cols_c + 64, b_off, i_start, S_len)
    
    D_i = (tl.sum(O_0 * dO_0, axis=1) + 
           tl.sum(O_1 * dO_1, axis=1))
    
    l_idx = bh * S_len + i_start + rows_q
    L_i = tl.load(L_ptr + l_idx, mask=(i_start + rows_q < S_len), other=0.0)
    
    dQ_acc = [tl.zeros((128, 64), tl.float32) for _ in range(2)]
            
    for j in range(i_start // 128 + 1):
        j_start = j * 128
        rows_k = tl.arange(0, 128)
        
        K_0 = load_chunk(K_ptr, rows_k, cols_c, b_off, j_start, S_len)
        K_1 = load_chunk(K_ptr, rows_k, cols_c + 64, b_off, j_start, S_len)
        
        V_0 = load_chunk(V_ptr, rows_k, cols_c, b_off, j_start, S_len)
        V_1 = load_chunk(V_ptr, rows_k, cols_c + 64, b_off, j_start, S_len)
        
        S = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        P = tl.exp(S - L_i[:, None])
        
        valid = ((i_start + rows_q[:, None]) >= (j_start + rows_k[None, :])) & \
                ((i_start + rows_q[:, None]) < S_len) & \
                ((j_start + rows_k[None, :]) < S_len)
        P = tl.where(valid, P, 0.0)
        
        dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        dS = P * (dP - D_i[:, None]) * scale
        dS = tl.where(valid, dS, 0.0)
        
        dQ_acc[0] += tl.dot(dS, K_0)
        dQ_acc[1] += tl.dot(dS, K_1)
            
    for idx in range(2):
        q_ptr = dQ_ptr + b_off + i_start * HEAD_DIM + idx * 64
        ptrs = q_ptr + rows_q[:, None] * HEAD_DIM + cols_c[None, :]
        mask = (i_start + rows_q[:, None]) < S_len
        tl.store(ptrs, dQ_acc[idx].to(tl.bfloat16), mask=mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, HEAD_DIM = Q.shape
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    grid = (triton.cdiv(S_len, 128), H, B)
    
    _bwd_dk_dv[grid](
        Q, K, V, O, dO, L, dK, dV,
        S_len, HEAD_DIM, scale, H,
        num_warps=8, num_stages=3)
        
    _bwd_dq[grid](
        Q, K, V, O, dO, L, dQ,
        S_len, HEAD_DIM, scale, H,
        num_warps=8, num_stages=3)