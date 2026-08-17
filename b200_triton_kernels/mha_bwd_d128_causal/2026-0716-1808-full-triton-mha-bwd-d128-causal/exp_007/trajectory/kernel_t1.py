import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
@triton.launch_metadata(maxnreg=255)
def _bwd_dk_dv(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_flat, dK, dV,
    S_len, HEAD_DIM, stride_b, stride_h, stride_s, stride_d,
    TILE, scale, pre_fetch):
    
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    B = 4
    H = 48
    
    j_start = pid_s * TILE
    bh = pid_b * H + pid_h
    
    rows = tl.arange(0, TILE)
    cols_c = tl.arange(0, 64)
    
    dK_accs = [tl.zeros((TILE, 64), tl.float32) for _ in range(2)]
    dV_accs = [tl.zeros((TILE, 64), tl.float32) for _ in range(2)]
    
    K_c = [desc_K.load([bh * S_len + j_start, c * 64]) for c in range(2)]
    V_c = [desc_V.load([bh * S_len + j_start, c * 64]) for c in range(2)]
    
    if pre_fetch and j_start // TILE < triton.cdiv(S_len, TILE) - 1:
        next_i = (j_start // TILE) + 1
        next_i_start = next_i * TILE
        next_bh_S = bh * S_len + next_i_start
        Q_c_i = [desc_Q.load([next_bh_S, c * 64]) for c in range(2)]
        dO_c_i = [desc_dO.load([next_bh_S, c * 64]) for c in range(2)]
        O_c_i = [desc_O.load([next_bh_S, c * 64]) for c in range(2)]
        
        D_i = tl.zeros((TILE,), tl.float32)
        for c in range(2):
            D_i += tl.sum(O_c_i[c] * dO_c_i[c], axis=1)
        l_idx = bh * S_len + next_i_start + rows
        L_i = tl.load(L_flat + l_idx, mask=(next_i_start + rows < S_len), other=0.0)
    else:
        Q_c_i = [None] * 2
        dO_c_i = [None] * 2
        O_c_i = [None] * 2
        D_i = None
        L_i = None
    
    for i in range(j_start // TILE, triton.cdiv(S_len, TILE)):
        i_start = i * TILE
        curr_D_i = D_i
        curr_L_i = L_i
        
        if pre_fetch and i_start // TILE < triton.cdiv(S_len, TILE) - 1:
            next_i = (i_start // TILE) + 1
            next_i_start = next_i * TILE
            next_bh_S = bh * S_len + next_i_start
            Q_c_i[0] = desc_Q.load([next_bh_S, 0])
            Q_c_i[1] = desc_Q.load([next_bh_S, 64])
            dO_c_i[0] = desc_dO.load([next_bh_S, 0])
            dO_c_i[1] = desc_dO.load([next_bh_S, 64])
            O_c_i[0] = desc_O.load([next_bh_S, 0])
            O_c_i[1] = desc_O.load([next_bh_S, 64])
            
            D_i = tl.zeros((TILE,), tl.float32)
            for c in range(2):
                D_i += tl.sum(O_c_i[c] * dO_c_i[c], axis=1)
            l_idx = bh * S_len + next_i_start + rows
            L_i = tl.load(L_flat + l_idx, mask=(next_i_start + rows < S_len), other=0.0)
        
        if curr_D_i is None:
            Q_c_i[0] = desc_Q.load([bh * S_len + i_start, 0])
            Q_c_i[1] = desc_Q.load([bh * S_len + i_start, 64])
            dO_c_i[0] = desc_dO.load([bh * S_len + i_start, 0])
            dO_c_i[1] = desc_dO.load([bh * S_len + i_start, 64])
            O_c_i[0] = desc_O.load([bh * S_len + i_start, 0])
            O_c_i[1] = desc_O.load([bh * S_len + i_start, 64])
            
            curr_D_i = tl.zeros((TILE,), tl.float32)
            for c in range(2):
                curr_D_i += tl.sum(O_c_i[c] * dO_c_i[c], axis=1)
            l_idx = bh * S_len + i_start + rows
            curr_L_i = tl.load(L_flat + l_idx, mask=(i_start + rows < S_len), other=0.0)
        
        S_mat = tl.zeros((TILE, TILE), tl.float32)
        for c in range(2):
            S_mat = tl.dot(Q_c_i[c], K_c[c].T, S_mat)
        S_mat *= scale
        
        P_mat = tl.exp(S_mat - curr_L_i[:, None])
        valid = (i_start + rows[:, None]) >= (j_start + rows[None, :])
        P_mat = tl.where(valid, P_mat, 0.0)
        
        dP_mat = tl.zeros((TILE, TILE), tl.float32)
        for c in range(2):
            dP_mat = tl.dot(dO_c_i[c], V_c[c].T, dP_mat)
        
        dS_mat = P_mat * (dP_mat - curr_D_i[:, None]) * scale
        dS_mat = tl.where(valid, dS_mat, 0.0)
        
        for c in range(2):
            dK_accs[c] = tl.dot(dS_mat.T, Q_c_i[c], dK_accs[c])
        
        for c in range(2):
            dV_accs[c] = tl.dot(P_mat.T, dO_c_i[c], dV_accs[c])
    
    for idx in range(2):
        k_ptr = dK + bh * stride_h + j_start * stride_s + idx * 64 * stride_d
        ptrs = k_ptr + rows[:, None] * stride_s + cols_c[None, :] * stride_d
        mask = (j_start + rows[:, None]) < S_len
        tl.store(ptrs, dK_accs[idx].to(tl.bfloat16), mask=mask)
        
        v_ptr = dV + bh * stride_h + j_start * stride_s + idx * 64 * stride_d
        ptrs = v_ptr + rows[:, None] * stride_s + cols_c[None, :] * stride_d
        mask = (j_start + rows[:, None]) < S_len
        tl.store(ptrs, dV_accs[idx].to(tl.bfloat16), mask=mask)


@triton.jit
@triton.launch_metadata(maxnreg=255)
def _bwd_dq(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_flat, dQ,
    S_len, HEAD_DIM, stride_b, stride_h, stride_s, stride_d,
    TILE, scale, pre_fetch):
    
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    B = 4
    H = 48
    
    i_start = pid_s * TILE
    bh = pid_b * H + pid_h
    
    rows = tl.arange(0, TILE)
    cols_c = tl.arange(0, 64)
    
    dQ_accs = [tl.zeros((TILE, 64), tl.float32) for _ in range(2)]
    
    Q_c_i = [desc_Q.load([bh * S_len + i_start, c * 64]) for c in range(2)]
    dO_c_i = [desc_dO.load([bh * S_len + i_start, c * 64]) for c in range(2)]
    O_c_i = [desc_O.load([bh * S_len + i_start, c * 64]) for c in range(2)]
    
    D_i = tl.zeros((TILE,), tl.float32)
    for c in range(2):
        D_i += tl.sum(O_c_i[c] * dO_c_i[c], axis=1)
    l_idx = bh * S_len + i_start + rows
    L_i = tl.load(L_flat + l_idx, mask=(i_start + rows < S_len), other=0.0)
    
    if pre_fetch and 0 < i_start // TILE:
        next_j = 1
        next_j_start = next_j * TILE
        next_bh_S = bh * S_len + next_j_start
        K_c_j = [desc_K.load([next_bh_S, c * 64]) for c in range(2)]
        V_c_j = [desc_V.load([next_bh_S, c * 64]) for c in range(2)]
    else:
        K_c_j = [None] * 2
        V_c_j = [None] * 2
        
    for j in range(i_start // TILE + 1):
        j_start = j * TILE
        curr_K_c_j = K_c_j
        curr_V_c_j = V_c_j
        
        if pre_fetch and j < i_start // TILE:
            next_j = j + 1
            next_j_start = next_j * TILE
            next_bh_S = bh * S_len + next_j_start
            K_c_j[0] = desc_K.load([next_bh_S, 0])
            K_c_j[1] = desc_K.load([next_bh_S, 64])
            V_c_j[0] = desc_V.load([next_bh_S, 0])
            V_c_j[1] = desc_V.load([next_bh_S, 64])
        
        if curr_K_c_j[0] is None:
            curr_K_c_j[0] = desc_K.load([bh * S_len + j_start, 0])
            curr_K_c_j[1] = desc_K.load([bh * S_len + j_start, 64])
            curr_V_c_j[0] = desc_V.load([bh * S_len + j_start, 0])
            curr_V_c_j[1] = desc_V.load([bh * S_len + j_start, 64])
            
        S_mat = tl.zeros((TILE, TILE), tl.float32)
        for c in range(2):
            S_mat = tl.dot(Q_c_i[c], curr_K_c_j[c].T, S_mat)
        S_mat *= scale
        
        P_mat = tl.exp(S_mat - L_i[:, None])
        valid = (i_start + rows[:, None]) >= (j_start + rows[None, :])
        P_mat = tl.where(valid, P_mat, 0.0)
        
        dP_mat = tl.zeros((TILE, TILE), tl.float32)
        for c in range(2):
            dP_mat = tl.dot(dO_c_i[c], curr_V_c_j[c].T, dP_mat)
        
        dS_mat = P_mat * (dP_mat - D_i[:, None]) * scale
        dS_mat = tl.where(valid, dS_mat, 0.0)
        
        for c in range(2):
            dQ_accs[c] = tl.dot(dS_mat, curr_K_c_j[c], dQ_accs[c])
            
    for idx in range(2):
        q_ptr = dQ + bh * stride_h + i_start * stride_s + idx * 64 * stride_d
        ptrs = q_ptr + rows[:, None] * stride_s + cols_c[None, :] * stride_d
        mask = (i_start + rows[:, None]) < S_len
        tl.store(ptrs, dQ_accs[idx].to(tl.bfloat16), mask=mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, HEAD_DIM = Q.shape
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    stride_b = H * S_len * HEAD_DIM
    stride_h = S_len * HEAD_DIM
    stride_s = HEAD_DIM
    stride_d = 1
    
    Q_flat = Q.view(-1, Q.shape[-1])
    K_flat = K.view(-1, K.shape[-1])
    V_flat = V.view(-1, V.shape[-1])
    O_flat = O.view(-1, O.shape[-1])
    dO_flat = dO.view(-1, dO.shape[-1])
    L_flat = L.view(-1)
    
    dQ = dQ.view(-1).contiguous()
    dK = dK.view(-1).contiguous()
    dV = dV.view(-1).contiguous()
    
    TILE = 64
    
    desc_Q = TensorDescriptor.from_tensor(Q_flat, [TILE, 64])
    desc_K = TensorDescriptor.from_tensor(K_flat, [TILE, 64])
    desc_V = TensorDescriptor.from_tensor(V_flat, [TILE, 64])
    desc_O = TensorDescriptor.from_tensor(O_flat, [TILE, 64])
    desc_dO = TensorDescriptor.from_tensor(dO_flat, [TILE, 64])
    
    grid = (triton.cdiv(S_len, TILE), H, B)
    
    _bwd_dk_dv[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, L_flat, dK, dV,
        S_len, HEAD_DIM, stride_b, stride_h, stride_s, stride_d,
        TILE, scale, True,
        num_warps=4, num_stages=2)
        
    _bwd_dq[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, L_flat, dQ,
        S_len, HEAD_DIM, stride_b, stride_h, stride_s, stride_d,
        TILE, scale, True,
        num_warps=4, num_stages=2)