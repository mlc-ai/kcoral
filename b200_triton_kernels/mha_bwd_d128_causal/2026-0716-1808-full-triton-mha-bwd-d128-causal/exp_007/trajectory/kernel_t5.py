import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


@triton.jit
def _bwd_dq_tma(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_flat, dQ,
    S_len, HEAD_DIM, stride_h, stride_s, stride_d, scale, H):
    
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    i_start = pid_s * 128
    bh = pid_b * H + pid_h
    
    rows_q = tl.arange(0, 128)
    cols_c = tl.arange(0, 64)
    
    dQ_acc = [tl.zeros((128, 64), tl.float32) for _ in range(2)]
    
    Q_0 = desc_Q.load([bh, i_start, 0])
    Q_1 = desc_Q.load([bh, i_start, 64])
    dO_0 = desc_dO.load([bh, i_start, 0])
    dO_1 = desc_dO.load([bh, i_start, 64])
    O_0 = desc_O.load([bh, i_start, 0])
    O_1 = desc_O.load([bh, i_start, 64])
    
    D_i = tl.sum(O_0 * dO_0, axis=1) + tl.sum(O_1 * dO_1, axis=1)
    
    l_idx = bh * S_len + i_start + rows_q
    L_i = tl.load(L_flat + l_idx, mask=(i_start + rows_q < S_len), other=0.0)
    
    K_0 = desc_K.load([bh, 0, 0])
    K_1 = desc_K.load([bh, 0, 64])
    V_0 = desc_V.load([bh, 0, 0])
    V_1 = desc_V.load([bh, 0, 64])
    
    for j in range(i_start // 128 + 1):
        j_start = j * 128
        
        if j < i_start // 128:
            next_j = j + 1
            next_j_start = next_j * 128
            K_0 = desc_K.load([bh, next_j_start, 0])
            K_1 = desc_K.load([bh, next_j_start, 64])
            V_0 = desc_V.load([bh, next_j_start, 0])
            V_1 = desc_V.load([bh, next_j_start, 64])
        
        rows_k = tl.arange(0, 128)
        
        S = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        P = tl.exp(S - L_i[:, None])
        
        key_valid = (j_start + rows_k[None, :]) < S_len
        valid = ((i_start + rows_q[:, None]) >= (j_start + rows_k[None, :])) & key_valid
        P = tl.where(valid, P, 0.0)
        
        dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        dS = P * (dP - D_i[:, None]) * scale
        dS = tl.where(valid, dS, 0.0)
        
        dQ_acc[0] += tl.dot(dS, K_0)
        dQ_acc[1] += tl.dot(dS, K_1)
            
    for idx in range(2):
        q_ptr = dQ + bh * stride_h + i_start * stride_s + idx * 64 * stride_d
        ptrs = q_ptr + rows_q[:, None] * stride_s + cols_c[None, :] * stride_d
        mask = ((i_start + rows_q[:, None]) < S_len) & (cols_c[None, :] < HEAD_DIM)
        tl.store(ptrs, dQ_acc[idx].to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd_dk_dv(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_flat, dK, dV,
    S_len, HEAD_DIM, stride_h, stride_s, stride_d, scale, H):
    
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    j_start = pid_s * 128
    bh = pid_b * H + pid_h
    
    rows_k = tl.arange(0, 128)
    cols_c = tl.arange(0, 64)
    
    dK_acc = [tl.zeros((128, 64), tl.float32) for _ in range(2)]
    dV_acc = [tl.zeros((128, 64), tl.float32) for _ in range(2)]
    
    K_0 = desc_K.load([bh, j_start, 0])
    K_1 = desc_K.load([bh, j_start, 64])
    V_0 = desc_V.load([bh, j_start, 0])
    V_1 = desc_V.load([bh, j_start, 64])
            
    for i in range(j_start // 128, triton.cdiv(S_len, 128)):
        i_start = i * 128
        rows_q = tl.arange(0, 128)
        
        Q_0 = desc_Q.load([bh, i_start, 0])
        Q_1 = desc_Q.load([bh, i_start, 64])
        O_0 = desc_O.load([bh, i_start, 0])
        O_1 = desc_O.load([bh, i_start, 64])
        dO_0 = desc_dO.load([bh, i_start, 0])
        dO_1 = desc_dO.load([bh, i_start, 64])
        
        D_i = tl.sum(O_0 * dO_0, axis=1) + tl.sum(O_1 * dO_1, axis=1)
        
        l_idx = bh * S_len + i_start + rows_q
        L_i = tl.load(L_flat + l_idx, mask=(i_start + rows_q < S_len), other=0.0)
        
        S = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        P = tl.exp(S - L_i[:, None])
        
        key_valid = (j_start + rows_k[None, :]) < S_len
        valid = ((i_start + rows_q[:, None]) >= (j_start + rows_k[None, :])) & key_valid
        P = tl.where(valid, P, 0.0)
        
        dP = tl.dot(dO_0, V_0.T) + tl.dot(dO_1, V_1.T)
        
        dS = P * (dP - D_i[:, None]) * scale
        dS = tl.where(valid, dS, 0.0)
        
        dK_acc[0] += tl.dot(dS.T, Q_0)
        dK_acc[1] += tl.dot(dS.T, Q_1)
        
        dV_acc[0] += tl.dot(P.T, dO_0)
        dV_acc[1] += tl.dot(P.T, dO_1)
            
    for idx in range(2):
        k_ptr = dK + bh * stride_h + j_start * stride_s + idx * 64 * stride_d
        ptrs = k_ptr + rows_k[:, None] * stride_s + cols_c[None, :] * stride_d
        mask = ((j_start + rows_k[:, None]) < S_len) & (cols_c[None, :] < HEAD_DIM)
        tl.store(ptrs, dK_acc[idx].to(tl.bfloat16), mask=mask)
        
        v_ptr = dV + bh * stride_h + j_start * stride_s + idx * 64 * stride_d
        ptrs = v_ptr + rows_k[:, None] * stride_s + cols_c[None, :] * stride_d
        mask = ((j_start + rows_k[:, None]) < S_len) & (cols_c[None, :] < HEAD_DIM)
        tl.store(ptrs, dV_acc[idx].to(tl.bfloat16), mask=mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, HEAD_DIM = Q.shape
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    stride_h = S_len * HEAD_DIM
    stride_s = HEAD_DIM
    stride_d = 1
    
    Q_flat = Q.view(B * H, S_len, HEAD_DIM).contiguous()
    K_flat = K.view(B * H, S_len, HEAD_DIM).contiguous()
    V_flat = V.view(B * H, S_len, HEAD_DIM).contiguous()
    O_flat = O.view(B * H, S_len, HEAD_DIM).contiguous()
    dO_flat = dO.view(B * H, S_len, HEAD_DIM).contiguous()
    
    desc_Q = TensorDescriptor.from_tensor(Q_flat, [1, 128, 64])
    desc_K = TensorDescriptor.from_tensor(K_flat, [1, 128, 64])
    desc_V = TensorDescriptor.from_tensor(V_flat, [1, 128, 64])
    desc_O = TensorDescriptor.from_tensor(O_flat, [1, 128, 64])
    desc_dO = TensorDescriptor.from_tensor(dO_flat, [1, 128, 64])
    
    grid = (triton.cdiv(S_len, 128), H, B)
    
    _bwd_dq_tma[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, L.view(-1), dQ,
        S_len, HEAD_DIM, stride_h, stride_s, stride_d, scale, H,
        num_warps=4, num_stages=3)
        
    _bwd_dk_dv[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, L.view(-1), dK, dV,
        S_len, HEAD_DIM, stride_h, stride_s, stride_d, scale, H,
        num_warps=8, num_stages=2)