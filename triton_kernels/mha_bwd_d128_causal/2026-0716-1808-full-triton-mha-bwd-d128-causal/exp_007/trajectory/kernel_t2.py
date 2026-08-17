import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


@triton.jit
def load_tile_chunk(base_ptr, rows, cols, b_off, s_off, stride_s, stride_d, S_len):
    ptr = base_ptr + b_off + s_off * stride_s
    tile_ptr = ptr + rows[:, None] * stride_s + cols[None, :] * stride_d
    mask = (s_off + rows[:, None]) < S_len
    return tl.load(tile_ptr, mask=mask, other=0.0)


@triton.jit
def compute_D(base_O, base_dO, rows, cols_c, b_off, s_off, stride_s, stride_d, S_len):
    O_c = load_tile_chunk(base_O, rows, cols_c, b_off, s_off, stride_s, stride_d, S_len)
    dO_c = load_tile_chunk(base_dO, rows, cols_c, b_off, s_off, stride_s, stride_d, S_len)
    return tl.sum(O_c * dO_c, axis=1)


@triton.jit
def _bwd_dk_dv(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_flat, dK, dV,
    S_len, HEAD_DIM, stride_h, stride_s, stride_d, scale):
    
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    B = 4
    H = 48
    
    j_start = pid_s * 128
    bh = pid_b * H + pid_h
    b_off = bh * stride_h
    
    rows_k = tl.arange(0, 128)
    cols_c = tl.arange(0, 64)
    
    dK_acc = [tl.zeros((128, 64), tl.float32) for _ in range(2)]
    dV_acc = [tl.zeros((128, 64), tl.float32) for _ in range(2)]
    
    K_c = [desc_K.load([bh * S_len + j_start, c * 64]) for c in range(2)]
    V_c = [desc_V.load([bh * S_len + j_start, c * 64]) for c in range(2)]
    
    for i in range(j_start // 128, triton.cdiv(S_len, 128)):
        i_start = i * 128
        rows_q = tl.arange(0, 128)
        
        Q_c = [desc_Q.load([bh * S_len + i_start, c * 64]) for c in range(2)]
        O_c = [desc_O.load([bh * S_len + i_start, c * 64]) for c in range(2)]
        dO_c = [desc_dO.load([bh * S_len + i_start, c * 64]) for c in range(2)]
        
        D_i = sum(tl.sum(O_c[c] * dO_c[c], axis=1) for c in range(2))
        
        l_idx = bh * S_len + i_start + rows_q
        L_i = tl.load(L_flat + l_idx, mask=(i_start + rows_q < S_len), other=0.0)
        
        S = sum(tl.dot(Q_c[c], K_c[c].T) for c in range(2))
        S *= scale
        
        P = tl.exp(S - L_i[:, None])
        
        key_valid = (j_start + rows_k[None, :]) < S_len
        valid = ((i_start + rows_q[:, None]) >= (j_start + rows_k[None, :])) & key_valid
        P = tl.where(valid, P, 0.0)
        
        dP = sum(tl.dot(dO_c[c], V_c[c].T) for c in range(2))
        
        dS = P * (dP - D_i[:, None]) * scale
        dS = tl.where(valid, dS, 0.0)
        
        for c in range(2):
            dK_acc[c] += tl.dot(dS.T, Q_c[c])
        
        for c in range(2):
            dV_acc[c] += tl.dot(P.T, dO_c[c])
            
    for idx in range(2):
        k_ptr = dK + b_off + j_start * stride_s + idx * 64 * stride_d
        ptrs = k_ptr + rows_k[:, None] * stride_s + cols_c[None, :] * stride_d
        mask = (j_start + rows_k[:, None]) < S_len
        tl.store(ptrs, dK_acc[idx].to(tl.bfloat16), mask=mask)
        
        v_ptr = dV + b_off + j_start * stride_s + idx * 64 * stride_d
        ptrs = v_ptr + rows_k[:, None] * stride_s + cols_c[None, :] * stride_d
        mask = (j_start + rows_k[:, None]) < S_len
        tl.store(ptrs, dV_acc[idx].to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd_dq(
    desc_Q, desc_K, desc_V, desc_O, desc_dO, L_flat, dQ,
    S_len, HEAD_DIM, stride_h, stride_s, stride_d, scale):
    
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    B = 4
    H = 48
    
    i_start = pid_s * 128
    bh = pid_b * H + pid_h
    b_off = bh * stride_h
    
    rows_q = tl.arange(0, 128)
    cols_c = tl.arange(0, 64)
    
    dQ_acc = [tl.zeros((128, 64), tl.float32) for _ in range(2)]
    
    Q_c = [desc_Q.load([bh * S_len + i_start, c * 64]) for c in range(2)]
    dO_c = [desc_dO.load([bh * S_len + i_start, c * 64]) for c in range(2)]
    O_c = [desc_O.load([bh * S_len + i_start, c * 64]) for c in range(2)]
    
    D_i = sum(tl.sum(O_c[c] * dO_c[c], axis=1) for c in range(2))
    
    l_idx = bh * S_len + i_start + rows_q
    L_i = tl.load(L_flat + l_idx, mask=(i_start + rows_q < S_len), other=0.0)
    
    for j in range(i_start // 128 + 1):
        j_start = j * 128
        rows_k = tl.arange(0, 128)
        
        K_c = [desc_K.load([bh * S_len + j_start, c * 64]) for c in range(2)]
        V_c = [desc_V.load([bh * S_len + j_start, c * 64]) for c in range(2)]
        
        S = sum(tl.dot(Q_c[c], K_c[c].T) for c in range(2))
        S *= scale
        
        P = tl.exp(S - L_i[:, None])
        
        key_valid = (j_start + rows_k[None, :]) < S_len
        valid = ((i_start + rows_q[:, None]) >= (j_start + rows_k[None, :])) & key_valid
        P = tl.where(valid, P, 0.0)
        
        dP = sum(tl.dot(dO_c[c], V_c[c].T) for c in range(2))
        
        dS = P * (dP - D_i[:, None]) * scale
        dS = tl.where(valid, dS, 0.0)
        
        for c in range(2):
            dQ_acc[c] += tl.dot(dS, K_c[c])
            
    for idx in range(2):
        q_ptr = dQ + b_off + i_start * stride_s + idx * 64 * stride_d
        ptrs = q_ptr + rows_q[:, None] * stride_s + cols_c[None, :] * stride_d
        mask = (i_start + rows_q[:, None]) < S_len
        tl.store(ptrs, dQ_acc[idx].to(tl.bfloat16), mask=mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, HEAD_DIM = Q.shape
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    stride_b = H * S_len * HEAD_DIM
    stride_h = S_len * HEAD_DIM
    stride_s = HEAD_DIM
    stride_d = 1
    
    Q_flat = Q.view(B * H, S_len, HEAD_DIM).contiguous()
    K_flat = K.view(B * H, S_len, HEAD_DIM).contiguous()
    V_flat = V.view(B * H, S_len, HEAD_DIM).contiguous()
    O_flat = O.view(B * H, S_len, HEAD_DIM).contiguous()
    dO_flat = dO.view(B * H, S_len, HEAD_DIM).contiguous()
    L_flat = L.view(-1)
    
    desc_Q = TensorDescriptor.from_tensor(Q_flat, [128, 128])
    desc_K = TensorDescriptor.from_tensor(K_flat, [128, 128])
    desc_V = TensorDescriptor.from_tensor(V_flat, [128, 128])
    desc_O = TensorDescriptor.from_tensor(O_flat, [128, 128])
    desc_dO = TensorDescriptor.from_tensor(dO_flat, [128, 128])
    
    grid = (triton.cdiv(S_len, 128), H, B)
    
    _bwd_dk_dv[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, L_flat, dK, dV,
        S_len, HEAD_DIM, stride_h, stride_s, stride_d, scale,
        num_warps=8, num_stages=3)
        
    _bwd_dq[grid](
        desc_Q, desc_K, desc_V, desc_O, desc_dO, L_flat, dQ,
        S_len, HEAD_DIM, stride_h, stride_s, stride_d, scale,
        num_warps=8, num_stages=3)


if __name__ == "__main__":
    B, H, S, d = 4, 48, 64, 128
    Q = torch.randn(B, H, S, d, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, d, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, d, dtype=torch.bfloat16, device="cuda")
    O = torch.randn(B, H, S, d, dtype=torch.bfloat16, device="cuda")
    dO = torch.randn(B, H, S, d, dtype=torch.bfloat16, device="cuda")
    L = torch.randn(B, H, S, dtype=torch.float32, device="cuda")
    
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    
    run(Q, K, V, O, dO, L, dQ, dK, dV)
    print("Done!")