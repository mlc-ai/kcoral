import math
import torch
import triton
import triton.language as tl


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
    D = tl.zeros((32,), tl.float32)
    for chunk in range(2):
        O_c = load_tile_chunk(base_O, rows, chunk * 64 + cols_c, b_off, s_off, stride_s, stride_d, S_len)
        dO_c = load_tile_chunk(base_dO, rows, chunk * 64 + cols_c, b_off, s_off, stride_s, stride_d, S_len)
        D += tl.sum(O_c * dO_c, axis=1)
    return D


@triton.jit
@triton.launch_metadata(maxnreg=255)
def _bwd_dq_dv_opt(
    Q, K, V, O, dO, L, dQ, dV_tmp,
    S_len, HEAD_DIM, stride_b, stride_h, stride_s, stride_d, stride_b_L, stride_h_L,
    TILE, scale, causal, CHUNK):
    
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    s_start = pid_s * TILE
    b_off = pid_b * stride_b + pid_h * stride_h
    
    rows = tl.arange(0, TILE)
    cols_d = tl.arange(0, HEAD_DIM)
    cols_c = tl.arange(0, CHUNK)
    
    D_i = compute_D(O, dO, rows, cols_c, b_off, s_start, stride_s, stride_d, S_len)
    L_i = tl.load(L + (pid_b * stride_b_L + pid_h * stride_h_L + s_start) + rows, 
                   mask=(s_start + rows < S_len), other=0.0)
    
    dQ_accs = [tl.zeros((TILE, CHUNK), tl.float32) for _ in range(2)]
    
    for j in range(s_start // TILE + 1):
        j_start = j * TILE
        
        S_mat = tl.zeros((TILE, TILE), tl.float32)
        for chunk in range(2):
            Q_c = load_tile_chunk(Q, rows, chunk * 64 + cols_c, b_off, s_start, stride_s, stride_d, S_len)
            K_c = load_tile_chunk(K, rows, chunk * 64 + cols_c, b_off, j_start, stride_s, stride_d, S_len)
            S_mat = tl.dot(Q_c, K_c.T, S_mat)
        S_mat *= scale
        
        P_mat = tl.exp(S_mat - L_i[:, None])
        valid = (s_start + rows[:, None]) >= (j_start + rows[None, :])
        P_mat = tl.where(valid, P_mat, 0.0)
        
        dP_mat = tl.zeros((TILE, TILE), tl.float32)
        for chunk in range(2):
            dO_c = load_tile_chunk(dO, rows, chunk * 64 + cols_c, b_off, s_start, stride_s, stride_d, S_len)
            V_c = load_tile_chunk(V, rows, chunk * 64 + cols_c, b_off, j_start, stride_s, stride_d, S_len)
            dP_mat = tl.dot(dO_c, V_c.T, dP_mat)
        
        dS_mat = P_mat * (dP_mat - D_i[:, None]) * scale
        dS_mat = tl.where(valid, dS_mat, 0.0)
        
        for chunk in range(2):
            K_c = load_tile_chunk(K, rows, chunk * 64 + cols_c, b_off, j_start, stride_s, stride_d, S_len)
            dQ_accs[chunk] = tl.dot(dS_mat, K_c, dQ_accs[chunk])
        
        dV_curr = [tl.zeros((TILE, CHUNK), tl.float32) for _ in range(2)]
        for chunk in range(2):
            dO_c = load_tile_chunk(dO, rows, chunk * 64 + cols_c, b_off, s_start, stride_s, stride_d, S_len)
            dV_curr[chunk] = tl.dot(P_mat.T, dO_c, dV_curr[chunk])
        
        for idx in range(2):
            base_ptr = dV_tmp + b_off + j_start * stride_s
            c = idx * 64 + cols_c
            ptrs = base_ptr + rows[:, None] * stride_s + c[None, :] * stride_d
            mask = ((j_start + rows[:, None]) < S_len)
            
            flat_ptrs = tl.ravel(ptrs)
            flat_vals = tl.ravel(dV_curr[idx])
            flat_mask = tl.ravel(mask)
            
            tl.atomic_add(flat_ptrs, flat_vals, mask=flat_mask)
    
    for idx in range(2):
        base_ptr = dQ + b_off + s_start * stride_s
        c = idx * 64 + cols_c
        ptrs = base_ptr + rows[:, None] * stride_s + c[None, :] * stride_d
        mask = ((s_start + rows[:, None]) < S_len)
        tl.store(ptrs, dQ_accs[idx].to(tl.bfloat16), mask=mask)


@triton.jit
@triton.launch_metadata(maxnreg=255)
def _bwd_dk_opt(
    Q, K, V, O, dO, L, dK,
    S_len, HEAD_DIM, stride_b, stride_h, stride_s, stride_d, stride_b_L, stride_h_L,
    TILE, scale, causal, CHUNK):
    
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    j_start = pid_s * TILE
    b_off = pid_b * stride_b + pid_h * stride_h
    
    rows = tl.arange(0, TILE)
    cols_d = tl.arange(0, HEAD_DIM)
    cols_c = tl.arange(0, CHUNK)
    
    dK_accs = [tl.zeros((TILE, CHUNK), tl.float32) for _ in range(2)]
    
    for i in range(j_start // TILE, triton.cdiv(S_len, TILE)):
        i_start = i * TILE
        
        D_i = compute_D(O, dO, rows, cols_c, b_off, i_start, stride_s, stride_d, S_len)
        L_i = tl.load(L + (pid_b * stride_b_L + pid_h * stride_h_L + i_start) + rows,
                       mask=(i_start + rows < S_len), other=0.0)
        
        S_mat = tl.zeros((TILE, TILE), tl.float32)
        for chunk in range(2):
            Q_c = load_tile_chunk(Q, rows, chunk * 64 + cols_c, b_off, i_start, stride_s, stride_d, S_len)
            K_c = load_tile_chunk(K, rows, chunk * 64 + cols_c, b_off, j_start, stride_s, stride_d, S_len)
            S_mat = tl.dot(Q_c, K_c.T, S_mat)
        S_mat *= scale
        
        P_mat = tl.exp(S_mat - L_i[:, None])
        valid = (i_start + rows[:, None]) >= (j_start + rows[None, :])
        P_mat = tl.where(valid, P_mat, 0.0)
        
        dP_mat = tl.zeros((TILE, TILE), tl.float32)
        for chunk in range(2):
            dO_c = load_tile_chunk(dO, rows, chunk * 64 + cols_c, b_off, i_start, stride_s, stride_d, S_len)
            V_c = load_tile_chunk(V, rows, chunk * 64 + cols_c, b_off, j_start, stride_s, stride_d, S_len)
            dP_mat = tl.dot(dO_c, V_c.T, dP_mat)
        
        dS_mat = P_mat * (dP_mat - D_i[:, None]) * scale
        dS_mat = tl.where(valid, dS_mat, 0.0)
        
        for chunk in range(2):
            Q_c = load_tile_chunk(Q, rows, chunk * 64 + cols_c, b_off, i_start, stride_s, stride_d, S_len)
            dK_accs[chunk] = tl.dot(dS_mat.T, Q_c, dK_accs[chunk])
    
    for idx in range(2):
        base_ptr = dK + b_off + j_start * stride_s
        c = idx * 64 + cols_c
        ptrs = base_ptr + rows[:, None] * stride_s + c[None, :] * stride_d
        mask = ((j_start + rows[:, None]) < S_len)
        tl.store(ptrs, dK_accs[idx].to(tl.bfloat16), mask=mask)


@triton.jit
def _cast_kernel(src, dst, n_elements):
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    mask = offsets < n_elements
    vals = tl.load(src + offsets, mask=mask, other=0.0)
    tl.store(dst + offsets, vals.to(tl.bfloat16), mask=mask)


def run(Q, K, V, O, dO, L, dQ, dK, dV):
    torch.cuda.set_device(Q.device)
    B, H, S_len, HEAD_DIM = Q.shape
    
    scale = 1.0 / math.sqrt(HEAD_DIM)
    
    stride_b = H * S_len * HEAD_DIM
    stride_h = S_len * HEAD_DIM
    stride_s = HEAD_DIM
    stride_d = 1
    
    stride_b_L = H * S_len
    stride_h_L = S_len
    
    raw_buf = alloc_fn(B * H * S_len * HEAD_DIM * 4, 16, torch.cuda.current_stream())
    dv_tmp = raw_buf.view(torch.float32)
    
    grid = (triton.cdiv(S_len, 32), H, B)
    
    _bwd_dq_dv_opt[grid](
        Q, K, V, O, dO, L, dQ, dv_tmp,
        S_len, HEAD_DIM, stride_b, stride_h, stride_s, stride_d, stride_b_L, stride_h_L,
        32, scale, True, 64)
    
    _bwd_dk_opt[grid](
        Q, K, V, O, dO, L, dK,
        S_len, HEAD_DIM, stride_b, stride_h, stride_s, stride_d, stride_b_L, stride_h_L,
        32, scale, True, 64)
    
    n_elements = B * H * S_len * HEAD_DIM
    cast_grid = (triton.cdiv(n_elements, 256),)
    _cast_kernel[cast_grid](dv_tmp, dV, n_elements)


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