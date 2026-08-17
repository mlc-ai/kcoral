import torch
import triton
import triton.language as tl


@triton.jit
def load_4d_slice(base_ptr, row_ptr, col_offs_start, row_len: tl.constexpr, col_len: tl.constexpr, total_rows, D):
    row_offs = tl.arange(0, row_len)
    col_offs = tl.arange(0, col_len)
    ptrs = base_ptr + (row_ptr + row_offs[:, None]) * D + (col_offs_start + col_offs[None, :])
    mask = (row_ptr + row_offs < total_rows)[:, None]
    return tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def store_4d_slice(base_ptr, row_ptr, col_offs_start, acc, row_len: tl.constexpr, col_len: tl.constexpr, total_rows, D):
    row_offs = tl.arange(0, row_len)
    col_offs = tl.arange(0, col_len)
    ptrs = base_ptr + (row_ptr + row_offs[:, None]) * D + (col_offs_start + col_offs[None, :])
    mask = (row_ptr + row_offs < total_rows)[:, None]
    tl.store(ptrs, acc, mask=mask)


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, scale, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    q_start = pid * BLOCK_M
    b_h = tl.program_id(1)
    
    q_row_ptr = b_h * S + q_start
    total_rows = b_h * S + S
    
    qi = []
    for i in range(4):
        q0 = load_4d_slice(Q_ptr, q_row_ptr + i * 32, 0, 32, 64, total_rows, HEAD_DIM)
        q1 = load_4d_slice(Q_ptr, q_row_ptr + i * 32, 64, 32, 64, total_rows, HEAD_DIM)
        qi.append([q0, q1])
    
    o_acc = [tl.zeros((32, 64), dtype=tl.float32) for _ in range(8)]
    
    m_row = tl.full((128,), -float("inf"), dtype=tl.float32)
    ell_row = tl.zeros((128,), dtype=tl.float32)
    
    row_idx = tl.arange(0, 128)
    row_mask = (q_start + row_idx < S).to(tl.float32)
    
    num_kv_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    for j_blk in range(num_kv_blocks):
        kv_start = j_blk * BLOCK_N
        kv_row_ptr = b_h * S + kv_start
        
        kj = []
        for j in range(4):
            k0 = load_4d_slice(K_ptr, kv_row_ptr + j * 32, 0, 32, 64, total_rows, HEAD_DIM)
            k1 = load_4d_slice(K_ptr, kv_row_ptr + j * 32, 64, 32, 64, total_rows, HEAD_DIM)
            kj.append([k0, k1])
        
        vj = []
        for j in range(4):
            v0 = load_4d_slice(V_ptr, kv_row_ptr + j * 32, 0, 32, 64, total_rows, HEAD_DIM)
            v1 = load_4d_slice(V_ptr, kv_row_ptr + j * 32, 64, 32, 64, total_rows, HEAD_DIM)
            vj.append([v0, v1])
        
        s_tiles = [[None]*4 for _ in range(4)]
        
        col_idx = tl.arange(0, 32)
        
        for i in range(4):
            for j in range(4):
                s_ij = tl.dot(qi[i][0], kj[j][0].T)
                s_ij = tl.dot(qi[i][1], kj[j][1].T, s_ij)
                s_ij = s_ij * scale
                
                valid = ((kv_start + j * 32 + col_idx) < S)[None, :]
                s_ij = tl.where(valid, s_ij, -float("inf"))
                
                s_tiles[i][j] = s_ij
        
        local_m = [None] * 4
        for i in range(4):
            local_m[i] = tl.max(s_tiles[i][0], axis=1)
            for j in range(1, 4):
                local_m[i] = tl.maximum(local_m[i], tl.max(s_tiles[i][j], axis=1))
        
        local_m_flat = local_m[0]
        for i in range(1, 4):
            local_m_flat = tl.cat([local_m_flat, local_m[i]], dim=0)
            
        old_m = m_row
        new_m = tl.maximum(old_m, local_m_flat)
        
        valid_old_m = old_m > -1e38
        alpha = tl.where(valid_old_m, tl.exp(old_m - new_m), 0.0)
        
        for i in range(4):
            a_i = alpha[i*32:(i+1)*32]
            valid_a = a_i[:, None]
            o_acc[2*i]   = tl.where(valid_a, o_acc[2*i] * a_i[:, None], 0.0)
            o_acc[2*i+1] = tl.where(valid_a, o_acc[2*i+1] * a_i[:, None], 0.0)
            
        for i in range(4):
            n_m_i = new_m[i*32:(i+1)*32]
            a_i = alpha[i*32:(i+1)*32]
            
            p_sum_i = 0.0
            
            for j in range(4):
                p_ij = tl.exp(s_tiles[i][j] - n_m_i[:, None])
                
                r_mask = row_mask[i*32:(i+1)*32, None]
                c_mask = (kv_start + j*32 + col_idx < S)[None, :]
                p_ij = p_ij * r_mask * c_mask
                
                p_sum_i += tl.sum(p_ij, axis=1)
                
                p_ij_bf16 = p_ij.to(tl.bfloat16)
                
                o_acc[2*i]   = tl.dot(p_ij_bf16, vj[j][0], o_acc[2*i])
                o_acc[2*i+1] = tl.dot(p_ij_bf16, vj[j][1], o_acc[2*i+1])
            
            e_i = ell_row[i*32:(i+1)*32]
            ell_row[i*32:(i+1)*32] = e_i * a_i + p_sum_i
            
        m_row = new_m
        
    for i in range(4):
        e_i = ell_row[i*32:(i+1)*32]
        inv_e_i = 1.0 / e_i
        
        out0 = (o_acc[2*i] * inv_e_i[:, None]).to(tl.bfloat16)
        out1 = (o_acc[2*i+1] * inv_e_i[:, None]).to(tl.bfloat16)
        
        store_4d_slice(O_ptr, q_row_ptr + i * 32, 0, out0, 32, 64, total_rows, HEAD_DIM)
        store_4d_slice(O_ptr, q_row_ptr + i * 32, 64, out1, 32, 64, total_rows, HEAD_DIM)
    
    row_idx_32 = tl.arange(0, 32)
    for i in range(4):
        lse_ptr = LSE_ptr + b_h * S + q_start + i * 32 + row_idx_32
        new_m_i = m_row[i*32:(i+1)*32]
        ell_i = ell_row[i*32:(i+1)*32]
        lse_val = new_m_i + tl.log(ell_i)
        
        lse_val = tl.where(new_m_i > -1e38, lse_val, 0.0)
        
        tl.store(lse_ptr, lse_val, mask=q_start + i * 32 + row_idx_32 < S)


BLOCK_M = 128
BLOCK_N = 128

def run(Q, K, V, O, LSE):
    """Compute Multi-Head Attention O and LSE into preallocated CUDA tensors."""
    B, H, S, D = Q.shape
    device = Q.device
    scale = 1.0 / (D ** 0.5)
    
    num_blocks = triton.cdiv(S, BLOCK_M)
    grid = (num_blocks, B * H)
    
    print(f"Triton MHA Config: Grid={grid}, S={S}, Scale={scale}")
    
    _attention_kernel[grid](
        Q.view(B * H, S, D), K.view(B * H, S, D), V.view(B * H, S, D),
        O.view(B * H, S, D), LSE,
        S, scale, H,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        num_warps=8, num_stages=2
    )