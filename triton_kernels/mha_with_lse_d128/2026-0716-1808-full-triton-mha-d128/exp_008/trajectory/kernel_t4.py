import torch
import triton
import triton.language as tl


@triton.jit
def load_4d_slice(base_ptr, row_ptr, col_offs, row_len: tl.constexpr, col_len: tl.constexpr, S, D):
    row_offs = tl.arange(0, row_len)
    col_offs = tl.arange(0, col_len)
    ptrs = base_ptr + (row_ptr + row_offs[:, None]) * D + col_offs[None, :]
    start_row = row_ptr % S
    mask = (start_row + row_offs < S)[:, None]
    return tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def store_4d_slice(base_ptr, row_ptr, col_offs, acc, row_len: tl.constexpr, col_len: tl.constexpr, S, D):
    row_offs = tl.arange(0, row_len)
    col_offs = tl.arange(0, col_len)
    ptrs = base_ptr + (row_ptr + row_offs[:, None]) * D + col_offs[None, :]
    start_row = row_ptr % S
    mask = (start_row + row_offs < S)[:, None]
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
    
    q0_0 = load_4d_slice(Q_ptr, q_row_ptr + 0, 0, 32, 64, S, HEAD_DIM)
    q0_1 = load_4d_slice(Q_ptr, q_row_ptr + 0, 64, 32, 64, S, HEAD_DIM)
    q1_0 = load_4d_slice(Q_ptr, q_row_ptr + 32, 0, 32, 64, S, HEAD_DIM)
    q1_1 = load_4d_slice(Q_ptr, q_row_ptr + 32, 64, 32, 64, S, HEAD_DIM)
    q2_0 = load_4d_slice(Q_ptr, q_row_ptr + 64, 0, 32, 64, S, HEAD_DIM)
    q2_1 = load_4d_slice(Q_ptr, q_row_ptr + 64, 64, 32, 64, S, HEAD_DIM)
    q3_0 = load_4d_slice(Q_ptr, q_row_ptr + 96, 0, 32, 64, S, HEAD_DIM)
    q3_1 = load_4d_slice(Q_ptr, q_row_ptr + 96, 64, 32, 64, S, HEAD_DIM)
    
    qi = [q0_0, q0_1, q1_0, q1_1, q2_0, q2_1, q3_0, q3_1]
    
    o_acc = [
        [tl.zeros((32, 64), dtype=tl.float32), tl.zeros((32, 64), dtype=tl.float32)] 
        for _ in range(4)
    ]
    
    m_row = [tl.full((32,), -float("inf"), dtype=tl.float32) for _ in range(4)]
    ell_row = [tl.zeros((32,), dtype=tl.float32) for _ in range(4)]
    
    row_idx = tl.arange(0, 32)
    row_mask = [
        (q_start + 0 + row_idx < S).to(tl.float32),
        (q_start + 32 + row_idx < S).to(tl.float32),
        (q_start + 64 + row_idx < S).to(tl.float32),
        (q_start + 96 + row_idx < S).to(tl.float32),
    ]
    
    num_kv_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    for j_blk in range(num_kv_blocks):
        kv_start = j_blk * BLOCK_N
        kv_row_ptr = b_h * S + kv_start
        
        k0_0 = load_4d_slice(K_ptr, kv_row_ptr + 0, 0, 32, 64, S, HEAD_DIM)
        k0_1 = load_4d_slice(K_ptr, kv_row_ptr + 0, 64, 32, 64, S, HEAD_DIM)
        k1_0 = load_4d_slice(K_ptr, kv_row_ptr + 32, 0, 32, 64, S, HEAD_DIM)
        k1_1 = load_4d_slice(K_ptr, kv_row_ptr + 32, 64, 32, 64, S, HEAD_DIM)
        k2_0 = load_4d_slice(K_ptr, kv_row_ptr + 64, 0, 32, 64, S, HEAD_DIM)
        k2_1 = load_4d_slice(K_ptr, kv_row_ptr + 64, 64, 32, 64, S, HEAD_DIM)
        k3_0 = load_4d_slice(K_ptr, kv_row_ptr + 96, 0, 32, 64, S, HEAD_DIM)
        k3_1 = load_4d_slice(K_ptr, kv_row_ptr + 96, 64, 32, 64, S, HEAD_DIM)
        
        kj = [
            [k0_0, k0_1],
            [k1_0, k1_1],
            [k2_0, k2_1],
            [k3_0, k3_1]
        ]
        
        v0_0 = load_4d_slice(V_ptr, kv_row_ptr + 0, 0, 32, 64, S, HEAD_DIM)
        v0_1 = load_4d_slice(V_ptr, kv_row_ptr + 0, 64, 32, 64, S, HEAD_DIM)
        v1_0 = load_4d_slice(V_ptr, kv_row_ptr + 32, 0, 32, 64, S, HEAD_DIM)
        v1_1 = load_4d_slice(V_ptr, kv_row_ptr + 32, 64, 32, 64, S, HEAD_DIM)
        v2_0 = load_4d_slice(V_ptr, kv_row_ptr + 64, 0, 32, 64, S, HEAD_DIM)
        v2_1 = load_4d_slice(V_ptr, kv_row_ptr + 64, 64, 32, 64, S, HEAD_DIM)
        v3_0 = load_4d_slice(V_ptr, kv_row_ptr + 96, 0, 32, 64, S, HEAD_DIM)
        v3_1 = load_4d_slice(V_ptr, kv_row_ptr + 96, 64, 32, 64, S, HEAD_DIM)
        
        vj = [
            [v0_0, v0_1],
            [v1_0, v1_1],
            [v2_0, v2_1],
            [v3_0, v3_1]
        ]
        
        s_tiles = [[None]*4 for _ in range(4)]
        
        col_idx = tl.arange(0, 64)
        col_mask = [
            (kv_start + 0 + col_idx < S).to(tl.float32),
            (kv_start + 32 + col_idx < S).to(tl.float32),
            (kv_start + 64 + col_idx < S).to(tl.float32),
            (kv_start + 96 + col_idx < S).to(tl.float32),
        ]
        col_mask_expanded = [m[None, :] for m in col_mask]
        
        for i in range(4):
            for j in range(4):
                s_ij = tl.dot(qi[i][0], kj[j][0].T)
                s_ij = tl.dot(qi[i][1], kj[j][1].T, s_ij)
                s_ij = s_ij * scale
                
                s_ij = s_ij * col_mask_expanded[j] + (-float("inf")) * (1 - col_mask_expanded[j])
                s_tiles[i][j] = s_ij
        
        for i in range(4):
            local_m_i = tl.max(s_tiles[i][0], axis=1)
            for j in range(1, 4):
                local_m_i = tl.maximum(local_m_i, tl.max(s_tiles[i][j], axis=1))
            
            local_m_i = local_m_i * row_mask[i] + (-float("inf")) * (1 - row_mask[i])
            
            old_m_i = m_row[i]
            new_m_i = tl.maximum(old_m_i, local_m_i)
            alpha_i = tl.exp(old_m_i - new_m_i)
            
            o_acc[i][0] = o_acc[i][0] * alpha_i[:, None]
            o_acc[i][1] = o_acc[i][1] * alpha_i[:, None]
            
            p_ij_sum = 0.0
            for j in range(4):
                p_ij = tl.exp(s_tiles[i][j] - new_m_i[:, None])
                
                row_mask_expanded = [m[:, None] for m in row_mask]
                p_ij_flat = p_ij * row_mask_expanded[i] * col_mask_expanded[j]
                
                p_ij_bf16 = p_ij_flat.to(tl.bfloat16)
                
                o_acc[i][0] = tl.dot(p_ij_bf16, vj[j][0], o_acc[i][0])
                o_acc[i][1] = tl.dot(p_ij_bf16, vj[j][1], o_acc[i][1])
                
                p_ij_sum += tl.sum(p_ij_flat, axis=1)
            
            ell_row[i] = ell_row[i] * alpha_i + p_ij_sum
            m_row[i] = new_m_i
            
    for i in range(4):
        inv_ell_i = 1.0 / ell_row[i]
        
        out0 = (o_acc[i][0] * inv_ell_i[:, None]).to(tl.bfloat16)
        out1 = (o_acc[i][1] * inv_ell_i[:, None]).to(tl.bfloat16)
        
        store_4d_slice(O_ptr, q_row_ptr + i * 32, 0, out0, 32, 64, S, HEAD_DIM)
        store_4d_slice(O_ptr, q_row_ptr + i * 32, 64, out1, 32, 64, S, HEAD_DIM)
    
    for i in range(4):
        lse_ptr = LSE_ptr + b_h * S + q_start + i * 32 + row_idx
        lse_val = m_row[i] + tl.log(ell_row[i])
        tl.store(lse_ptr, lse_val, mask=q_start + i * 32 + row_idx < S)


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
        num_warps=4, num_stages=2
    )