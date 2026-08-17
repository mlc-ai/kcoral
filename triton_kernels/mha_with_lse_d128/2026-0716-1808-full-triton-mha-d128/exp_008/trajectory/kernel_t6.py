import torch
import triton
import triton.language as tl


@triton.jit
def load_3d_slice(base_ptr, row_ptr, col_offs_start, row_len, col_len, total_rows, D):
    row_offs = tl.arange(0, row_len)
    col_offs = tl.arange(0, col_len)
    ptrs = base_ptr + (row_ptr + row_offs[:, None]) * D + (col_offs_start + col_offs[None, :])
    mask = (row_ptr + row_offs < total_rows)[:, None]
    return tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def store_3d_slice(base_ptr, row_ptr, col_offs_start, acc, row_len, col_len, total_rows, D):
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
    bh = tl.program_id(1)
    
    q_row_ptr = bh * S + q_start
    total_rows = bh * S + S
    
    o_acc0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    o_acc1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    m_row = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    ell_row = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    row_idx = tl.arange(0, BLOCK_M)
    row_mask = (q_start + row_idx < S).to(tl.float32)
    
    num_kv_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    for j_blk in range(num_kv_blocks):
        kv_start = j_blk * BLOCK_N
        kv_row_ptr = bh * S + kv_start
        
        acc_o = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        num_k_tiles = tl.cdiv(HEAD_DIM, 64)
        
        for k_tile_idx in range(num_k_tiles):
            k_offset = k_tile_idx * 64
            
            q_part = load_3d_slice(Q_ptr, q_row_ptr, k_offset, BLOCK_M, 64, total_rows, HEAD_DIM)
            k_part = load_3d_slice(K_ptr, kv_row_ptr, k_offset, BLOCK_N, 64, total_rows, HEAD_DIM)
            
            acc_o = tl.dot(q_part, k_part.T, acc_o)
        
        acc_o = acc_o * scale
        
        col_idx = tl.arange(0, BLOCK_N)
        col_mask = (kv_start + col_idx < S)[None, :]
        acc_o = tl.where(col_mask, acc_o, -float("inf"))
        
        local_m = tl.max(acc_o, axis=1)
        local_m = local_m * row_mask + (-float("inf")) * (1 - row_mask)
        
        old_m = m_row
        new_m = tl.maximum(old_m, local_m)
        
        alpha = tl.exp(old_m - new_m)
        alpha = alpha * row_mask
        
        o_acc0 = o_acc0 * alpha[:, None]
        o_acc1 = o_acc1 * alpha[:, None]
        
        p_cur = tl.exp(acc_o - new_m[:, None])
        p_cur = p_cur * row_mask[:, None]
        
        ell_row = ell_row * alpha + tl.sum(p_cur, axis=1)
        
        p_cur_bf16 = p_cur.to(tl.bfloat16)
        
        v0 = load_3d_slice(V_ptr, kv_row_ptr, 0, BLOCK_N, 64, total_rows, HEAD_DIM)
        v1 = load_3d_slice(V_ptr, kv_row_ptr, 64, BLOCK_N, 64, total_rows, HEAD_DIM)
        
        o_acc0 = tl.dot(p_cur_bf16, v0, o_acc0)
        o_acc1 = tl.dot(p_cur_bf16, v1, o_acc1)
        
        m_row = new_m
        
    inv_ell = 1.0 / (ell_row + 1e-20)
    
    out0 = (o_acc0 * inv_ell[:, None]).to(tl.bfloat16)
    out1 = (o_acc1 * inv_ell[:, None]).to(tl.bfloat16)
    
    store_3d_slice(O_ptr, q_row_ptr, 0, out0, BLOCK_M, 64, total_rows, HEAD_DIM)
    store_3d_slice(O_ptr, q_row_ptr, 64, out1, BLOCK_M, 64, total_rows, HEAD_DIM)
    
    row_idx_32 = tl.arange(0, BLOCK_M)
    lse_ptr = LSE_ptr + bh * S + q_start + row_idx_32
    lse_val = m_row + tl.log(ell_row + 1e-20)
    
    lse_val = tl.where(row_mask > 0, lse_val, 0.0)
    
    tl.store(lse_ptr, lse_val, mask=q_start + row_idx_32 < S)


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