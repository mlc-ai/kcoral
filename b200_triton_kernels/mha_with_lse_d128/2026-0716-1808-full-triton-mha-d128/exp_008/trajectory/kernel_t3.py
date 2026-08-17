import torch
import triton
import triton.language as tl


@triton.jit
def load_2d_slice(base_ptr, row_ptr, col_offs_start, row_len: tl.constexpr, col_len: tl.constexpr, S, D):
    row_offs = tl.arange(0, row_len)
    col_offs = tl.arange(0, col_len)
    ptrs = base_ptr + (row_ptr + row_offs[:, None]) * D + (col_offs_start + col_offs[None, :])
    start_row = row_ptr % S
    mask = (start_row + row_offs < S)[:, None] & (col_offs_start + col_offs < D)[None, :]
    return tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def store_2d_slice(base_ptr, row_ptr, col_offs_start, acc, row_len: tl.constexpr, col_len: tl.constexpr, S, D):
    row_offs = tl.arange(0, row_len)
    col_offs = tl.arange(0, col_len)
    ptrs = base_ptr + (row_ptr + row_offs[:, None]) * D + (col_offs_start + col_offs[None, :])
    start_row = row_ptr % S
    mask = (start_row + row_offs < S)[:, None] & (col_offs_start + col_offs < D)[None, :]
    tl.store(ptrs, acc, mask=mask)


@triton.jit
def _attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    S, scale, H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    q_start = pid * BLOCK_M
    b_h = tl.program_id(1)
    
    q_row_ptr = b_h * S + q_start
    
    q0 = load_2d_slice(Q_ptr, q_row_ptr, 0, BLOCK_M, 64, S, HEAD_DIM)
    q1 = load_2d_slice(Q_ptr, q_row_ptr, 64, BLOCK_M, 64, S, HEAD_DIM)
    
    o_acc0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    o_acc1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    m_row = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    ell_row = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    row_idx = tl.arange(0, BLOCK_M)
    row_mask = q_start + row_idx < S
    
    num_kv_blocks = (S + BLOCK_N - 1) // BLOCK_N
    
    for j in range(num_kv_blocks):
        kv_start = j * BLOCK_N
        kv_row_ptr = b_h * S + kv_start
        
        k0 = load_2d_slice(K_ptr, kv_row_ptr, 0, BLOCK_N, 64, S, HEAD_DIM)
        k1 = load_2d_slice(K_ptr, kv_row_ptr, 64, BLOCK_N, 64, S, HEAD_DIM)
        
        acc_o = tl.dot(q0, k0.T)
        acc_o = tl.dot(q1, k1.T, acc_o)
        
        acc_o = acc_o * scale
        
        col_idx = tl.arange(0, BLOCK_N)
        col_mask = kv_start + col_idx < S
        acc_o = tl.where(col_mask[None, :], acc_o, -float("inf"))
        
        local_m = tl.max(acc_o, axis=1)
        local_m = tl.where(row_mask, local_m, -float("inf"))
        
        old_m = m_row
        new_m = tl.maximum(old_m, local_m)
        
        alpha = tl.exp(old_m - new_m)
        
        o_acc0 = o_acc0 * alpha[:, None]
        o_acc1 = o_acc1 * alpha[:, None]
        
        p_cur = tl.exp(acc_o - new_m[:, None])
        p_cur = tl.where(row_mask[:, None] & col_mask[None, :], p_cur, 0.0)
        
        ell_row = ell_row * alpha + tl.sum(p_cur, axis=1)
        ell_row = tl.where(row_mask, ell_row, 0.0)
        
        p_cur_reshaped = tl.reshape(p_cur, (BLOCK_M, 2, 64), can_reorder=False)
        p_cur_transposed = p_cur_reshaped.T
        p0, p1 = tl.split(p_cur_transposed)
        
        p0 = p0.to(tl.bfloat16)
        p1 = p1.to(tl.bfloat16)
        
        v00 = load_2d_slice(V_ptr, kv_row_ptr, 0, 64, 64, S, HEAD_DIM)
        v01 = load_2d_slice(V_ptr, kv_row_ptr, 64, 64, 64, S, HEAD_DIM)
        v10 = load_2d_slice(V_ptr, kv_row_ptr + 64, 0, 64, 64, S, HEAD_DIM)
        v11 = load_2d_slice(V_ptr, kv_row_ptr + 64, 64, 64, 64, S, HEAD_DIM)
        
        o_acc0 = tl.dot(p0, v00, o_acc0)
        o_acc0 = tl.dot(p1, v10, o_acc0)
        o_acc1 = tl.dot(p0, v01, o_acc1)
        o_acc1 = tl.dot(p1, v11, o_acc1)
        
        m_row = tl.where(row_mask, new_m, -float("inf"))
        
    inv_ell = 1.0 / ell_row
    out0 = (o_acc0 * inv_ell[:, None]).to(tl.bfloat16)
    out1 = (o_acc1 * inv_ell[:, None]).to(tl.bfloat16)
    
    store_2d_slice(O_ptr, q_row_ptr, 0, out0, BLOCK_M, 64, S, HEAD_DIM)
    store_2d_slice(O_ptr, q_row_ptr, 64, out1, BLOCK_M, 64, S, HEAD_DIM)
    
    lse_ptr = LSE_ptr + (b_h // H) * (H * S) + (b_h % H) * S + q_start + row_idx
    lse_val = m_row + tl.log(ell_row)
    tl.store(lse_ptr, lse_val, mask=q_start + row_idx < S)


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
        num_warps=8, num_stages=3
    )