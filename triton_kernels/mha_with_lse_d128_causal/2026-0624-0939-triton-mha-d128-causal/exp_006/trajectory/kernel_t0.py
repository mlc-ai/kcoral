import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    s_dim, d_dim, H,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    
    acc_o_0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_o_1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    m_i = tl.full((BLOCK_M,), -1e20, tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    q_base = q_ptr + (b * H + h) * s_dim * d_dim + i * d_dim
    
    for j in range(0, i + 1):
        k_base = k_ptr + (b * H + h) * s_dim * d_dim + j * d_dim
        v_base = v_ptr + (b * H + h) * s_dim * d_dim + j * d_dim
        
        q_ptrs_0 = q_base + rows[:, None] * d_dim + cols[None, :]
        q_0 = tl.load(q_ptrs_0, mask=(i * BLOCK_M + rows[:, None]) < s_dim, other=0.0)
        q_ptrs_1 = q_base + rows[:, None] * d_dim + (cols[None, :] + 64)
        q_1 = tl.load(q_ptrs_1, mask=(i * BLOCK_M + rows[:, None]) < s_dim, other=0.0)
        
        k_ptrs_0 = k_base + rows[:, None] * d_dim + cols[None, :]
        k_0 = tl.load(k_ptrs_0, mask=(j * BLOCK_N + rows[:, None]) < s_dim, other=0.0)
        k_ptrs_1 = k_base + rows[:, None] * d_dim + (cols[None, :] + 64)
        k_1 = tl.load(k_ptrs_1, mask=(j * BLOCK_N + rows[:, None]) < s_dim, other=0.0)
        
        v_ptrs_0 = v_base + rows[:, None] * d_dim + cols[None, :]
        v_0 = tl.load(v_ptrs_0, mask=(j * BLOCK_N + rows[:, None]) < s_dim, other=0.0)
        v_ptrs_1 = v_base + rows[:, None] * d_dim + (cols[None, :] + 64)
        v_1 = tl.load(v_ptrs_1, mask=(j * BLOCK_N + rows[:, None]) < s_dim, other=0.0)
        
        s = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        s = tl.dot(q_0, k_0.T, s)
        s = tl.dot(q_1, k_1.T, s)
        s = s * scale
        
        q_idx = i * BLOCK_M + rows
        k_idx = j * BLOCK_N + rows
        mask = (q_idx[:, None] >= k_idx[None, :]) & (q_idx[:, None] < s_dim) & (k_idx[None, :] < s_dim)
        s = tl.where(mask, s, -1e20)
        
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        l_new_part = tl.sum(p, axis=1)
        
        exp_diff = tl.exp(m_i - m_new)
        l_i = l_i * exp_diff + l_new_part
        m_i = m_new
        
        acc_o_0 = acc_o_0 * exp_diff[:, None]
        acc_o_1 = acc_o_1 * exp_diff[:, None]
        
        acc_o_0 = tl.dot(p, v_0, acc_o_0)
        acc_o_1 = tl.dot(p, v_1, acc_o_1)
    
    inv_l = 1.0 / l_i
    acc_o_0 = acc_o_0 * inv_l[:, None]
    acc_o_1 = acc_o_1 * inv_l[:, None]
    
    o_base = o_ptr + (b * H + h) * s_dim * d_dim + i * d_dim
    o_ptrs_0 = o_base + rows[:, None] * d_dim + cols[None, :]
    o_ptrs_1 = o_base + rows[:, None] * d_dim + (cols[None, :] + 64)
    
    q_idx = i * BLOCK_M + rows
    mask_out = q_idx[:, None] < s_dim
    tl.store(o_ptrs_0, acc_o_0, mask=mask_out)
    tl.store(o_ptrs_1, acc_o_1, mask=mask_out)
    
    lse = m_i + tl.log(l_i)
    lse_base = lse_ptr + (b * H + h) * s_dim + i * BLOCK_M
    lse_ptrs = lse_base + rows
    tl.store(lse_ptrs, lse, mask=(q_idx < s_dim))


def run(q, k, v, o, lse):
    """Compute causal multi-head attention with LSE output."""
    torch.cuda.set_device(q.device)
    B = q.shape[0]
    H = q.shape[1]
    s_dim = q.shape[2]
    d_dim = q.shape[3]
    
    scale = 1.0 / (d_dim ** 0.5)
    
    grid = (triton.cdiv(s_dim, 64), H, B)
    _attention_kernel[grid](
        q, k, v, o, lse,
        s_dim, d_dim, H,
        scale,
        BLOCK_M=64,
        BLOCK_N=64,
        num_warps=8,
        num_stages=3,
    )