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
    
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 64)
    
    q_offs_row = (i * 64 + rows) * d_dim
    q_ptrs_0 = q_ptr + (b*H + h)*s_dim*d_dim + q_offs_row[:, None] + cols[None, :]
    q_ptrs_1 = q_ptr + (b*H + h)*s_dim*d_dim + q_offs_row[:, None] + cols[None, :] + 64
    q_mask = (i * 64 + rows[:, None]) < s_dim
    
    q_0 = tl.load(q_ptrs_0, mask=q_mask, other=0.0)
    q_1 = tl.load(q_ptrs_1, mask=q_mask, other=0.0)
    
    acc_o_0 = tl.zeros((64, 64), tl.float32)
    acc_o_1 = tl.zeros((64, 64), tl.float32)
    
    m_i = tl.full((64,), -1e20, tl.float32)
    l_i = tl.full((64,), 0.0, tl.float32)
    
    for j in range(0, i + 1):
        k_offs_row = (j * 64 + rows) * d_dim
        k_ptrs_0 = k_ptr + (b*H + h)*s_dim*d_dim + k_offs_row[:, None] + cols[None, :]
        k_ptrs_1 = k_ptr + (b*H + h)*s_dim*d_dim + k_offs_row[:, None] + cols[None, :] + 64
        k_mask = (j * 64 + rows[:, None]) < s_dim
        
        k_0 = tl.load(k_ptrs_0, mask=k_mask, other=0.0)
        k_1 = tl.load(k_ptrs_1, mask=k_mask, other=0.0)
        
        v_ptrs_0 = v_ptr + (b*H + h)*s_dim*d_dim + k_offs_row[:, None] + cols[None, :]
        v_ptrs_1 = v_ptr + (b*H + h)*s_dim*d_dim + k_offs_row[:, None] + cols[None, :] + 64
        
        v_0 = tl.load(v_ptrs_0, mask=k_mask, other=0.0)
        v_1 = tl.load(v_ptrs_1, mask=k_mask, other=0.0)
        
        s = tl.zeros((64, 64), tl.float32)
        s = tl.dot(q_0, k_0.T, s)
        s = tl.dot(q_1, k_1.T, s)
        s = s * scale
        
        q_idx = i * 64 + rows[:, None]
        k_idx = j * 64 + cols[None, :]
        
        if i == j:
            mask = (q_idx >= k_idx) & (q_idx < s_dim) & (k_idx < s_dim)
        else:
            mask = (q_idx < s_dim) & (k_idx < s_dim)
        
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
    
    o_ptrs_0 = o_ptr + (b*H + h)*s_dim*d_dim + q_offs_row[:, None] + cols[None, :]
    o_ptrs_1 = o_ptr + (b*H + h)*s_dim*d_dim + q_offs_row[:, None] + cols[None, :] + 64
    
    mask_out = (i * 64 + rows[:, None]) < s_dim
    
    o_0 = acc_o_0.to(tl.bfloat16)
    o_1 = acc_o_1.to(tl.bfloat16)
    
    tl.store(o_ptrs_0, o_0, mask=mask_out)
    tl.store(o_ptrs_1, o_1, mask=mask_out)
    
    q_idx = i * 64 + rows
    lse = m_i + tl.log(l_i)
    lse_base = lse_ptr + (b * H + h) * s_dim + i * 64
    tl.store(lse_base + rows, lse, mask=(q_idx < s_dim))


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