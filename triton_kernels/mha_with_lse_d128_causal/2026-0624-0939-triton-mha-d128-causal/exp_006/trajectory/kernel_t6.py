import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    q_desc, k_desc, v_desc, o_desc, lse_ptr,
    s_dim, H,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    bh = b * H + h
    
    q_0 = tl.reshape(q_desc.load([bh, i * BLOCK_M, 0]), [BLOCK_M, 64], can_reorder=True).to(tl.float32)
    q_1 = tl.reshape(q_desc.load([bh, i * BLOCK_M, 64]), [BLOCK_M, 64], can_reorder=True).to(tl.float32)
    
    acc_o_0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_o_1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    m_i = tl.full((BLOCK_M,), -1e20, tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    for j in range(0, i + 1):
        k_0 = tl.reshape(k_desc.load([bh, j * BLOCK_N, 0]), [BLOCK_N, 64], can_reorder=True).to(tl.float32)
        k_1 = tl.reshape(k_desc.load([bh, j * BLOCK_N, 64]), [BLOCK_N, 64], can_reorder=True).to(tl.float32)
        
        v_0 = tl.reshape(v_desc.load([bh, j * BLOCK_N, 0]), [BLOCK_N, 64], can_reorder=True).to(tl.float32)
        v_1 = tl.reshape(v_desc.load([bh, j * BLOCK_N, 64]), [BLOCK_N, 64], can_reorder=True).to(tl.float32)
        
        s = tl.dot(q_0, k_0.T)
        s = tl.dot(q_1, k_1.T, s)
        s = s * scale
        
        q_idx = i * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        k_idx = j * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        
        mask = (q_idx >= k_idx) & (q_idx < s_dim) & (k_idx < s_dim)
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
    
    o_0 = acc_o_0.to(tl.bfloat16)
    o_1 = acc_o_1.to(tl.bfloat16)
    
    o_desc.store([bh, i * BLOCK_M, 0], o_0)
    o_desc.store([bh, i * BLOCK_M, 64], o_1)
    
    q_idx = i * BLOCK_M + tl.arange(0, BLOCK_M)
    lse = m_i + tl.log(l_i)
    lse_base = lse_ptr + (b * H + h) * s_dim + i * BLOCK_M
    tl.store(lse_base + tl.arange(0, BLOCK_M), lse, mask=(q_idx < s_dim))


def run(q, k, v, o, lse):
    """Compute causal multi-head attention with LSE output."""
    torch.cuda.set_device(q.device)
    B = q.shape[0]
    H = q.shape[1]
    s_dim = q.shape[2]
    d_dim = q.shape[3]
    
    scale = 1.0 / (d_dim ** 0.5)
    
    grid = (triton.cdiv(s_dim, 64), H, B)
    
    q_flat = q.view(B * H, s_dim, d_dim)
    k_flat = k.view(B * H, s_dim, d_dim)
    v_flat = v.view(B * H, s_dim, d_dim)
    o_flat = o.view(B * H, s_dim, d_dim)
    
    q_desc = TensorDescriptor.from_tensor(q_flat, [1, 64, 64])
    k_desc = TensorDescriptor.from_tensor(k_flat, [1, 64, 64])
    v_desc = TensorDescriptor.from_tensor(v_flat, [1, 64, 64])
    o_desc = TensorDescriptor.from_tensor(o_flat, [1, 64, 64])
    
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, lse,
        s_dim, H,
        scale,
        BLOCK_M=64,
        BLOCK_N=64,
        num_warps=4,
        num_stages=3,
    )