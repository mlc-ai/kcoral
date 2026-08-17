import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    s_dim, H,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    
    q_desc = tl.make_tensor_descriptor(
        q_ptr + (b*H + h)*s_dim*128,
        shape=[s_dim, 128],
        strides=[128, 1],
        block_shape=[BLOCK_M, 64],
        padding_option="zero"
    )
    
    k_desc = tl.make_tensor_descriptor(
        k_ptr + (b*H + h)*s_dim*128,
        shape=[s_dim, 128],
        strides=[128, 1],
        block_shape=[BLOCK_N, 64],
        padding_option="zero"
    )
    
    v_desc = tl.make_tensor_descriptor(
        v_ptr + (b*H + h)*s_dim*128,
        shape=[s_dim, 128],
        strides=[128, 1],
        block_shape=[BLOCK_N, 64],
        padding_option="zero"
    )
    
    o_desc = tl.make_tensor_descriptor(
        o_ptr + (b*H + h)*s_dim*128,
        shape=[s_dim, 128],
        strides=[128, 1],
        block_shape=[BLOCK_M, 64],
        padding_option="zero"
    )
    
    q_0 = q_desc.load([i * BLOCK_M, 0]).to(tl.float32)
    q_1 = q_desc.load([i * BLOCK_M, 64]).to(tl.float32)
    
    acc_o_0 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    acc_o_1 = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    
    m_i = tl.full((BLOCK_M,), -1e20, tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    
    for j in range(0, i + 1):
        k_0 = k_desc.load([j * BLOCK_N, 0]).to(tl.float32)
        k_1 = k_desc.load([j * BLOCK_N, 64]).to(tl.float32)
        
        v_0 = v_desc.load([j * BLOCK_N, 0]).to(tl.float32)
        v_1 = v_desc.load([j * BLOCK_N, 64]).to(tl.float32)
        
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
    
    o_desc.store([i * BLOCK_M, 0], acc_o_0.to(tl.bfloat16))
    o_desc.store([i * BLOCK_M, 64], acc_o_1.to(tl.bfloat16))
    
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
    
    grid = (triton.cdiv(s_dim, 128), H, B)
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
        
    triton.set_allocator(alloc_fn)
    
    _attention_kernel[grid](
        q, k, v, o, lse,
        s_dim, H,
        scale,
        BLOCK_M=128,
        BLOCK_N=64,
        num_warps=8,
        num_stages=2,
    )