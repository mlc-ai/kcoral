import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    S_len,
    H,
    D,
    B,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(2)
    h = tl.program_id(1)
    i = tl.program_id(0)
    
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    
    q_idx = i * BLOCK_M + rows
    
    Q_desc = tl.make_tensor_descriptor(
        Q, 
        shape=[B * H * S_len, 128], 
        strides=[128, 1], 
        block_shape=[BLOCK_M, 64], 
        padding_option="zero"
    )
    
    K_desc = tl.make_tensor_descriptor(
        K, 
        shape=[B * H * S_len, 128], 
        strides=[128, 1], 
        block_shape=[BLOCK_N, 64], 
        padding_option="zero"
    )
    
    V_desc = tl.make_tensor_descriptor(
        V, 
        shape=[B * H * S_len, 128], 
        strides=[128, 1], 
        block_shape=[BLOCK_N, 64], 
        padding_option="zero"
    )
    
    base_Q = b * H * S_len + h * S_len + i * BLOCK_M
    base_K = b * H * S_len + h * S_len
    
    Q_0 = Q_desc.load([base_Q, 0])
    Q_1 = Q_desc.load([base_Q, 64])
    
    m = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    l = tl.full((BLOCK_M,), 0.0, tl.float32)
    O_acc_0 = tl.zeros((BLOCK_M, 64), tl.float32)
    O_acc_1 = tl.zeros((BLOCK_M, 64), tl.float32)
    
    for j in range(i + 1):
        offset_K = base_K + j * BLOCK_N
        
        K_0 = K_desc.load([offset_K, 0])
        K_1 = K_desc.load([offset_K, 64])
        V_0 = V_desc.load([offset_K, 0])
        V_1 = V_desc.load([offset_K, 64])
        
        S = (tl.dot(Q_0, K_0.T) + tl.dot(Q_1, K_1.T)) * scale
        
        is_causal = q_idx[:, None] >= (j * BLOCK_N + cols[None, :])
        S = tl.where(is_causal, S, -float('inf'))
        
        m_old = m
        m_local = tl.max(S, axis=1)
        m = tl.maximum(m_old, m_local)
        P = tl.exp(S - m[:, None])
        P = tl.where(m > -float('inf'), P, 0.0)
        
        l_local = tl.sum(P, axis=1)
        l = l * tl.exp(m_old - m) + l_local
        
        O_acc_0 = O_acc_0 * tl.exp(m_old - m)[:, None] + tl.dot(P, V_0)
        O_acc_1 = O_acc_1 * tl.exp(m_old - m)[:, None] + tl.dot(P, V_1)
    
    l_safe = tl.where(l > 0, l, 1.0)
    O_0 = (O_acc_0 / l_safe[:, None]).to(tl.bfloat16)
    O_1 = (O_acc_1 / l_safe[:, None]).to(tl.bfloat16)
    
    d_indices_0 = tl.arange(0, 64)
    d_indices_1 = tl.arange(0, 64) + 64
    
    base_offset_O = b * H * S_len * 128 + h * S_len * 128
    out_ptrs_0 = O + base_offset_O + q_idx[:, None] * 128 + d_indices_0[None, :]
    out_ptrs_1 = O + base_offset_O + q_idx[:, None] * 128 + d_indices_1[None, :]
    
    tl.store(out_ptrs_0, O_0, mask=(q_idx[:, None] < S_len))
    tl.store(out_ptrs_1, O_1, mask=(q_idx[:, None] < S_len))
    
    lse = m + tl.log(l_safe)
    
    lse_base = b * H * S_len + h * S_len
    lse_ptrs = LSE + lse_base + q_idx
    tl.store(lse_ptrs, lse, mask=(q_idx < S_len))


def run(Q, K, V, O, LSE):
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    torch.cuda.set_device(Q.device)
    B, H, S_len, D = Q.shape
    
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (triton.cdiv(S_len, BLOCK_M), H, B)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S_len, H, D, B,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )