import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q, K, V, O, LSE,
    S, D, H,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    b = pid_n // H
    h = pid_n % H
    
    start_n = pid_m * BLOCK_M
    
    # Base pointers for the selected (batch, head)
    q_ptr = Q + (b * H + h) * S * D
    k_ptr = K + (b * H + h) * S * D
    v_ptr = V + (b * H + h) * S * D
    o_ptr = O + (b * H + h) * S * D
    
    # Device-side tensor descriptors covering the sequence and head dimension
    q_desc = tl.make_tensor_descriptor(
        q_ptr, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_ptr, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_ptr, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr, shape=[S, D], strides=[D, 1],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    
    # Initial Q tile loaded onto-chip; reused for every J iteration
    Q_0 = q_desc.load([start_n, 0])  
    
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    
    scale = 1.0 / (D ** 0.5)
    
    num_blocks = tl.cdiv(S, BLOCK_M)
    
    for j in range(num_blocks):
        k_idx = j * BLOCK_M
        
        K_j = k_desc.load([k_idx, 0])
        V_j = v_desc.load([k_idx, 0])
        
        S_tile = tl.dot(Q_0, K_j.T) * scale 
        
        m_local = tl.max(S_tile, axis=1)
        m_new = tl.maximum(m_i, m_local)
        
        exp_old = tl.exp(m_i - m_new)
        P = tl.exp(S_tile - m_new[:, None])
        
        l_local = tl.sum(P, axis=1)
        l_i = exp_old * l_i + l_local
        
        O_acc = O_acc * exp_old[:, None] + tl.dot(P, V_j)
        
        m_i = m_new
    
    O_acc = O_acc / l_i[:, None]
    
    lse = m_i + tl.log(l_i)
    
    o_desc.store([start_n, 0], O_acc)
    
    lse_offsets = start_n + tl.arange(0, BLOCK_M)
    lse_ptr = LSE + (b * H + h) * S + lse_offsets
    mask_lse = lse_offsets < S
    tl.store(lse_ptr, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    """Compute non-causal multi-head attention with LSE output."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    block_m = 64
    grid = (triton.cdiv(S, block_m), B * H)
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        S, D, H,
        BLOCK_M=block_m,
        num_warps=4,
        num_stages=2,
    )