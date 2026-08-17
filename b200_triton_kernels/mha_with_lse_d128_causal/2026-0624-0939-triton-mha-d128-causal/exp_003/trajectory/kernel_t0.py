import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    S, D,
    q_strides_0, q_strides_1, q_strides_2,
    k_strides_0, k_strides_1, k_strides_2,
    v_strides_0, v_strides_1, v_strides_2,
    o_strides_0, o_strides_1, o_strides_2,
    lse_strides_0, lse_strides_1, lse_strides_2,
    scale,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes causal attention using a Flash Attention style loop.
    Processes a block of queries and loops over all key/value blocks.
    Accumulates Output (O) and LogSumExp (LSE) in FP32.
    """
    b = tl.program_id(1)
    h = tl.program_id(2)
    n = tl.program_id(0) * BLOCK_N
    
    rows_n = tl.arange(0, BLOCK_N)
    cols_d = tl.arange(0, 64)
    
    base_offset = b * q_strides_0 + h * q_strides_1
    
    q_s = n + rows_n[:, None]
    Q0_ptr = Q + base_offset + q_s * q_strides_2 + cols_d
    Q1_ptr = Q + base_offset + q_s * q_strides_2 + 64 + cols_d
    
    q_mask = (n + rows_n)[:, None] < S
    Q0 = tl.load(Q0_ptr, mask=q_mask, other=0.0)
    Q1 = tl.load(Q1_ptr, mask=q_mask, other=0.0)
    
    F0 = tl.zeros((BLOCK_N, 64), tl.float32)
    F1 = tl.zeros((BLOCK_N, 64), tl.float32)
    m = tl.full((BLOCK_N,), -float('inf'), tl.float32)
    D_denom = tl.zeros((BLOCK_N,), tl.float32)
    
    num_k_iters = tl.cdiv(S, BLOCK_K)
    
    for j in range(num_k_iters):
        k_s = j * BLOCK_K + rows_n[:, None]
        K0_ptr = K + base_offset + k_s * k_strides_2 + cols_d
        K1_ptr = K + base_offset + k_s * k_strides_2 + 64 + cols_d
        V0_ptr = V + base_offset + k_s * v_strides_2 + cols_d
        V1_ptr = V + base_offset + k_s * v_strides_2 + 64 + cols_d
        
        k_mask = (j * BLOCK_K + rows_n)[:, None] < S
        K0 = tl.load(K0_ptr, mask=k_mask, other=0.0)
        K1 = tl.load(K1_ptr, mask=k_mask, other=0.0)
        V0 = tl.load(V0_ptr, mask=k_mask, other=0.0)
        V1 = tl.load(V1_ptr, mask=k_mask, other=0.0)
        
        S_attn = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        S_attn = S_attn * scale
        
        q_pos = (n + rows_n)[:, None]
        k_pos = (j * BLOCK_K + rows_n)[None, :]
        causal_mask = (k_pos <= q_pos) & (q_pos < S)
        
        S_attn = tl.where(causal_mask, S_attn, -float('inf'))
        
        m_local = tl.max(S_attn, axis=1, keep_dims=True)
        m_new = tl.maximum(m[:, None], m_local)
        m_new_squeezed = m_new[:, 0]
        
        P = tl.exp(S_attn - m_new)
        P = tl.where(causal_mask, P, 0.0)
        
        D_part = tl.sum(P, axis=1, keep_dims=True)
        D_new = D_denom[:, None] * tl.exp(m[:, None] - m_new) + D_part
        D_new_squeezed = D_new[:, 0]
        
        exp_diff = tl.exp(m[:, None] - m_new)
        F0 = F0 * exp_diff + tl.dot(P, V0)
        F1 = F1 * exp_diff + tl.dot(P, V1)
        
        m = m_new_squeezed
        D_denom = D_new_squeezed
        
    inv_D = 1.0 / D_denom[:, None]
    O0 = F0 * inv_D
    O1 = F1 * inv_D
    
    o_s = n + rows_n[:, None]
    O0_ptr = O + base_offset + o_s * o_strides_2 + cols_d
    O1_ptr = O + base_offset + o_s * o_strides_2 + 64 + cols_d
    tl.store(O0_ptr, O0, mask=q_mask)
    tl.store(O1_ptr, O1, mask=q_mask)
    
    LSE_val = m + tl.log(D_denom)
    lse_ptr = LSE + b * lse_strides_0 + h * lse_strides_1 + (n + rows_n) * lse_strides_2
    tl.store(lse_ptr, LSE_val, mask=(n + rows_n) < S)


def run(Q, K, V, O, LSE):
    """Compute causal MHA and its corresponding LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    q_strides = [Q.stride(0), Q.stride(1), Q.stride(2)]
    k_strides = [K.stride(0), K.stride(1), K.stride(2)]
    v_strides = [V.stride(0), V.stride(1), V.stride(2)]
    o_strides = [O.stride(0), O.stride(1), O.stride(2)]
    lse_strides = [LSE.stride(0), LSE.stride(1), LSE.stride(2)]
    
    BLOCK_N = 64
    BLOCK_K = 64
    
    grid = (triton.cdiv(S, BLOCK_N), B, H)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S, D,
        q_strides[0], q_strides[1], q_strides[2],
        k_strides[0], k_strides[1], k_strides[2],
        v_strides[0], v_strides[1], v_strides[2],
        o_strides[0], o_strides[1], o_strides[2],
        lse_strides[0], lse_strides[1], lse_strides[2],
        scale,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )