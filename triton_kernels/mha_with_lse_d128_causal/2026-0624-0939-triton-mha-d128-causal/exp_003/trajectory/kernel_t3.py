import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    S, D, H, scale,
    q_strides_0, q_strides_1, q_strides_2,
    k_strides_0, k_strides_1, k_strides_2,
    v_strides_0, v_strides_1, v_strides_2,
    o_strides_0, o_strides_1, o_strides_2,
    lse_strides_0, lse_strides_1, lse_strides_2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(1)
    h = tl.program_id(2)
    n_idx = tl.program_id(0)
    start_m = n_idx * BLOCK_M
    
    base_offset = b * q_strides_0 + h * q_strides_1
    
    seq_m = tl.arange(0, BLOCK_M)
    seq_k = tl.arange(0, BLOCK_N)
    cols_d = tl.arange(0, 64)
    
    q_mask = (start_m + seq_m)[:, None] < S
    
    Q0_ptr = Q + base_offset + (start_m + seq_m[:, None]) * q_strides_2 + cols_d[None, :]
    Q1_ptr = Q + base_offset + (start_m + seq_m[:, None]) * q_strides_2 + 64 + cols_d[None, :]
    Q0 = tl.load(Q0_ptr, mask=q_mask, other=0.0)
    Q1 = tl.load(Q1_ptr, mask=q_mask, other=0.0)
    
    acc_O0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    acc_O1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    m = tl.full((BLOCK_M, 1), -float('inf'), dtype=tl.float32)
    d = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    
    num_k_iters = min(n_idx + 1, tl.cdiv(S, BLOCK_N))
    
    for j in range(num_k_iters):
        k_mask = (j * BLOCK_N + seq_k)[None, :] < S
        
        K0_ptr = K + base_offset + (j * BLOCK_N + seq_k[:, None]) * k_strides_2 + cols_d[None, :]
        K1_ptr = K + base_offset + (j * BLOCK_N + seq_k[:, None]) * k_strides_2 + 64 + cols_d[None, :]
        V0_ptr = V + base_offset + (j * BLOCK_N + seq_k[:, None]) * v_strides_2 + cols_d[None, :]
        V1_ptr = V + base_offset + (j * BLOCK_N + seq_k[:, None]) * v_strides_2 + 64 + cols_d[None, :]
        
        K0 = tl.load(K0_ptr, mask=k_mask, other=0.0)
        K1 = tl.load(K1_ptr, mask=k_mask, other=0.0)
        V0 = tl.load(V0_ptr, mask=k_mask, other=0.0)
        V1 = tl.load(V1_ptr, mask=k_mask, other=0.0)
        
        S_curr = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        S_curr *= scale
        
        q_pos = start_m + seq_m
        k_pos = j * BLOCK_N + seq_k
        causal_mask = (k_pos[None, :] <= q_pos[:, None]) & (q_pos[:, None] < S)
        
        S_curr = tl.where(causal_mask, S_curr, float('-inf'))
        
        m_curr = tl.max(S_curr, axis=1, keep_dims=True)
        m_new = tl.maximum(m, m_curr)
        exp_old = tl.exp(m - m_new)
        
        P = tl.exp(S_curr - m_new)
        P = tl.where(causal_mask, P, 0.0)
        
        d_curr = tl.sum(P, axis=1, keep_dims=True)
        d = d * exp_old + d_curr
        
        acc_O0 = acc_O0 * exp_old + tl.dot(P, V0)
        acc_O1 = acc_O1 * exp_old + tl.dot(P, V1)
        
        m = m_new
        
    inv_d = tl.where(d > 0, 1.0 / d, 0.0)
    O0 = acc_O0 * inv_d
    O1 = acc_O1 * inv_d
    
    valid_seq = (start_m + seq_m)[:, None] < S
    
    o_ptr0 = O + base_offset + (start_m + seq_m[:, None]) * o_strides_2 + cols_d[None, :]
    o_ptr1 = O + base_offset + (start_m + seq_m[:, None]) * o_strides_2 + 64 + cols_d[None, :]
    
    tl.store(o_ptr0, O0.to(tl.bfloat16), mask=valid_seq)
    tl.store(o_ptr1, O1.to(tl.bfloat16), mask=valid_seq)
    
    lse_ptr = LSE + b * lse_strides_0 + h * lse_strides_1 + start_m + seq_m
    LSE_val = m.squeeze(0) + tl.log(d.squeeze(0))
    tl.store(lse_ptr, LSE_val, mask=(start_m + seq_m) < S)


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
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S, D, H, scale,
        q_strides[0], q_strides[1], q_strides[2],
        k_strides[0], k_strides[1], k_strides[2],
        v_strides[0], v_strides[1], v_strides[2],
        o_strides[0], o_strides[1], o_strides[2],
        lse_strides[0], lse_strides[1], lse_strides[2],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )