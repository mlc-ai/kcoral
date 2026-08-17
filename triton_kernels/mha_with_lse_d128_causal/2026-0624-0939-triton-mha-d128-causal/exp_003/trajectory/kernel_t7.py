import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    desc_q, desc_k, desc_v, desc_o,
    LSE,
    S, H, scale,
    lse_strides_0, lse_strides_1, lse_strides_2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(1)
    h = tl.program_id(2)
    n_idx = tl.program_id(0)
    start_m = n_idx * BLOCK_M
    
    if start_m >= S:
        return
        
    bh_idx = b * H + h
    
    seq_m = tl.arange(0, BLOCK_M)
    seq_k = tl.arange(0, BLOCK_N)
    
    Q0 = desc_q.load([bh_idx, start_m, 0]).squeeze(0)
    Q1 = desc_q.load([bh_idx, start_m, 64]).squeeze(0)
    
    Q0_fp32 = tl.cast(Q0, tl.float32)
    Q1_fp32 = tl.cast(Q1, tl.float32)
    
    acc_O0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    acc_O1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    d = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    max_seq = min(start_m + BLOCK_M - 1, S - 1)
    max_j = max_seq // BLOCK_N
    num_k_iters = max_j + 1
    if num_k_iters > triton.cdiv(S, BLOCK_N):
        num_k_iters = triton.cdiv(S, BLOCK_N)
    
    for j in range(num_k_iters):
        K0 = desc_k.load([bh_idx, j * BLOCK_N, 0]).squeeze(0)
        K1 = desc_k.load([bh_idx, j * BLOCK_N, 64]).squeeze(0)
        V0 = desc_v.load([bh_idx, j * BLOCK_N, 0]).squeeze(0)
        V1 = desc_v.load([bh_idx, j * BLOCK_N, 64]).squeeze(0)
        
        K0_fp32 = tl.cast(K0, tl.float32)
        K1_fp32 = tl.cast(K1, tl.float32)
        
        S_curr = tl.dot(Q0_fp32, K0_fp32.T) + tl.dot(Q1_fp32, K1_fp32.T)
        S_curr *= scale
        
        q_pos = start_m + seq_m
        k_pos = j * BLOCK_N + seq_k
        causal_mask = (k_pos[None, :] <= q_pos[:, None]) & (q_pos[:, None] < S) & (k_pos[None, :] < S)
        
        S_curr = tl.where(causal_mask, S_curr, float('-inf'))
        
        m_curr = tl.max(S_curr, axis=1)
        m_new = tl.maximum(m, m_curr)
        
        exp_old = tl.exp(m - m_new)
        
        P = tl.exp(S_curr - m_new[:, None])
        P = tl.where(causal_mask, P, 0.0)
        
        d_curr = tl.sum(P, axis=1)
        d = d * exp_old + d_curr
        
        acc_O0 = acc_O0 * exp_old[:, None] + tl.dot(P, V0)
        acc_O1 = acc_O1 * exp_old[:, None] + tl.dot(P, V1)
        
        m = m_new
        
    inv_d = tl.where(d > 0, 1.0 / d, 0.0)
    O0 = acc_O0 * inv_d[:, None]
    O1 = acc_O1 * inv_d[:, None]
    
    valid_seq = (start_m + seq_m)[:, None] < S
    
    desc_o.store([bh_idx, start_m, 0], O0.to(tl.bfloat16), mask=valid_seq)
    desc_o.store([bh_idx, start_m, 64], O1.to(tl.bfloat16), mask=valid_seq)
    
    LSE_val = tl.where(d > 0, m + tl.log(d), float('-inf'))
    
    lse_ptr = LSE + b * lse_strides_0 + h * lse_strides_1 + (start_m + seq_m) * lse_strides_2
    tl.store(lse_ptr, LSE_val, mask=(start_m + seq_m) < S)


def run(Q, K, V, O, LSE):
    """Compute causal MHA and its corresponding LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    Q_3d = Q.view(B * H, S, D)
    K_3d = K.view(B * H, S, D)
    V_3d = V.view(B * H, S, D)
    O_3d = O.view(B * H, S, D)
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    desc_q = TensorDescriptor.from_tensor(Q_3d, [1, BLOCK_M, 128], padding_option="zero")
    desc_k = TensorDescriptor.from_tensor(K_3d, [1, BLOCK_N, 128], padding_option="zero")
    desc_v = TensorDescriptor.from_tensor(V_3d, [1, BLOCK_N, 128], padding_option="zero")
    desc_o = TensorDescriptor.from_tensor(O_3d, [1, BLOCK_M, 128])
    
    lse_strides = [LSE.stride(0), LSE.stride(1), LSE.stride(2)]
    
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    _attention_kernel[grid](
        desc_q, desc_k, desc_v, desc_o,
        LSE,
        S, H, scale,
        lse_strides[0], lse_strides[1], lse_strides[2],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )