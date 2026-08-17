import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    S, D, scale,
    desc_q, desc_k, desc_v, desc_o,
    N: tl.constexpr,
    D_QK: tl.constexpr,
):
    b = tl.program_id(1)
    h = tl.program_id(2)
    n_idx = tl.program_id(0)
    n_start = n_idx * N
    
    bh_idx = b * (S // N) + h * (S // N) + n_idx
    
    seq_n = tl.arange(0, N)
    seq_k = tl.arange(0, 64)
    
    Q0 = desc_q.load([bh_idx, n_start, 0]).squeeze(0)
    Q1 = desc_q.load([bh_idx, n_start, 64]).squeeze(0)
    
    acc_O0 = tl.zeros((N, 64), dtype=tl.float32)
    acc_O1 = tl.zeros((N, 64), dtype=tl.float32)
    d_lse = tl.zeros((N, 1), dtype=tl.float32)
    d_scale = tl.zeros((N, 1), dtype=tl.float32)
    
    num_k_iters = triton.cdiv(n_start + N, D_QK)
    
    for k_idx in range(num_k_iters):
        K0 = desc_k.load([bh_idx, k_idx * 64, 0]).squeeze(0)
        K1 = desc_k.load([bh_idx, k_idx * 64, 64]).squeeze(0)
        V0 = desc_v.load([bh_idx, k_idx * 64, 0]).squeeze(0)
        V1 = desc_v.load([bh_idx, k_idx * 64, 64]).squeeze(0)
        
        S_curr = (Q0 @ K0.T) + (Q1 @ K1.T)
        S_curr *= scale
        
        valid = (n_start + seq_n)[:, None] >= (k_idx * 64 + seq_k)[None, :]
        valid = valid & ((n_start + seq_n)[:, None] < S)
        
        S_curr = tl.where(valid, S_curr, float('-inf'))
        
        m_curr = tl.max(S_curr, axis=1, keep_dims=True)
        m_new = tl.maximum(d_lse, m_curr)
        
        exp_old = tl.exp(d_lse - m_new)
        d_lse = m_new
        d_scale = d_scale * exp_old
        
        P = tl.exp(S_curr - m_new)
        P = tl.where(valid, P, 0.0)
        
        d_scale += tl.sum(P, axis=1, keep_dims=True)
        
        acc_O0 += (P @ V0)
        acc_O1 += (P @ V1)
        
    inv_d = 1.0 / d_scale
    O0 = acc_O0 * inv_d
    O1 = acc_O1 * inv_d
    
    seq_d = tl.arange(0, 64)
    l_idx = n_start + seq_n
    valid_seq = l_idx < S
    
    o_base = O + bh_idx * S * D
    ptr0 = o_base + l_idx[:, None] * D + seq_d[None, :]
    ptr1 = o_base + l_idx[:, None] * D + (seq_d + 64)[None, :]
    tl.store(ptr0, O0.to(tl.bfloat16), mask=valid_seq[:, None])
    tl.store(ptr1, O1.to(tl.bfloat16), mask=valid_seq[:, None])
    
    lse_out = d_lse + tl.log(d_scale)
    lse_flat = tl.reshape(lse_out, (N,))
    
    lse_base = LSE + bh_idx * S
    lse_ptr = lse_base + l_idx
    tl.store(lse_ptr, lse_flat, mask=valid_seq)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    if Q.stride(2) % 2 != 0 or K.stride(2) % 2 != 0 or V.stride(2) % 2 != 0 or O.stride(2) % 2 != 0:
        raise ValueError("Strides must be even for TMA alignment")
    if LSE.stride(2) % 4 != 0:
        raise ValueError("LSE stride must be multiple of 4 for TMA alignment")
        
    if Q.data_ptr() % 16 != 0 or K.data_ptr() % 16 != 0 or V.data_ptr() % 16 != 0 or O.data_ptr() % 16 != 0 or LSE.data_ptr() % 16 != 0:
        raise ValueError("Pointers must be 16-byte aligned for TMA")
    
    scale = 1.0 / (D ** 0.5)
    
    Q_3d = Q.view(B * H, S, D)
    K_3d = K.view(B * H, S, D)
    V_3d = V.view(B * H, S, D)
    O_3d = O.view(B * H, S, D)
    
    N = 64
    D_QK = 64
    
    desc_q = TensorDescriptor.from_tensor(Q_3d, [1, N, 128])
    desc_k = TensorDescriptor.from_tensor(K_3d, [1, 64, 128])
    desc_v = TensorDescriptor.from_tensor(V_3d, [1, 64, 128])
    desc_o = TensorDescriptor.from_tensor(O_3d, [1, N, 128])
    
    grid = (triton.cdiv(S, N), B, H)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S, D, scale,
        desc_q, desc_k, desc_v, desc_o,
        N=N,
        D_QK=D_QK,
        num_warps=4,
        num_stages=2,
    )