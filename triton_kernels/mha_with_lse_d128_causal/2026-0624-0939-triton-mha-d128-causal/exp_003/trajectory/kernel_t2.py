import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    S, D, H, scale,
    desc_q, desc_k, desc_v, desc_o,
    N: tl.constexpr,
):
    b = tl.program_id(1)
    h = tl.program_id(2)
    n_idx = tl.program_id(0)
    n_start = n_idx * N
    
    b_h = b * H + h
    
    seq_n = tl.arange(0, N)
    seq_k = tl.arange(0, 64)
    
    Q = desc_q.load([b_h, n_start, 0]).squeeze(0)
    
    acc_O = tl.zeros((N, 128), dtype=tl.float32)
    m = tl.full((N, 1), -float('inf'), dtype=tl.float32)
    d = tl.zeros((N, 1), dtype=tl.float32)
    
    q_end = min(n_start + N - 1, S - 1)
    k_max = triton.cdiv(q_end + 1, 64) - 1
    if k_max < 0:
        k_max = 0
    
    for j in range(k_max + 1):
        K = desc_k.load([b_h, j * 64, 0]).squeeze(0)
        V = desc_v.load([b_h, j * 64, 0]).squeeze(0)
        
        S_curr = tl.dot(Q, K.T)
        S_curr *= scale
        
        q_pos = n_start + seq_n
        k_pos = j * 64 + seq_k
        valid = (q_pos[:, None] >= k_pos[None, :]) & (q_pos[:, None] < S)
        
        S_curr = tl.where(valid, S_curr, float('-inf'))
        
        m_curr = tl.max(S_curr, axis=1, keep_dims=True)
        m_new = tl.maximum(m, m_curr)
        
        exp_old = tl.exp(m - m_new)
        
        P = tl.exp(S_curr - m_new)
        P = tl.where(valid, P, 0.0)
        
        d_curr = tl.sum(P, axis=1, keep_dims=True)
        d = d * exp_old + d_curr
        
        acc_O = acc_O * exp_old + tl.dot(P, V)
        m = m_new
        
    inv_d = tl.where(d > 0, 1.0 / d, 0.0)
    O = acc_O * inv_d
    
    seq_d = tl.arange(0, 128)
    valid_seq = (n_start + seq_n)[:, None] < S
    
    o_ptr = O + b_h * S * 128 + (n_start + seq_n)[:, None] * 128 + seq_d[None, :]
    tl.store(o_ptr, O.to(tl.bfloat16), mask=valid_seq)
    
    lse_ptr = LSE + b_h * S + n_start + seq_n
    LSE_val = m.squeeze(0) + tl.log(d.squeeze(0))
    tl.store(lse_ptr, LSE_val, mask=(n_start + seq_n) < S)


def run(Q, K, V, O, LSE):
    """Compute causal MHA and its corresponding LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    if Q.data_ptr() % 16 != 0 or K.data_ptr() % 16 != 0 or V.data_ptr() % 16 != 0 or O.data_ptr() % 16 != 0:
        raise ValueError("Pointers must be 16-byte aligned for TMA")
        
    Q_3d = Q.view(B * H, S, D)
    K_3d = K.view(B * H, S, D)
    V_3d = V.view(B * H, S, D)
    O_3d = O.view(B * H, S, D)
    
    N = 64
    
    desc_q = TensorDescriptor.from_tensor(Q_3d, [1, N, 128])
    desc_k = TensorDescriptor.from_tensor(K_3d, [1, N, 128])
    desc_v = TensorDescriptor.from_tensor(V_3d, [1, N, 128])
    desc_o = TensorDescriptor.from_tensor(O_3d, [1, N, 128])
    
    grid = (triton.cdiv(S, N), B, H)
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S, D, H, scale,
        desc_q, desc_k, desc_v, desc_o,
        N=N,
        num_warps=4,
        num_stages=2,
    )