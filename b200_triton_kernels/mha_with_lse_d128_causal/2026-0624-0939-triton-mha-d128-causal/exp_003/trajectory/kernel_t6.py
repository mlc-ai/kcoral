import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _attention_kernel(
    descs_q, descs_k, descs_v, descs_o,
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
    
    b_h_idx = b * H + h
    
    seq_m = tl.arange(0, BLOCK_M)
    seq_k = tl.arange(0, BLOCK_N)
    
    desc_q = descs_q[b_h_idx]
    desc_k = descs_k[b_h_idx]
    desc_v = descs_v[b_h_idx]
    desc_o = descs_o[b_h_idx]
    
    Q0 = desc_q.load([start_m, 0])
    Q1 = desc_q.load([start_m, 64])
    
    acc_O0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    acc_O1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    d = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    max_j = (start_m + BLOCK_M - 1) // BLOCK_N
    num_k_iters = max_j + 1
    if num_k_iters > tl.cdiv(S, BLOCK_N):
        num_k_iters = tl.cdiv(S, BLOCK_N)
    
    for j in range(num_k_iters):
        K0 = desc_k.load([j * BLOCK_N, 0])
        K1 = desc_k.load([j * BLOCK_N, 64])
        V0 = desc_v.load([j * BLOCK_N, 0])
        V1 = desc_v.load([j * BLOCK_N, 64])
        
        S_curr = tl.dot(Q0, K0.T) + tl.dot(Q1, K1.T)
        S_curr *= scale
        
        q_pos = start_m + seq_m
        k_pos = j * BLOCK_N + seq_k
        causal_mask = (k_pos[None, :] <= q_pos[:, None]) & (q_pos[:, None] < S)
        
        S_curr = tl.where(causal_mask, S_curr, float('-inf'))
        
        m_curr = tl.max(S_curr, axis=1)
        m_new = tl.maximum(m, m_curr)
        
        exp_old = tl.exp(m - m_new)
        
        P = tl.exp(S_curr - m_new[:, None])
        P = tl.where(causal_mask, P, 0.0)
        
        d_curr = tl.sum(P, axis=1)
        d = d * exp_old + d_curr
        
        V0_fp32 = tl.cast(V0, tl.float32)
        V1_fp32 = tl.cast(V1, tl.float32)
        
        acc_O0 = acc_O0 * exp_old[:, None] + tl.dot(P, V0_fp32)
        acc_O1 = acc_O1 * exp_old[:, None] + tl.dot(P, V1_fp32)
        
        m = m_new
        
    inv_d = tl.where(d > 0, 1.0 / d, 0.0)
    O0 = acc_O0 * inv_d[:, None]
    O1 = acc_O1 * inv_d[:, None]
    
    desc_o.store([start_m, 0], O0.to(tl.bfloat16))
    desc_o.store([start_m, 64], O1.to(tl.bfloat16))
    
    lse_d = tl.where(d > 0, d, 1.0)
    LSE_val = m + tl.log(lse_d)
    
    lse_ptr = LSE + b * lse_strides_0 + h * lse_strides_1 + (start_m + seq_m) * lse_strides_2
    tl.store(lse_ptr, LSE_val, mask=(start_m + seq_m) < S)


def run(Q, K, V, O, LSE):
    """Compute causal MHA and its corresponding LSE."""
    torch.cuda.set_device(Q.device)
    
    def ensure_aligned(tensor, name):
        ptr = tensor.data_ptr()
        if ptr % 16 != 0:
            offset_bytes = 16 - (ptr % 16)
            offset_elements = offset_bytes // 2
            return tensor.reshape(-1)[offset_elements:].reshape(tensor.shape)
        return tensor

    Q = ensure_aligned(Q, "Q")
    K = ensure_aligned(K, "K")
    V = ensure_aligned(V, "V")
    O = ensure_aligned(O, "O")
    LSE = ensure_aligned(LSE, "LSE")
    
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    Q_3d = Q.view(B * H, S, D)
    K_3d = K.view(B * H, S, D)
    V_3d = V.view(B * H, S, D)
    O_3d = O.view(B * H, S, D)
    
    BLOCK_M = 64
    BLOCK_N = 64
    
    descs_q = []
    descs_k = []
    descs_v = []
    descs_o = []
    
    for b_h_idx in range(B * H):
        Q_b_h = Q_3d[b_h_idx]
        K_b_h = K_3d[b_h_idx]
        V_b_h = V_3d[b_h_idx]
        O_b_h = O_3d[b_h_idx]
        
        descs_q.append(TensorDescriptor.from_tensor(Q_b_h, [BLOCK_M, 64]))
        descs_k.append(TensorDescriptor.from_tensor(K_b_h, [BLOCK_N, 64]))
        descs_v.append(TensorDescriptor.from_tensor(V_b_h, [BLOCK_N, 64]))
        descs_o.append(TensorDescriptor.from_tensor(O_b_h, [BLOCK_M, 64]))
        
    lse_strides = [LSE.stride(0), LSE.stride(1), LSE.stride(2)]
    
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    _attention_kernel[grid](
        descs_q, descs_k, descs_v, descs_o,
        LSE,
        S, H, scale,
        lse_strides[0], lse_strides[1], lse_strides[2],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )