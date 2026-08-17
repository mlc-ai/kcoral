import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def exp_safe(x):
    """Numerically safe exponential mapping -inf -> 0."""
    return tl.where(x == -tl.inf, 0.0, tl.exp(x))


@triton.jit
def _attention_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    LSE_ptr,
    seq_len,
    H,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    scale,
):
    q_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    q_seq_len = q_block_idx * BLOCK_M
    
    Q = Q_desc.load([b_idx, h_idx, q_seq_len, 0])
    
    O_acc = tl.zeros((BLOCK_M, D), tl.float32)
    m_i = tl.full((BLOCK_M,), -50000.0, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    
    q_valid = q_seq_len + tl.arange(0, BLOCK_M) < seq_len
    
    max_q_pos = min(q_seq_len + BLOCK_M - 1, seq_len - 1)
    block_idx = max_q_pos // BLOCK_N
    
    q_pos = (q_seq_len + tl.arange(0, BLOCK_M))[:, None]
    
    for k_block_idx in range(0, block_idx + 1):
        k_seq_len = k_block_idx * BLOCK_N
        
        K = K_desc.load([b_idx, h_idx, k_seq_len, 0])
        V = V_desc.load([b_idx, h_idx, k_seq_len, 0])
        
        S = tl.dot(Q, K.T) * scale
        
        k_pos = (k_seq_len + tl.arange(0, BLOCK_N))[None, :]
        valid_mask = (q_pos >= k_pos) & q_valid[:, None]
        S = tl.where(valid_mask, S, -tl.inf)
        
        row_max_s = tl.max(S, axis=1, keep_dims=True)
        m_i_new = tl.maximum(m_i[:, None], row_max_s)
        
        alpha = exp_safe(m_i[:, None] - m_i_new)
        O_acc *= alpha
        
        P = exp_safe(S - m_i_new)
        P_sum = tl.sum(P, axis=1, keep_dims=True)
        
        l_i = l_i * exp_safe(m_i - m_i_new[:, 0]) + P_sum[:, 0]
        m_i = m_i_new[:, 0]
        
        P_bf16 = P.to(tl.bfloat16)
        O_acc += tl.dot(P_bf16, V)
        
    O = O_acc / l_i[:, None]
    LSE = m_i + tl.log(l_i)
    
    O = tl.where(q_valid[:, None], O, 0.0)
    O_desc.store([b_idx, h_idx, q_seq_len, 0], O.to(tl.bfloat16))
    
    lse_offsets = (b_idx * H + h_idx) * seq_len + q_seq_len + tl.arange(0, BLOCK_M)
    tl.store(LSE_ptr + lse_offsets, LSE, mask=q_valid)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    BLOCK_M, BLOCK_N = 128, 128
    scale = 1.0 / (D ** 0.5)
    
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D], padding="zero")
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D], padding="zero")
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D], padding="zero")
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D], padding="zero")
    
    grid = (triton.cdiv(S, BLOCK_M), H, B)
    
    _attention_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        S, H, BLOCK_M, BLOCK_N, D, scale,
        num_warps=4, num_stages=2,
    )