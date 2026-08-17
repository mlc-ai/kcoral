import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


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
    scale: tl.constexpr,
):
    """
    Optimized causal attention kernel utilizing TMA descriptors for all HBM loads and stores.
    Computes O and Log-Sum-Exp (LSE) for bf16 inputs targeting Hopper architectures.
    """
    q_block_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)
    
    q_seq_len = q_block_idx * BLOCK_M
    
    # Causal mask limits iteration to the maximum relevant key block
    q_actual_len = min(q_seq_len + BLOCK_M, seq_len)
    block_idx = min(q_actual_len // BLOCK_N, seq_len // BLOCK_N - 1)
    
    # Load Q tile directly using TMA
    Q = Q_desc.load([b_idx, h_idx, q_seq_len, 0])
    
    # Accumulators initialized to identity values
    O_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    m_i = tl.full((BLOCK_M,), -tl.inf, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    
    # Precognised query coordinates relative to global sequence
    q_pos = (q_seq_len + tl.arange(0, BLOCK_M))[:, None]
    
    for k_block_idx in range(0, block_idx + 1):
        k_seq_len = k_block_idx * BLOCK_N
        
        K = K_desc.load([b_idx, h_idx, k_seq_len, 0])
        V = V_desc.load([b_idx, h_idx, k_seq_len, 0])
        
        S = tl.dot(Q, K.T) * scale
        
        # Apply lower triangular boundary condition dynamically
        k_pos = (k_seq_len + tl.arange(0, BLOCK_N))[None, :]
        valid_mask = q_pos >= k_pos
        S = tl.where(valid_mask, S, -tl.inf)
        
        # Standard FlashAttention state updates mapped cleanly across CTAs
        m_i_new = tl.maximum(tl.max(S, axis=1, keep_dims=True), m_i[:, None])
        O_acc *= tl.exp(m_i[:, None] - m_i_new)
        
        P = tl.where(m_i_new == -tl.inf, 0.0, tl.exp(S - m_i_new))
        P_sum = tl.sum(P, axis=1, keep_dims=True)
        
        l_i = l_i * tl.exp(m_i[:, None] - m_i_new[:, 0]) + P_sum[:, 0]
        m_i = m_i_new[:, 0]
        
        # Ensure strictly matching precision domains for optimized WGMMA execution routing 
        P_bf16 = P.to(tl.bfloat16)
        O_acc += tl.dot(P_bf16, V)
        
    # Normalize Output and Extrapolate final block Log-Sum-Exponentials (natively base-e)
    O = O_acc / l_i[:, None]
    LSE = m_i + tl.log(l_i)
    
    # Dispatch results cleanly leveraging built-in TMA boundary checks 
    O_desc.store([b_idx, h_idx, q_seq_len, 0], O)
    
    lse_offsets = (b_idx * H + h_idx) * seq_len + q_seq_len + tl.arange(0, BLOCK_M)
    q_valid = q_seq_len + tl.arange(0, BLOCK_M) < seq_len
    tl.store(LSE_ptr + lse_offsets, LSE, mask=q_valid)


def run(Q, K, V, O, LSE):
    """
    Host-side wrapper orchestrating TMA descriptors and issuing the causal attention kernel.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    BLOCK_M, BLOCK_N = 128, 128
    scale = 1.0 / (D ** 0.5)
    
    Q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_M, BLOCK_N])
    K_desc = TensorDescriptor.from_tensor(K, [BLOCK_M, BLOCK_N])
    V_desc = TensorDescriptor.from_tensor(V, [BLOCK_M, BLOCK_N])
    O_desc = TensorDescriptor.from_tensor(O, [BLOCK_M, BLOCK_N])
    
    grid = (triton.cdiv(S, BLOCK_M), H, B)
    
    _attention_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        S, H, BLOCK_M, BLOCK_N, scale=scale,
        num_warps=4, num_stages=3,
    )