import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE,
    H, S,
    scale,
    stride_LSE_b, stride_LSE_h, stride_LSE_s,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    FlashAttention-style MHA kernel for (B, H, S, D) tensors.
    Uses Tensor Descriptors for TMA loads/stores on Hopper.
    Each program processes one (b, h, q_idx) tile of size BLOCK_Q x D.
    Iterates over all N-blocks of K and V to accumulate the output.
    """
    pid = tl.program_id(0)
    num_blocks_q = tl.cdiv(S, BLOCK_Q)
    head_idx = pid // num_blocks_q
    b = head_idx // H
    h = head_idx % H
    q_block = pid % num_blocks_q
    q_idx = q_block * BLOCK_Q
    
    num_blocks_n = tl.cdiv(S, BLOCK_N)
    
    # Load Q tile [BLOCK_Q, 128]
    q_tile = Q_desc.load([b, h, q_idx])
    
    q_row_offs = q_idx + tl.arange(0, BLOCK_Q)
    q_mask = (q_row_offs < S)[:, None]
    
    o = tl.zeros((BLOCK_Q, 128), dtype=tl.float32)
    m = tl.full((BLOCK_Q,), 0.0, dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    
    for n_block in range(num_blocks_n):
        n_idx = n_block * BLOCK_N
        
        # Load K and V tiles [BLOCK_N, 128]
        k_tile = K_desc.load([b, h, n_idx])
        v_tile = V_desc.load([b, h, n_idx])
        
        # SS-GEMM: Q @ K^T
        s = tl.dot(q_tile, k_tile.T)
        s = s * scale
        
        m_old = m
        m = tl.maximum(m_old, tl.max(s, axis=1))
        
        p = tl.exp(s - m)
        
        # Explicit mask enforcing valid matrix boundaries prevents corruption
        n_row_offs = n_idx + tl.arange(0, BLOCK_N)
        k_mask = (n_row_offs < S)[None, :]
        p = p * (q_mask & k_mask)
        
        l_old = l
        l = l_old * tl.exp(m_old - m) + tl.sum(p, axis=1)
        
        o = o * tl.exp(m_old - m)[:, None]
        
        # RS-GEMM: P @ V
        p_bf16 = tl.cast(p, tl.bfloat16)
        o = tl.dot(p_bf16, v_tile, o)
        
    o = o / l[:, None]
    
    # Store O output safely mapping 3D coordinates
    O_desc.store([b, h, q_idx], o)
    
    # Store LSE output safely mapping 1D coordinates
    lse = m + tl.log(l)
    lse_row_offs = q_idx + tl.arange(0, BLOCK_Q)
    lse_mask = lse_row_offs < S
    
    lse_ptr = LSE + b * stride_LSE_b + h * stride_LSE_h + lse_row_offs * stride_LSE_s
    lse = tl.where(lse_mask, lse, 0.0)
    tl.store(lse_ptr, lse, mask=lse_mask)


def run(Q, K, V, O, LSE):
    """Compute Non-causal Multi-Head Attention output O and LogSumExp LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_Q = 64
    BLOCK_N = 64
    
    # Utilizing TMA descriptors bypasses manual pointer math entirely, avoiding memcorruptions
    Q_desc = TensorDescriptor.from_tensor(Q, [B, H, BLOCK_Q])
    K_desc = TensorDescriptor.from_tensor(K, [B, H, BLOCK_N])
    V_desc = TensorDescriptor.from_tensor(V, [B, H, BLOCK_N])
    O_desc = TensorDescriptor.from_tensor(O, [B, H, BLOCK_Q])
    
    num_blocks_q = triton.cdiv(S, BLOCK_Q)
    grid = (B * H * num_blocks_q,)
    
    _mha_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        H, S,
        scale,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_Q=BLOCK_Q, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )