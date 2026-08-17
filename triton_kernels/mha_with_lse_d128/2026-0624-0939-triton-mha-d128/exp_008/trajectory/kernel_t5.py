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
    q_tile = Q_desc.load([b, h, q_idx, 0])
    q_tile = tl.reshape(q_tile, [BLOCK_Q, 128])
    
    q_row_offs = q_idx + tl.arange(0, BLOCK_Q)
    q_mask = (q_row_offs < S)[:, None]
    
    o = tl.zeros((BLOCK_Q, 128), dtype=tl.float32)
    m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    
    for n_block in range(num_blocks_n):
        n_idx = n_block * BLOCK_N
        
        # Load K tile [BLOCK_N, 128]
        k_tile = K_desc.load([b, h, n_idx, 0])
        k_tile = tl.reshape(k_tile, [BLOCK_N, 128])
        
        # SS-GEMM: Q @ K^T
        s = tl.dot(q_tile, k_tile.T)
        s = s * scale
        
        # Explicit mask enforcing valid matrix boundaries prevents corruption
        n_row_offs = n_idx + tl.arange(0, BLOCK_N)
        k_mask = (n_row_offs < S)[None, :]
        s = tl.where(q_mask & k_mask, s, -float('inf'))
        
        m_old = m
        m = tl.maximum(m, tl.max(s, axis=1))
        p = tl.exp(s - m)
        
        l_old = l
        l = l_old * tl.exp(m_old - m) + tl.sum(p, axis=1)
        
        o = o * tl.exp(m_old - m)[:, None]
        
        # Load V tile [BLOCK_N, 128]
        v_tile = V_desc.load([b, h, n_idx, 0])
        v_tile = tl.reshape(v_tile, [BLOCK_N, 128])
        
        # RS-GEMM: P @ V
        p_bf16 = tl.cast(p, tl.bfloat16)
        o = tl.dot(p_bf16, v_tile, o)
        
    o = o / l[:, None]
    
    # Store O output
    o_tile = tl.reshape(o, [1, 1, BLOCK_Q, 128])
    O_desc.store([b, h, q_idx, 0], o_tile)
    
    # Store LSE output safely mapping 1D coordinates
    lse = m + tl.log(l)
    lse_row_offs = q_idx + tl.arange(0, BLOCK_Q)
    lse_mask = lse_row_offs < S
    
    b_offs_lse = tl.full((BLOCK_Q,), b * stride_LSE_b, dtype=tl.int32)
    h_offs_lse = tl.full((BLOCK_Q,), h * stride_LSE_h, dtype=tl.int32)
    lse_offs = b_offs_lse + h_offs_lse + lse_row_offs * stride_LSE_s
    tl.store(LSE + lse_offs, lse, mask=lse_mask)


def run(Q, K, V, O, LSE):
    """Compute Non-causal Multi-Head Attention output O and LogSumExp LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    
    BLOCK_Q = 64
    BLOCK_N = 64
    
    # Utilizing TMA descriptors bypasses manual pointer math entirely, avoiding memcorruptions
    Q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_Q, 128])
    K_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, 128])
    V_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, 128])
    O_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_Q, 128])
    
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