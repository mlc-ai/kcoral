import math
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_kernel(
    Q_desc,
    K_desc,
    V_desc,
    O_desc,
    LSE_ptr,
    S,
    scale,
    BLOCK_Q: tl.constexpr,
):
    """
    Optimized FlashAttention-style MHA forward kernel.
    
    Computes O = softmax(Q @ K^T / sqrt(D)) @ V and LSE = log-sum-exp(P)
    where P = Q @ K^T / sqrt(D).
    
    Uses tensor descriptors for TMA loads/stores and WGMMA for dot products.
    Implements online softmax (FlashAttention algorithm).
    """
    bh = tl.program_id(0)
    q_blk = tl.program_id(1)
    d_blk = tl.program_id(2)
    
    q_start = q_blk * BLOCK_Q
    d_start = d_blk * 64
    
    q_idx = q_start + tl.arange(0, BLOCK_Q)
    
    # Load our Q slice exactly once for both D-chunks.
    Q0 = tl.squeeze(Q_desc.load([bh, 0, q_start, 0]))
    Q1 = tl.squeeze(Q_desc.load([bh, 0, q_start, 64]))
    
    # Maintain precise FP32 online-softmax state over the full sequence
    m = tl.full((BLOCK_Q,), -1e20, dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_Q, 64), dtype=tl.float32)
    
    num_kv_blks = tl.cdiv(S, 64)
    
    # Sequentially pipeline all KV blocks along the Sequence dimension
    for k_blk in range(num_kv_blks):
        # QK^T Dot Product accumulation across both halves of D
        K0 = tl.squeeze(K_desc.load([bh, 0, k_blk * 64, 0]))
        K1 = tl.squeeze(K_desc.load([bh, 0, k_blk * 64, 64]))
        
        S_block = tl.dot(Q0, K0.T, out_dtype=tl.float32)
        S_block = tl.dot(Q1, K1.T, S_block, out_dtype=tl.float32)
        
        # Standard FlashAttention maximum tracking & exponentiation logic
        S_block = S_block * scale
        
        # Masking for partially filled tiles to ensure 0 contribution
        kv_idx = k_blk * 64 + tl.arange(0, 64)
        mask = (kv_idx < S)
        S_block = tl.where(mask[None, :], S_block, -1e20)
        
        cur_m = tl.max(S_block, axis=1)
        m_new = tl.maximum(m, cur_m)
        
        P = tl.exp(S_block - m_new[:, None])
        cur_l = tl.sum(P, axis=1)
        
        # Accurately preserve numerics when updating cumulative statistics mid-loop
        l_new = l * tl.exp(m - m_new) + cur_l
        
        O_acc = O_acc * tl.exp(m - m_new)[:, None]
        
        m = m_new
        l = l_new
        
        # PV Dot Product accumulation
        V = tl.squeeze(V_desc.load([bh, 0, k_blk * 64, d_start]))
        O_acc = tl.dot(P, V, O_acc, out_dtype=tl.float32)
        
    # Final division by cumulative scalar sums to resolve correct probabilities
    O = O_acc / l[:, None]
    
    # Store output block
    O_desc.store([bh, 0, q_start, d_start], O.to(tl.bfloat16), boundary_check=((2, S),))
    
    # Emit natural logarithm directly matching standard PyTorch SDPA conventions
    if d_blk == 0:
        lse_ptr = LSE_ptr + bh * S + q_idx
        lse = m + tl.log(l)
        tl.store(lse_ptr, lse, mask=(q_idx < S))


def run(Q, K, V, O, LSE):
    """Compute Multi-Head Attention O and Log-Sum-Exp LSE into preallocated CUDA tensors."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    Q_desc = TensorDescriptor.from_tensor(Q, block_shape=[1, 1, BLOCK_Q, 64])
    K_desc = TensorDescriptor.from_tensor(K, block_shape=[1, 1, 64, 64])
    V_desc = TensorDescriptor.from_tensor(V, block_shape=[1, 1, 64, 64])
    O_desc = TensorDescriptor.from_tensor(O, block_shape=[1, 1, BLOCK_Q, 64])
    
    # Use exactly 128 rows per block and 4 active warps 
    BLOCK_Q = 128
    grid = ((B * H, triton.cdiv(S, BLOCK_Q)), (triton.cdiv(D, 64),))
    
    _mha_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE,
        S,
        scale,
        BLOCK_Q=BLOCK_Q,
        num_warps=4,
        num_stages=3,
    )