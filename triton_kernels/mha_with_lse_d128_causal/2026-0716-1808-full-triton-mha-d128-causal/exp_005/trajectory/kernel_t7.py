import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def flash_attention_causal(
    q_desc,
    k_desc,
    v_desc,
    o_desc,
    out_lse,
    S_len,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Optimized FlashAttention-Style Causal Forward Pass
    
    Grid map: Dim 0 encodes the Query Tile Step (optimally sized 128 rows), 
              Dim 1 encodes the flattened Batch and Head index (B*H).
    """
    step = tl.program_id(0)
    b_h = tl.program_id(1)
    
    # Limit search span natively by causal constraint masking properties.
    num_kv_steps = min(step + 1, tl.cdiv(S_len, BLOCK_M))
    
    # Fundamental running states initialized for numerically stable Online Softmax calculation.
    m_old = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    l_old = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    
    # Accumulators explicitly partitioned exactly mapping across the 128-element feature width.
    o_acc_0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    o_acc_1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    rows = tl.arange(0, BLOCK_M)
    
    # Load the persistent Q block explicitly mapped across its contiguous head-dimension slices.
    q0 = q_desc.load([b_h * S_len + step * BLOCK_M, 0])
    q1 = q_desc.load([b_h * S_len + step * BLOCK_M, 64])
    
    for i in range(num_kv_steps):
        # Synchronously extract perfectly aligned native 2D block tensors mapped directly via optimized descriptors.
        k0 = k_desc.load([b_h * S_len + i * BLOCK_M, 0])
        k1 = k_desc.load([b_h * S_len + i * BLOCK_M, 64])
        v0 = v_desc.load([b_h * S_len + i * BLOCK_M, 0])
        v1 = v_desc.load([b_h * S_len + i * BLOCK_M, 64])
        
        # Inner reduction accumulating exactly over the complete 128-element head dimension space, 
        # cleanly partitioned into two independent exact 128x64 sub-block WGMMA computations.
        s = tl.dot(q0, k0.T) + tl.dot(q1, k1.T)
        s = s * scale
        
        # Hard boundary enforcing causal logic constraints.
        mask_q = step * BLOCK_M + rows[:, None]
        mask_k = i * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
        causal_mask = (mask_k <= mask_q)
        
        s = tl.where(causal_mask, s, -float('inf'))
        
        m_curr = tl.maximum(m_old, tl.max(s, axis=1))
        p = tl.exp(s - m_curr[:, None])
        p = tl.where(causal_mask, p, 0.0)
        
        exp_m_diff = tl.exp(m_old - m_curr)
        l_curr = l_old * exp_m_diff + tl.sum(p, axis=1)
        
        # Scale prior accumulator dynamically compensating for exponential shifts.
        o_acc_0 *= exp_m_diff[:, None]
        o_acc_1 *= exp_m_diff[:, None]
        
        # Value weightings mapped perfectly over to native D=128 output space.
        p = p.to(tl.bfloat16)
        o_acc_0 = tl.dot(p, v0, acc=o_acc_0)
        o_acc_1 = tl.dot(p, v1, acc=o_acc_1)
                
        m_old = m_curr
        l_old = l_curr
        
    # Finalizing calculations normalizing aggregated outputs to exact expected distribution.
    o_0 = (o_acc_0 / l_old[:, None]).to(tl.bfloat16)
    o_1 = (o_acc_1 / l_old[:, None]).to(tl.bfloat16)
    
    # Direct, highly-optimized native 4D stores avoiding slow runtime indexing evaluation overhead.
    o_desc.store(o_0, [b_h * S_len + step * BLOCK_M, 0])
    o_desc.store(o_1, [b_h * S_len + step * BLOCK_M, 64])
    
    base_offset_lse = b_h * S_len + step * BLOCK_M
    ptr_lse = out_lse + base_offset_lse + rows
    row_mask_lse = (step * BLOCK_M + rows < S_len)
    
    tl.store(ptr_lse, m_old + tl.log(l_old), mask=row_mask_lse)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention forward."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    assert Q.shape == K.shape == V.shape
    
    scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 128
    BLOCK_N = 128
    HEAD_DIM = 128
    
    # Flatten 4D tensors to 2D to avoid complex 4D TMA rank mismatches 
    Q_2d = Q.reshape(B * H * S, D)
    K_2d = K.reshape(B * H * S, D)
    V_2d = V.reshape(B * H * S, D)
    O_2d = O.reshape(B * H * S, D)
    
    # Utilize generic rank-matched 2D TensorDescriptors mapped cleanly over original tensors utilizing optimized blocking
    q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_M, 64])
    k_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_M, 64])
    v_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_M, 64])
    o_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK_M, 64])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    flash_attention_causal[grid](
        q_desc, k_desc, v_desc, o_desc,
        LSE,
        S, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
        num_warps=4,
        num_stages=2,
    )