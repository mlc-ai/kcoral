import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_desc,
    H, S, D,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Optimized non-causal multi-head attention forward kernel."""
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    b_idx = pid_y // H
    h_idx = pid_y % H
    
    start_q = pid_x * BLOCK_Q
    q_row = start_q + tl.arange(0, BLOCK_Q)
    col_D = tl.arange(0, D)
    
    # Load Q tile once; it will be re-used across all inner loop reductions
    q = Q_desc.load([b_idx * H * S + start_q, 0])
    
    o_acc = tl.zeros((BLOCK_Q, D), tl.float32)
    m = tl.full((BLOCK_Q, 1), -float('inf'), tl.float32)
    l = tl.full((BLOCK_Q, 1), 0.0, tl.float32)
    valid_flag = (q_row[None, :] < S).to(tl.float32)
    
    scale = 1.0 / tl.sqrt(D)
    num_kv_tiles = tl.cdiv(S, BLOCK_KV)
    
    for kv_idx in range(num_kv_tiles):
        kv_start = kv_idx * BLOCK_KV
        
        # Fetch K and V blocks from HBM asynchronously via TMA hardware units
        k = K_desc.load([b_idx * H * S + kv_start, 0])
        v = V_desc.load([b_idx * H * S + kv_start, 0])
        
        # 1. Compute Unnormalized Attention Scores (Q @ K^T)
        s = tl.dot(q, k.T) * scale
        
        # 2. Track Running Maximum for numerical stability
        m_prev = m
        m = tl.maximum(m, tl.max(s, axis=1, keep_dims=True))
        
        # 3. Compute Exp(Scores - Max)
        p = tl.exp(s - m)
        
        # 4. Update Normalization Factor
        l_scale = tl.exp(m_prev - m)
        l = l * l_scale + tl.sum(p, axis=1, keep_dims=True)
        
        # 5. Accumulate Output Representation (P @ V)
        o_acc = o_acc * l_scale + tl.dot(p, v)

    # Finalize Output Matrix
    o_final = (o_acc / l).to(tl.bfloat16)
    out_offset = [b_idx * H * S + start_q, 0]
    O_desc.store(out_offset, o_final)
    
    # Finalize Log Sum Exp Vector
    lse = m + tl.log(l)
    lse_val = tl.where(valid_flag.squeeze(), lse.squeeze(), -float('inf'))
    lse_offset = [b_idx * H + h_idx, start_q]
    LSE_desc.store(lse_offset, lse_val[None, :])


def run(Q, K, V, O, LSE):
    """Compute non-causal MHA forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S = Q.shape[0], Q.shape[1], Q.shape[2]
    D = 128
    
    BLOCK_Q = 64
    BLOCK_KV = 64
    
    Q_2d = Q.reshape(B * H * S, D)
    K_2d = K.reshape(B * H * S, D)
    V_2d = V.reshape(B * H * S, D)
    O_2d = O.reshape(B * H * S, D)
    LSE_2d = LSE.reshape(B * H, S)
    
    Q_desc = TensorDescriptor.from_tensor(Q_2d, [BLOCK_Q, D])
    K_desc = TensorDescriptor.from_tensor(K_2d, [BLOCK_KV, D])
    V_desc = TensorDescriptor.from_tensor(V_2d, [BLOCK_KV, D])
    O_desc = TensorDescriptor.from_tensor(O_2d, [BLOCK_Q, D])
    LSE_desc = TensorDescriptor.from_tensor(LSE_2d, [1, BLOCK_Q])
    
    grid = (triton.cdiv(S, BLOCK_Q), B * H)
    
    _mha_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE_desc,
        H, S, D,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        num_warps=8,
        num_stages=2,
    )