import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _mha_fwd_kernel(
    Q_desc, K_desc, V_desc, O_desc, LSE_desc,
    H, S, scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Optimized non-causal multi-head attention forward kernel."""
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    b_idx = pid_y // H
    h_idx = pid_y % H
    
    start_q = pid_x * BLOCK_Q
    
    # Load Q tile once; it will be re-used across all inner loop reductions
    q0 = Q_desc.load([b_idx, h_idx, start_q, 0]).squeeze()
    q1 = Q_desc.load([b_idx, h_idx, start_q, 64]).squeeze()
    
    o_acc0 = tl.zeros((BLOCK_Q, 64), tl.float32)
    o_acc1 = tl.zeros((BLOCK_Q, 64), tl.float32)
    m = tl.full((BLOCK_Q, 1), -float('inf'), tl.float32)
    l = tl.full((BLOCK_Q, 1), 0.0, tl.float32)
    
    num_kv_tiles = tl.cdiv(S, BLOCK_KV)
    
    for kv_idx in range(num_kv_tiles):
        kv_start = kv_idx * BLOCK_KV
        
        # Fetch K and V blocks from HBM asynchronously via TMA hardware units
        k0 = K_desc.load([b_idx, h_idx, kv_start, 0]).squeeze()
        k1 = K_desc.load([b_idx, h_idx, kv_start, 64]).squeeze()
        v0 = V_desc.load([b_idx, h_idx, kv_start, 0]).squeeze()
        v1 = V_desc.load([b_idx, h_idx, kv_start, 64]).squeeze()
        
        # 1. Compute Unnormalized Attention Scores (Q @ K^T)
        s_acc = tl.zeros((BLOCK_Q, BLOCK_KV), tl.float32)
        s_acc = tl.dot(q0, k0.T, acc=s_acc)
        s_acc = tl.dot(q1, k1.T, acc=s_acc)
        s = s_acc * scale
        
        # 2. Track Running Maximum for numerical stability
        m_prev = m
        m = tl.maximum(m, tl.max(s, axis=1, keep_dims=True))
        
        # 3. Compute Exp(Scores - Max)
        p = tl.exp(s - m)
        
        # 4. Update Normalization Factor
        l_scale = tl.exp(m_prev - m)
        l = l * l_scale + tl.sum(p, axis=1, keep_dims=True)
        
        # 5. Accumulate Output Representation (P @ V)
        o_acc0 = o_acc0 * l_scale + tl.dot(p, v0)
        o_acc1 = o_acc1 * l_scale + tl.dot(p, v1)

    # Finalize Output Matrix
    o_final0 = (o_acc0 / l).to(tl.bfloat16)
    o_final1 = (o_acc1 / l).to(tl.bfloat16)
    O_desc.store([b_idx, h_idx, start_q, 0], o_final0[None, None, :, :])
    O_desc.store([b_idx, h_idx, start_q, 64], o_final1[None, None, :, :])
    
    # Finalize Log Sum Exp Vector
    lse = m + tl.log(l)
    q_row = start_q + tl.arange(0, BLOCK_Q)
    valid_flag = (q_row < S).to(tl.float32)
    lse_val = tl.where(valid_flag > 0, lse.squeeze(), -float('inf'))
    LSE_desc.store([b_idx * H + h_idx, start_q], lse_val[None, :])


def run(Q, K, V, O, LSE):
    """Compute non-causal MHA forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S = Q.shape[0], Q.shape[1], Q.shape[2]
    D = Q.shape[3]
    
    scale = 1.0 / math.sqrt(D)
    
    BLOCK_Q = 64
    BLOCK_KV = 64
    
    Q_c = Q.contiguous().reshape(B, H, S, D)
    K_c = K.contiguous().reshape(B, H, S, D)
    V_c = V.contiguous().reshape(B, H, S, D)
    O_c = O.contiguous().reshape(B, H, S, D)
    LSE_2d = LSE.reshape(B * H, S)
    
    Q_desc = TensorDescriptor.from_tensor(Q_c, [1, 1, BLOCK_Q, 64])
    K_desc = TensorDescriptor.from_tensor(K_c, [1, 1, BLOCK_KV, 64])
    V_desc = TensorDescriptor.from_tensor(V_c, [1, 1, BLOCK_KV, 64])
    O_desc = TensorDescriptor.from_tensor(O_c, [1, 1, BLOCK_Q, 64])
    LSE_desc = TensorDescriptor.from_tensor(LSE_2d, [1, BLOCK_Q])
    
    grid = (triton.cdiv(S, BLOCK_Q), B * H)
    
    _mha_fwd_kernel[grid](
        Q_desc, K_desc, V_desc, O_desc, LSE_desc,
        H, S, scale,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
        num_warps=8,
        num_stages=2,
    )