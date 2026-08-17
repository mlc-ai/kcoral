import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _mha_kernel(
    desc_Q,
    desc_K,
    desc_V,
    O_ptr,
    LSE_ptr,
    S_len,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Non-causal multi-head attention forward pass targeting Hopper (SM90/SM90a).
    Computes Output O and Log-Sum-Exp LSE.
    Replaces broken `extern_shared_memory` logic with safe TMA descriptor loads.
    """
    bh_idx = tl.program_id(0)
    block_row = tl.program_id(1)
    
    # Linear offset in the flattened [B * H * S, D] tensor layout
    offset_s = bh_idx * S_len + block_row * BLOCK_M
    
    # Load the entire Q tile [128, 128]
    q = desc_Q.load([offset_s, 0])
    
    out = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m = tl.full((BLOCK_M, 1), -float("inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    
    total_blocks = tl.cdiv(S_len, BLOCK_N)
    
    for step in range(total_blocks):
        offset_kv = bh_idx * S_len + step * BLOCK_N
        
        # Load K and V tiles [128, 128]
        k = desc_K.load([offset_kv, 0])
        v = desc_V.load([offset_kv, 0])
        
        s = tl.dot(q, k.T)
        s = s * scale
        
        # Mask out sequence lengths exceeding S_len bounding box limit
        valid_k = ((step * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]) < S_len)
        s = tl.where(valid_k, s, -float("inf"))
        
        row_max = tl.max(s, axis=1, keep_dims=True)
        
        m_prev = m
        m_new = tl.maximum(m_prev, row_max)
        
        exp_s = tl.exp(s - m_new)
        sum_exp = tl.sum(exp_s, axis=1, keep_dims=True)
        
        exp_m_diff = tl.exp(m_prev - m_new)
        l_new = exp_m_diff * l + sum_exp
        
        m = m_new
        l = l_new
        
        out = out * exp_m_diff
        
        # Attention outputs calculation using native precision to avoid rank deficiencies 
        p = exp_s.to(tl.bfloat16)
        out = tl.dot(p, v, acc=out)
    
    valid_rows = (block_row * BLOCK_M + tl.arange(0, BLOCK_M)) < S_len
    
    if total_blocks > 0:
        out = out / l
    
    row_offsets = tl.arange(0, BLOCK_M)[:, None] * 128
    col_offsets = tl.arange(0, BLOCK_D)[None, :] * 1
    
    # Mask uninitialized / OOB rows (boundary safe writes protecting generic variable padding logic)
    offset_O = bh_idx * S_len * 128 + block_row * BLOCK_M * 128 + row_offsets + col_offsets
    masked_out = tl.where(valid_rows[:, None], out.to(tl.bfloat16), 0.0)
    tl.store(O_ptr + offset_O, masked_out)
    
    if total_blocks > 0:
        global_lse = m + tl.log(l)
    else:
        global_lse = tl.full((BLOCK_M, 1), 0.0, dtype=tl.float32)
    
    masked_lse = tl.where(valid_rows[:, None], global_lse, 0.0)
    offset_LSE = bh_idx * S_len + block_row * BLOCK_M + tl.arange(0, BLOCK_M)
    
    tl.store(LSE_ptr + offset_LSE, masked_lse.squeeze(1), mask=valid_rows)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_D = 128
    
    Q_flat = Q.view(B * H * S, D)
    K_flat = K.view(B * H * S, D)
    V_flat = V.view(B * H * S, D)
    
    desc_Q = TensorDescriptor.from_tensor(Q_flat, [BLOCK_M, BLOCK_D])
    desc_K = TensorDescriptor.from_tensor(K_flat, [BLOCK_N, BLOCK_D])
    desc_V = TensorDescriptor.from_tensor(V_flat, [BLOCK_N, BLOCK_D])
    
    grid = (B * H, triton.cdiv(S, BLOCK_M))
    
    _mha_kernel[grid](
        desc_Q, desc_K, desc_V,
        O, LSE,
        S, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2
    )