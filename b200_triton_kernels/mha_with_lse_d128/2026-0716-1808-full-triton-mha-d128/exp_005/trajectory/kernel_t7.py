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
):
    """
    Non-causal multi-head attention forward pass targeting Hopper (SM90/SM90a).
    Computes Output O and Log-Sum-Exp LSE using standard TMA block loads.
    Replaces incorrect manual shared memory management and loop unrolling 
    with direct block tensor operations and implicit software pipelining.
    """
    bh_idx = tl.program_id(0)
    block_row = tl.program_id(1)
    
    offset_s = bh_idx * S_len + block_row * 128
    
    # Load Q tile split into two independent maximum chunk sub-blocks mapped across conflict-free banks.
    q0 = desc_Q.load([offset_s, 0])
    q1 = desc_Q.load([offset_s, 64])
    
    out0 = tl.zeros((128, 64), dtype=tl.float32)
    out1 = tl.zeros((128, 64), dtype=tl.float32)
    
    m = tl.full((128, 1), -float("inf"), dtype=tl.float32)
    l = tl.zeros((128, 1), dtype=tl.float32)
    
    total_blocks = tl.cdiv(S_len, 128)
    
    for step in range(total_blocks):
        offset_kv = bh_idx * S_len + step * 128
        
        k0 = desc_K.load([offset_kv, 0])
        k1 = desc_K.load([offset_kv, 64])
        v0 = desc_V.load([offset_kv, 0])
        v1 = desc_V.load([offset_kv, 64])
        
        s = tl.dot(q0, k0.T)
        s = tl.dot(q1, k1.T, acc=s)
        
        s = s * scale
        
        valid_k = ((step * 128 + tl.arange(0, 128)[None, :]) < S_len)
        s = tl.where(valid_k, s, -float("inf"))
        
        row_max = tl.max(s, axis=1, keep_dims=True)
        
        m_prev = m
        m_new = tl.maximum(m, row_max)
        
        exp_s = tl.exp(s - m_new)
        sum_exp = tl.sum(exp_s, axis=1, keep_dims=True)
        
        exp_m_diff = tl.exp(m_prev - m_new)
        l_new = exp_m_diff * l + sum_exp
        
        m = m_new
        l = l_new
        
        out0 = out0 * exp_m_diff
        out1 = out1 * exp_m_diff
        
        p = exp_s.to(tl.bfloat16)
        
        out0 = tl.dot(p, v0, acc=out0)
        out1 = tl.dot(p, v1, acc=out1)
    
    if total_blocks > 0:
        out0 = out0 / l
        out1 = out1 / l
    
    valid_rows = (block_row * 128 + tl.arange(0, 128)) < S_len
    
    row_offsets = tl.arange(0, 128)[:, None] * 128
    col_offsets_0 = tl.arange(0, 64)[None, :] * 1
    col_offsets_1 = tl.arange(0, 64)[None, :] * 1 + 64
    
    base_offset_O = bh_idx * S_len * 128
    block_offset_O = block_row * 128 * 128
    
    ptr_O_0 = O_ptr + base_offset_O + block_offset_O + row_offsets + col_offsets_0
    ptr_O_1 = O_ptr + base_offset_O + block_offset_O + row_offsets + col_offsets_1
    
    masked_out0 = tl.where(valid_rows[:, None], out0.to(tl.bfloat16), 0.0)
    masked_out1 = tl.where(valid_rows[:, None], out1.to(tl.bfloat16), 0.0)
    
    tl.store(ptr_O_0, masked_out0, mask=valid_rows[:, None])
    tl.store(ptr_O_1, masked_out1, mask=valid_rows[:, None])
    
    if total_blocks > 0:
        global_lse = m.squeeze(1) + tl.log(l.squeeze(1))
    else:
        global_lse = tl.zeros((128,), dtype=tl.float32)
    
    masked_lse = tl.where(valid_rows, global_lse, 0.0)
    
    ptr_LSE = LSE_ptr + bh_idx * S_len + block_row * 128 + tl.arange(0, 128)
    tl.store(ptr_LSE, masked_lse, mask=valid_rows)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    Q_flat = Q.view(B * H * S, D)
    K_flat = K.view(B * H * S, D)
    V_flat = V.view(B * H * S, D)
    
    # Utilize optimally sized sub-block tiling configuration matching standard hardware expectations
    desc_Q = TensorDescriptor.from_tensor(Q_flat, [128, 64])
    desc_K = TensorDescriptor.from_tensor(K_flat, [128, 64])
    desc_V = TensorDescriptor.from_tensor(V_flat, [128, 64])
    
    grid = (B * H, triton.cdiv(S, 128))
    
    _mha_kernel[grid](
        desc_Q, desc_K, desc_V,
        O, LSE,
        S, scale,
        num_warps=4, num_stages=2
    )