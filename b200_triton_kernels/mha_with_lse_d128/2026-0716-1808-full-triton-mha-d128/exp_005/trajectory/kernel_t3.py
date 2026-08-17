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
    Computes Output O and Log-Sum-Exp LSE. 
    Replaces broken local tensor slicing and implicit shared memory logic with 
    direct usage of TMA-backed block tensors.
    """
    bh_idx = tl.program_id(0)
    block_row = tl.program_id(1)
    
    # Calculate linear row offset bounding our entire Q slice fetch 
    offset_s = bh_idx * S_len + block_row * 128
    
    # Asynchronously fetch independently chunked swirled Q blocks mapping directly to hardware TMA layouts
    q0 = desc_Q.load([offset_s, 0])
    q1 = desc_Q.load([offset_s, 64])
    
    out0 = tl.zeros((128, 64), dtype=tl.float32)
    out1 = tl.zeros((128, 64), dtype=tl.float32)
    
    m = tl.full((128,), -float("inf"), dtype=tl.float32)
    l = tl.zeros((128,), dtype=tl.float32)
    
    total_blocks = tl.cdiv(S_len, 128)
    
    for step in range(total_blocks):
        offset_kv = bh_idx * S_len + step * 128
        
        k0 = desc_K.load([offset_kv, 0])
        k1 = desc_K.load([offset_kv, 64])
        v0 = desc_V.load([offset_kv, 0])
        v1 = desc_V.load([offset_kv, 64])
        
        s = tl.zeros((128, 128), dtype=tl.float32)
        
        # Transpose exactly along innermost D-dimension slices leveraging native WGMMA orientation
        k0_transposed = tl.permute(k0, [1, 0])
        s = tl.dot(q0, k0_transposed, acc=s)
        
        k1_transposed = tl.permute(k1, [1, 0])
        s = tl.dot(q1, k1_transposed, acc=s)
        
        s_scaled = s * scale
        
        valid_k = ((step * 128 + tl.arange(0, 128)[None, :]) < S_len)
        safe_s = tl.where(valid_k, s_scaled, -float("inf"))
        
        row_max = tl.max(safe_s, axis=1, keep_dims=True)
        
        m_prev = m
        m_new = tl.maximum(m_prev, row_max)
        
        exp_s = tl.exp(safe_s - m_new)
        sum_exp = tl.sum(exp_s, axis=1, keep_dims=True)
        
        exp_m_diff = tl.exp(m_prev - m_new)
        l_new = exp_m_diff * l + sum_exp
        
        m = m_new
        l = l_new
        
        out0 = out0 * exp_m_diff
        out1 = out1 * exp_m_diff
        
        # Convert strictly necessary intermediate numerical precision components seamlessly 
        p = exp_s.to(tl.bfloat16)
        
        out0 = tl.dot(p, v0, acc=out0)
        out1 = tl.dot(p, v1, acc=out1)
    
    if total_blocks > 0:
        out0 = out0 / l
        out1 = out1 / l
    
    valid_rows = (block_row * 128 + tl.arange(0, 128)) < S_len
    
    base_offset_O = bh_idx * S_len * 128
    block_offset_O = block_row * 128 * 128
    
    row_offsets = tl.arange(0, 128)[:, None] * 128
    col_offsets_0 = tl.arange(0, 64)[None, :] * 1
    col_offsets_1 = tl.arange(0, 64)[None, :] * 1 + 64
    
    ptr_O_0 = O_ptr + base_offset_O + block_offset_O + row_offsets + col_offsets_0
    ptr_O_1 = O_ptr + base_offset_O + block_offset_O + row_offsets + col_offsets_1
    
    masked_out0 = tl.where(valid_rows[:, None], out0.to(tl.bfloat16), 0.0)
    masked_out1 = tl.where(valid_rows[:, None], out1.to(tl.bfloat16), 0.0)
    
    tl.store(ptr_O_0, masked_out0, mask=valid_rows[:, None])
    tl.store(ptr_O_1, masked_out1, mask=valid_rows[:, None])
    
    if total_blocks > 0:
        global_lse = m + tl.log(l)
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