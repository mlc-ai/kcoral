import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _mha_kernel(Q, K, V, O, LSE, S, desc_Q, desc_K, desc_V, scale):
    """
    Optimized non-causal multi-head attention kernel targeting Hopper (SM90).
    Computes Output O and Log-Sum-Exp LSE.
    """
    bh_idx = tl.program_id(0)
    block_row = tl.program_id(1)
    
    st_Q = triton.extern_shared_memory((2, 128, 64), dtype=tl.bfloat16)
    st_K = triton.extern_shared_memory((2, 128, 128, 2), dtype=tl.bfloat16)
    st_V = triton.extern_shared_memory((2, 128, 128, 2), dtype=tl.bfloat16)
    st_P = triton.extern_shared_memory((128, 128), dtype=tl.bfloat16)
    
    ptr_to_st_Q = st_Q.to_ptr()
    ptr_to_st_Q = ptr_to_st_Q.reshape((2, 128, 64)).to_ptr()
    
    ptr_to_st_K = st_K.to_ptr()
    ptr_to_st_K = ptr_to_st_K.reshape((2, 128, 128, 2)).to_ptr()
    
    ptr_to_st_V = st_V.to_ptr()
    ptr_to_st_V = ptr_to_st_V.reshape((2, 128, 128, 2)).to_ptr()
    
    ptr_to_st_P = st_P.to_ptr()
    
    row_ptr_q = (tl.arange(0, 128)[:, None] * 128)
    row_ptr_k = (tl.arange(0, 128)[:, None] * 128)
    col_ptr_0 = (tl.arange(0, 64)[None, :] * 1)
    col_ptr_1 = (tl.arange(0, 64)[None, :] * 1) + 64
    
    row_ptr_p = (tl.arange(0, 128)[:, None] * 128)
    col_ptr_p = (tl.arange(0, 128)[None, :] * 1)
    
    O_ptr = O.to_ptr()
    LSE_ptr = LSE.to_ptr()
    
    desc_Q.load_async(ptr_to_st_Q[0], [block_row * 128, 0])
    desc_Q.load_async(ptr_to_st_Q[1], [block_row * 128, 64])
    
    out0 = tl.zeros((128, 64), dtype=tl.float32)
    out1 = tl.zeros((128, 64), dtype=tl.float32)
    m = tl.full((128,), -float("inf"), dtype=tl.float32)
    l = tl.zeros((128,), dtype=tl.float32)
    
    total_blocks = tl.cdiv(S, 128)
    
    if total_blocks > 0:
        desc_K.load_async(ptr_to_st_K[0][0], [0 * 128, 0])
        desc_K.load_async(ptr_to_st_K[0][1], [0 * 128, 64])
        
        desc_V.load_async(ptr_to_st_V[0][0], [0 * 128, 0])
        desc_V.load_async(ptr_to_st_V[0][1], [0 * 128, 64])
    
    for step in range(total_blocks):
        stage = step % 2
        
        q0 = tl.load(ptr_to_st_Q[0] + row_ptr_q + col_ptr_0, cache_modifier=".cg")
        q1 = tl.load(ptr_to_st_Q[1] + row_ptr_q + col_ptr_1, cache_modifier=".cg")
        
        k0 = tl.load(ptr_to_st_K[stage][0] + row_ptr_k + col_ptr_0, cache_modifier=".cg")
        k1 = tl.load(ptr_to_st_K[stage][1] + row_ptr_k + col_ptr_1, cache_modifier=".cg")
        
        s = tl.zeros((128, 128), dtype=tl.float32)
        
        k0_transposed = tl.permute(k0, [1, 0])
        s = tl.dot(q0, k0_transposed, acc=s)
        
        k1_transposed = tl.permute(k1, [1, 0])
        s = tl.dot(q1, k1_transposed, acc=s)
        
        s_scaled = s * scale
        
        valid_k = (step * 128 + tl.arange(0, 128)[None, :]) < S
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
        
        p_bf16 = exp_s.to(tl.bfloat16)
        tl.store(ptr_to_st_P + row_ptr_p + col_ptr_p, p_bf16)
        
        v0 = tl.load(ptr_to_st_V[stage][0] + row_ptr_k + col_ptr_0, cache_modifier=".cg")
        v1 = tl.load(ptr_to_st_V[stage][1] + row_ptr_k + col_ptr_1, cache_modifier=".cg")
        
        p = tl.load(ptr_to_st_P + row_ptr_p + col_ptr_p)
        
        out0 = tl.dot(p, v0, acc=out0)
        out1 = tl.dot(p, v1, acc=out1)
        
        if step + 1 < total_blocks:
            next_stage = (step + 1) % 2
            desc_K.load_async(ptr_to_st_K[next_stage][0], [(step + 1) * 128, 0])
            desc_K.load_async(ptr_to_st_K[next_stage][1], [(step + 1) * 128, 64])
            
            desc_V.load_async(ptr_to_st_V[next_stage][0], [(step + 1) * 128, 0])
            desc_V.load_async(ptr_to_st_V[next_stage][1], [(step + 1) * 128, 64])
    
    out0 = out0 / l
    out1 = out1 / l
    
    valid_rows = (block_row * 128 + tl.arange(0, 128)[:, None]) < S
    
    masked_out0 = tl.where(valid_rows, out0.to(tl.bfloat16), 0.0)
    masked_out1 = tl.where(valid_rows, out1.to(tl.bfloat16), 0.0)
    
    ptr_O_0 = O_ptr + bh_idx * S * 128 + block_row * 128 * 128 + row_ptr_q * 128 + col_ptr_0 * 1
    ptr_O_1 = O_ptr + bh_idx * S * 128 + block_row * 128 * 128 + row_ptr_q * 128 + col_ptr_1 * 1
    
    tl.store(ptr_O_0, masked_out0, mask=valid_rows)
    tl.store(ptr_O_1, masked_out1, mask=valid_rows)
    
    global_lse = m + tl.log(l)
    masked_lse = tl.where(valid_rows, global_lse, 0.0)
    valid_lse = valid_rows.squeeze(1)
    
    ptr_LSE = LSE_ptr + bh_idx * S + block_row * 128 + tl.arange(0, 128)
    tl.store(ptr_LSE, masked_lse.squeeze(1), mask=valid_lse)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    Q_flat = Q.view(B * H * S, D)
    K_flat = K.view(B * H * S, D)
    V_flat = V.view(B * H * S, D)
    
    desc_Q = TensorDescriptor.from_tensor(Q_flat, [128, 64])
    desc_K = TensorDescriptor.from_tensor(K_flat, [128, 64])
    desc_V = TensorDescriptor.from_tensor(V_flat, [128, 64])
    
    grid = (B * H, triton.cdiv(S, 128))
    
    _mha_kernel[grid](
        Q, K, V, O, LSE, S,
        desc_Q, desc_K, desc_V, scale,
        num_warps=4, num_stages=2
    )