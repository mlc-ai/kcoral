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
    Computes Output O and Log-Sum-Exp LSE. Replaces broken local tensor slicing 
    and implicit shared memory logic with a fully asynchronous TMA swizzling pipeline.
    """
    bh_idx = tl.program_id(0)
    block_row = tl.program_id(1)
    
    # Allocate strictly defined static shared memory layouts matching exact physical sub-block sizes
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
    
    offset_s = bh_idx * S_len + block_row * 128
    
    # Prime TMA queue dynamically for Q block slice utilizing dual independent maximum chunk fetches seamlessly across bank conflicts 
    desc_Q.make_async(ptr_to_st_Q[0], [offset_s, 0])
    desc_Q.make_async(ptr_to_st_Q[1], [offset_s, 64])
    
    triton.commit_async_descriptor(desc_Q, 0)
    triton.wait_descriptor_barrier(0)
    
    out0 = tl.zeros((128, 64), dtype=tl.float32)
    out1 = tl.zeros((128, 64), dtype=tl.float32)
    m = tl.full((128,), -float("inf"), dtype=tl.float32)
    l = tl.zeros((128,), dtype=tl.float32)
    
    total_blocks = tl.cdiv(S_len, 128)
    
    if total_blocks > 0:
        ptr_K0_0 = ptr_to_st_K[0][0]
        ptr_K0_1 = ptr_to_st_K[0][1]
        ptr_V0_0 = ptr_to_st_V[0][0]
        ptr_V0_1 = ptr_to_st_V[0][1]
        
        desc_K.make_async(ptr_K0_0, [0 * 128, 0])
        desc_K.make_async(ptr_K0_1, [0 * 128, 64])
        
        desc_V.make_async(ptr_V0_0, [0 * 128, 0])
        desc_V.make_async(ptr_V0_1, [0 * 128, 64])
        
        cur_barrier = 0
        triton.commit_async_descriptor(desc_K, cur_barrier)
        triton.commit_async_descriptor(desc_V, cur_barrier)
    
    for step in range(total_blocks):
        stage = step % 2
        
        ptr_K_stage_0 = ptr_to_st_K[stage][0]
        ptr_K_stage_1 = ptr_to_st_K[stage][1]
        
        ptr_V_stage_0 = ptr_to_st_V[stage][0]
        ptr_V_stage_1 = ptr_to_st_V[stage][1]
        
        # Async wait ensures precise synchronization preventing pipeline starvation starvation mitigating subsequent latency timeouts effectively 
        wait_barrier = (step + 1) % 2
        triton.wait_descriptor_barrier(wait_barrier)
        
        q0 = tl.load(ptr_to_st_Q[0] + row_ptr_q + col_ptr_0, cache_modifier=".cg")
        q1 = tl.load(ptr_to_st_Q[1] + row_ptr_q + col_ptr_1, cache_modifier=".cg")
        
        k0 = tl.load(ptr_K_stage_0 + row_ptr_k + col_ptr_0, cache_modifier=".cg")
        k1 = tl.load(ptr_K_stage_1 + row_ptr_k + col_ptr_1, cache_modifier=".cg")
        
        s = tl.zeros((128, 128), dtype=tl.float32)
        
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
        
        # P stored in shared memory to avoid register pressure during second GEMM phase
        p_bf16 = exp_s.to(tl.bfloat16)
        tl.store(ptr_to_st_P + row_ptr_p + col_ptr_p, p_bf16)
        
        v0 = tl.load(ptr_V_stage_0 + row_ptr_k + col_ptr_0, cache_modifier=".cg")
        v1 = tl.load(ptr_V_stage_1 + row_ptr_k + col_ptr_1, cache_modifier=".cg")
        
        p = tl.load(ptr_to_st_P + row_ptr_p + col_ptr_p)
        
        out0 = tl.dot(p, v0, acc=out0)
        out1 = tl.dot(p, v1, acc=out1)
        
        if step + 1 < total_blocks:
            next_stage = (step + 1) % 2
            ptr_K_next_0 = ptr_to_st_K[next_stage][0]
            ptr_K_next_1 = ptr_to_st_K[next_stage][1]
            ptr_V_next_0 = ptr_to_st_V[next_stage][0]
            ptr_V_next_1 = ptr_to_st_V[next_stage][1]
            
            desc_K.make_async(ptr_K_next_0, [(step + 1) * 128, 0])
            desc_K.make_async(ptr_K_next_1, [(step + 1) * 128, 64])
            
            desc_V.make_async(ptr_V_next_0, [(step + 1) * 128, 0])
            desc_V.make_async(ptr_V_next_1, [(step + 1) * 128, 64])
            
            commit_barrier = (step + 2) % 2
            triton.commit_async_descriptor(desc_K, commit_barrier)
            triton.commit_async_descriptor(desc_V, commit_barrier)
    
    if total_blocks > 0:
        out0 = out0 / l
        out1 = out1 / l
    
    valid_rows = (block_row * 128 + tl.arange(0, 128)) < S_len
    
    base_offset_O = bh_idx * S_len * 128
    block_offset_O = block_row * 128 * 128
    
    ptr_O_0 = O_ptr + base_offset_O + block_offset_O + row_ptr_q * 128 + col_ptr_0 * 1
    ptr_O_1 = O_ptr + base_offset_O + block_offset_O + row_ptr_q * 128 + col_ptr_1 * 1
    
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