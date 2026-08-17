import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import math


@triton.jit
def _mha_kernel(
    Q_flat_ptr, K_flat_ptr, V_flat_ptr, O_ptr, LSE_ptr,
    S_len, scale, B, H, HEAD_DIM,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, NUM_STAGES: tl.constexpr
):
    """
    Non-causal multi-head attention forward pass targeting Hopper (SM90/SM90a).
    Computes Output O and Log-Sum-Exp LSE. Fixes descriptor initialization, 
    includes explicit shared memory management, and applies boundary guards.
    """
    bh_idx = tl.program_id(0)
    block_row = tl.program_id(1)
    
    if O_ptr is None or LSE_ptr is None:
        return
        
    # Request shared memory safely bounded by allocation limits
    st_P = triton.alloc_shared(BLOCK_M * BLOCK_N * 2) 
    
    desc_Q = tl.make_tensor_descriptor(
        Q_flat_ptr, 
        shape=[B * H * S_len, HEAD_DIM], 
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, BLOCK_D // 2], 
        padding_option="zero"
    )
    
    desc_K = tl.make_tensor_descriptor(
        K_flat_ptr, 
        shape=[B * H * S_len, HEAD_DIM], 
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, BLOCK_D // 2], 
        padding_option="zero"
    )
    
    desc_V = tl.make_tensor_descriptor(
        V_flat_ptr, 
        shape=[B * H * S_len, HEAD_DIM], 
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, BLOCK_D // 2], 
        padding_option="zero"
    )
    
    offset_s = bh_idx * S_len + block_row * BLOCK_M
    
    q0 = desc_Q.load([offset_s, 0])
    q1 = desc_Q.load([offset_s, BLOCK_D // 2])
    
    out0 = tl.zeros((BLOCK_M, BLOCK_D // 2), dtype=tl.float32)
    out1 = tl.zeros((BLOCK_M, BLOCK_D // 2), dtype=tl.float32)
    
    m = tl.full((BLOCK_M, 1), -float("inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    
    total_blocks = tl.cdiv(S_len, BLOCK_N)
    
    for step in range(total_blocks):
        stage = step % NUM_STAGES
        kv_offset = bh_idx * S_len + step * BLOCK_N
        
        desc_K.load_async(st_K[stage][0], [kv_offset, 0])
        desc_K.load_async(st_K[stage][1], [kv_offset, BLOCK_D // 2])
        
        desc_V.load_async(st_V[stage][0], [kv_offset, 0])
        desc_V.load_async(st_V[stage][1], [kv_offset, BLOCK_D // 2])
        
        cur_barrier = step % NUM_STAGES
        triton.commit_async_descriptor(desc_K, cur_barrier)
        triton.commit_async_descriptor(desc_V, cur_barrier)
        
        wait_barrier = step % NUM_STAGES
        triton.wait_descriptor_barrier(wait_barrier)
        
        k0 = desc_K.load([kv_offset, 0])
        k1 = desc_K.load([kv_offset, BLOCK_D // 2])
        v0 = desc_V.load([kv_offset, 0])
        v1 = desc_V.load([kv_offset, BLOCK_D // 2])
        
        s = tl.dot(q0, k0.T)
        s = tl.dot(q1, k1.T, acc=s)
        s = s * scale
        
        valid_k = ((step * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]) < S_len)
        s = tl.where(valid_k, s, -float("inf"))
        
        row_max = tl.max(s, axis=1)
        
        m_prev = m
        m_new = tl.maximum(m, row_max[None, :])
        
        exp_s = tl.exp(s - m_new)
        sum_exp = tl.sum(exp_s, axis=1)
        
        exp_m_diff = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
        for i in range(BLOCK_M):
            exp_m_diff[i] = tl.exp(m_prev[i] - m_new[i])
            
        l_new = exp_m_diff * l + sum_exp[None, :]
        
        m = m_new
        l = l_new
        
        out0 = out0 * exp_m_diff
        out1 = out1 * exp_m_diff
        
        for i in range(BLOCK_M):
            for i_ptr in range(BLOCK_N):
                st_P[i * BLOCK_N + i_ptr] = exp_s[i, i_ptr].to(tl.bfloat16)
                
        p = st_P[(BLOCK_M * BLOCK_N).to(tl.int32)].to(dtype=tl.float32)
        p = p.reshape((BLOCK_M, BLOCK_N))
        
        out0 = tl.dot(p, v0, acc=out0)
        out1 = tl.dot(p, v1, acc=out1)
        
        if step + 1 < total_blocks:
            next_stage = (step + 1) % NUM_STAGES
            next_kv = bh_idx * S_len + (step + 1) * BLOCK_N
            
            desc_K.load_async(st_K[next_stage][0], [next_kv, 0])
            desc_K.load_async(st_K[next_stage][1], [next_kv, BLOCK_D // 2])
            
            desc_V.load_async(st_V[next_stage][0], [next_kv, 0])
            desc_V.load_async(st_V[next_stage][1], [next_kv, BLOCK_D // 2])
            
            commit_barrier = (step + 1) % NUM_STAGES
            triton.commit_async_descriptor(desc_K, commit_barrier)
            triton.commit_async_descriptor(desc_V, commit_barrier)
            
    if total_blocks > 0:
        out0 = out0 / l
        out1 = out1 / l
    
    valid_rows = (block_row * BLOCK_M + tl.arange(0, BLOCK_M)) < S_len
    
    row_offsets = tl.arange(0, BLOCK_M)[:, None] * HEAD_DIM
    col_offsets_0 = tl.arange(0, BLOCK_D // 2)[None, :] * 1
    col_offsets_1 = tl.arange(0, BLOCK_D // 2)[None, :] * 1 + (BLOCK_D // 2)
    
    base_offset_O = bh_idx * S_len * HEAD_DIM
    block_offset_O = block_row * BLOCK_M * HEAD_DIM
    
    ptr_O_0 = O_ptr + base_offset_O + block_offset_O + row_offsets + col_offsets_0
    ptr_O_1 = O_ptr + base_offset_O + block_offset_O + row_offsets + col_offsets_1
    
    masked_out0 = tl.where(valid_rows[:, None], out0.to(tl.bfloat16), 0.0)
    masked_out1 = tl.where(valid_rows[:, None], out1.to(tl.bfloat16), 0.0)
    
    tl.store(ptr_O_0, masked_out0, mask=valid_rows[:, None])
    tl.store(ptr_O_1, masked_out1, mask=valid_rows[:, None])
    
    if total_blocks > 0:
        global_lse = m.squeeze(1) + tl.log(l.squeeze(1))
    else:
        global_lse = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    masked_lse = tl.where(valid_rows, global_lse, 0.0)
    
    ptr_LSE = LSE_ptr + bh_idx * S_len + block_row * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(ptr_LSE, masked_lse, mask=valid_rows)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_D = 128
    NUM_STAGES = 3
    
    Q_flat = Q.view(B * H * S, D)
    K_flat = K.view(B * H * S, D)
    V_flat = V.view(B * H * S, D)
    
    torch.backends.cudnn.deterministic = True
    
    grid = (B * H, triton.cdiv(S, BLOCK_M), 1)
    
    _mha_kernel[grid](
        Q_flat.data_ptr(), K_flat.data_ptr(), V_flat.data_ptr(), 
        O.data_ptr(), LSE.data_ptr(),
        S, scale, B, H, D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, NUM_STAGES=NUM_STAGES,
        num_warps=8, num_stages=NUM_STAGES, num_ctas=3
    )