import torch
import triton
import triton.language as tl
from triton.language.extra._math import mbar_make, mbar_expect_arrivals, mbar_copy, mbarrier_wait
from triton.language.extra.blocks import ptr_load, store, make_block_ptr, advance_float4, load_float4
from triton.language.extra.linAlg import tma_copy


@triton.jit
def fused_softmax_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    LSE_ptr,
    S,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    scale: tl.constexpr,
):
    block_i = tl.program_id(0)
    bh_idx = tl.program_id(1)
    
    offset_m = block_i * BLOCK_M
    
    if offset_m >= S:
        return
    
    q_buf = align_ptr(extern_shared_array((BLOCK_M * HEAD_DIM * 2 + 128,), itemsize=1))
    k_buf = align_ptr(extern_shared_array((NUM_STAGES * BLOCK_N * HEAD_DIM * 2 + 128,), itemsize=1))
    v_buf = align_ptr(extern_shared_array((NUM_STAGES * BLOCK_N * HEAD_DIM * 2 + 128,), itemsize=1))
    
    mbar_q = mbar_make()
    mbar_k = mbar_make()
    mbar_v = mbar_make()
    
    q_bar = [mbar_make()]
    k_bar = [mbar_make(), mbar_make()]
    v_bar = [mbar_make(), mbar_make()]
    
    p_q = make_block_ptr(base=q_buf, shape=[BLOCK_M, HEAD_DIM], strides=[HEAD_DIM, 1], element_type=tl.bfloat16, boundary_check=())
    p_k = make_block_ptr(base=k_buf, shape=[NUM_STAGES, BLOCK_N, HEAD_DIM], strides=[BLOCK_N * HEAD_DIM, HEAD_DIM, 1], element_type=tl.bfloat16, boundary_check=())
    p_v = make_block_ptr(base=v_buf, shape=[NUM_STAGES, BLOCK_N, HEAD_DIM], strides=[BLOCK_N * HEAD_DIM, HEAD_DIM, 1], element_type=tl.bfloat16, boundary_check=())
    
    acc_o = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    num_kv_blocks = tl.cdiv(S, BLOCK_N)
    max_j = min(num_kv_blocks - 1, block_i * BLOCK_M // BLOCK_N)
    
    q_ptr = Q_ptr + bh_idx * S * HEAD_DIM + offset_m * HEAD_DIM
    tma_copy.async_cg(q_buf, q_ptr, [1, BLOCK_M, HEAD_DIM])
    mbar_expect_arrivals(mbar_q, 1)
    mbar_copy.async_cg(q_bar[0], mbar_q, 1)
    
    if max_j >= 0:
        k_ptr = K_ptr + bh_idx * S * HEAD_DIM
        v_ptr = V_ptr + bh_idx * S * HEAD_DIM
        tma_copy.async_cg(k_buf, k_ptr, [1, BLOCK_N, HEAD_DIM])
        mbar_expect_arrivals(mbar_k, 1)
        mbar_copy.async_cg(k_bar[0], mbar_k, 1)
        
        tma_copy.async_cg(v_buf, v_ptr, [1, BLOCK_N, HEAD_DIM])
        mbar_expect_arrivals(mbar_v, 1)
        mbar_copy.async_cg(v_bar[0], mbar_v, 1)
    
    for j in range(0, max_j + 1):
        stage = j % 2
        
        mbarrier_wait(k_bar[stage])
        mbarrier_wait(v_bar[stage])
        
        if j < max_j:
            next_stage = (j + 1) % 2
            k_ptr = K_ptr + bh_idx * S * HEAD_DIM + ((j + 1) * BLOCK_N * HEAD_DIM)
            v_ptr = V_ptr + bh_idx * S * HEAD_DIM + ((j + 1) * BLOCK_N * HEAD_DIM)
            
            tma_copy.async_cg(k_buf + next_stage * BLOCK_N * HEAD_DIM * 2, k_ptr, [1, BLOCK_N, HEAD_DIM])
            mbar_expect_arrivals(mbar_k, 1)
            mbar_copy.async_cg(k_bar[next_stage], mbar_k, 1)
            
            tma_copy.async_cg(v_buf + next_stage * BLOCK_N * HEAD_DIM * 2, v_ptr, [1, BLOCK_N, HEAD_DIM])
            mbar_expect_arrivals(mbar_v, 1)
            mbar_copy.async_cg(v_bar[next_stage], mbar_v, 1)
            
        mbarrier_wait(q_bar[stage])
        
        q_shared = read_from_smem(p_q, 0, BLOCK_M, HEAD_DIM, q_buf)
        k_shared = read_from_smem(p_k, stage * BLOCK_N * HEAD_DIM * 2, BLOCK_N, HEAD_DIM, k_buf)
        v_shared = read_from_smem(p_v, stage * BLOCK_N * HEAD_DIM * 2, BLOCK_N, HEAD_DIM, v_buf)
        
        acc_s = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in range(0, HEAD_DIM // 4):
            q_chunk = q_shared[i*4:(i+1)*4]
            k_chunk = k_shared[i*4:(i+1)*4]
            acc_s += tl.dot(q_chunk, k_chunk.T)
            
        acc_s = acc_s * scale
        
        n_indices = j * BLOCK_N + tl.arange(0, BLOCK_N)
        q_indices = offset_m + tl.arange(0, BLOCK_M)
        mask = (n_indices[None, :] <= q_indices[:, None]) & (n_indices[None, :] < S)
        
        acc_s = tl.where(mask, acc_s, -1e20)
        
        m_old = m
        m_row = tl.reduce(acc_s, 1, tl.maximum)
        m_new = tl.maximum(m_old, m_row)
        
        exp_s = tl.exp(acc_s - m_new)
        exp_s = tl.where(mask, exp_s, 0.0)
        
        l_row = tl.reduce(exp_s, 1, tl.sum)
        l_new = l * tl.exp(m_old - m_new) + l_row
        
        o_scale = tl.exp(m_old - m_new)
        acc_o = acc_o * o_scale[:, None]
        
        for i in range(0, BLOCK_N // 4):
            e_chunk = exp_s[:, i*4:(i+1)*4]
            v_chunk = v_shared[i*4:(i+1)*4]
            acc_o += tl.dot(e_chunk, v_chunk)
            
        m = m_new
        l = l_new
        
    mbar_expect_arrivals(mbar_q, 0)
    mbar_expect_arrivals(mbar_k, 0)
    mbar_expect_arrivals(mbar_v, 0)
    
    safe_l = tl.where(l > 0, l, 1.0)
    acc_o = acc_o / safe_l[:, None]
    
    lse = m + tl.log(l)
    
    acc_o = acc_o.to(tl.bfloat16)
    
    query_mask = (offset_m + tl.arange(0, BLOCK_M)) < S
    o_ptr = O_ptr + (bh_idx * S + offset_m) * HEAD_DIM
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, HEAD_DIM)
    ptrs = o_ptr + rows[:, None] * HEAD_DIM + cols[None, :]
    tl.store(ptrs, acc_o, mask=query_mask[:, None])
    
    lse_ptr = LSE_ptr + bh_idx * S + offset_m
    tl.store(lse_ptr + tl.arange(0, BLOCK_M), lse, mask=query_mask)


def align_ptr(ptr):
    return ptr + (16 - (ptr % 16)) % 16


def read_from_smem(smem_ptr, offset_bytes, rows, stride, smem_base):
    ptr = make_block_ptr(base=smem_base + smem_ptr + offset_bytes, shape=[rows, stride], strides=[stride, 1], element_type=tl.bfloat16, boundary_check=())
    vals = []
    for i in range(0, rows * stride // 4):
        vals.append(load_float4(ptr))
        advance_float4(ptr)
    return vals


@triton.jit
def extern_shared_array(shape, itemsize):
    return ptr_load(0, dtype=tl.pointer)

NUM_STAGES = 2


def run(Q, K, V, O, LSE):
    """Compute Causal Multi-Head Attention forward pass returning O and LSE."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    scale = 1.0 / tl.sqrt(tl.float32(D))
    
    grid = (triton.cdiv(S, 128), B * H)
    
    fused_softmax_kernel[grid](
        Q, K, V, O, LSE, S,
        HEAD_DIM=128,
        BLOCK_M=128,
        BLOCK_N=64,
        scale=scale,
        num_warps=4,
    )