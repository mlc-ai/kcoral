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
    out_o,
    out_lse,
    S_len,
    sqrt_D,
    HEAD_DIM: tl.constexpr,
):
    """
    Optimized FlashAttention-Style Causal Forward Pass
    
    Grid map: Dim 0 encodes the Query Tile Step (rows 0-127, 128-255, etc.), 
              Dim 1 encodes the flattened Batch and Head index combined (B*H).
    """
    assert HEAD_DIM == 128
    
    step = tl.program_id(0)
    bh = tl.program_id(1)
    
    # Allocate shared memory space for the double buffered pipeline 
    # and the persistent Q tile across the entire KV reduction sequence.
    shared_q = allocate_shared_array((128, 128), tl.bfloat16)
    shared_k = allocate_shared_array((2, 128, 128), tl.bfloat16)
    shared_v = allocate_shared_array((2, 128, 128), tl.bfloat16)
    
    # Mandatory 128-byte alignment assertions for robust TMA execution.
    assert_aligned(shared_q)
    assert_aligned(shared_k)
    assert_aligned(shared_v)
    
    # Base offsets for seamless 4-Dimensional memory navigation mapping.
    start_q = bh * S_len + step * 128
    
    # Prime the TMA Hardware Pipeline loops asynchronously.
    q_desc.load_async(shared_q, [start_q, 0])
    k_desc.load_async(shared_k[0], [bh * S_len + 0, 0])
    v_desc.load_async(shared_v[0], [bh * S_len + 0, 0])
    
    barrier = tl.enq_tma_barrier()
    barrier.wait()
    
    # Extracted block tensors mapped directly from optimized shared memory layouts.
    q = tl.load(shared_q) 
    
    # Fundamental running states initialized for numerically stable Online Softmax calculation.
    m_old = tl.full((128,), -1e20, dtype=tl.float32)
    l_old = tl.full((128,), 0.0, dtype=tl.float32)
    o_acc_0 = tl.zeros((128, 64), dtype=tl.float32)
    o_acc_1 = tl.zeros((128, 64), dtype=tl.float32)
    
    # Precomputed coordinate tensors for accurate masking limits.
    rows = tl.arange(0, 128)
    cols = tl.arange(0, 128)
    
    tile_idx = 0
    # Limit search span natively by causal constraint masking properties.
    for k_step in range(step + 1):
        barrier.wait()
        
        # Prefetch remaining pipeline stages while processing the current step.
        if k_step + 1 <= step:
            next_tile = 1 - tile_idx
            next_k = k_step * 128 + (1 - tile_idx) * 128
            barrier = tl.enq_tma_barrier()
            k_desc.load_async(shared_k[next_tile], [bh * S_len + next_k, 0])
            v_desc.load_async(shared_v[next_tile], [bh * S_len + next_k, 0])
            
        k_loaded = tl.load(shared_k[tile_idx, :])
        v_loaded = tl.load(shared_v[tile_idx, :])
        
        # Inner reduction accumulating over the complete block (D = 128).
        zero = tl.zeros((128, 128), dtype=tl.float32)
        s = zero
        for j in range(2):
            chunk_q = tl.split(q)[j]
            chunk_k = tl.split(k_loaded)[j]
            s = tl.dot(chunk_q, chunk_k.T, acc=s)
            
        s = s * sqrt_D
        
        # Hard boundary enforcing causal logic constraints.
        mask_k = k_step * 128 + cols
        mask_q = step * 128 + rows
        causal_mask = (mask_k <= mask_q)
        s = s * causal_mask
        
        m_curr = tl.maximum(tl.reshape(m_old, [-1]), tl.max(s, axis=1))
        p = tl.exp(s - m_curr[:, None])
        p = p * causal_mask
        l_curr = l_old * tl.exp(m_old - m_curr) + tl.sum(p, axis=1)
        
        # Scale prior accumulator dynamically compensating for exponential shifts.
        exp_m_diff = tl.exp(m_old - m_curr)
        o_acc_0 = o_acc_0 * exp_m_diff[:, None]
        o_acc_1 = o_acc_1 * exp_m_diff[:, None]
        
        # Value weightings mapped to output chunks mapped perfectly over to D space.
        p = p.to(tl.bfloat16)
        for j in range(2):
            chunk_v = tl.split(v_loaded)[j]
            if j == 0:
                o_acc_0 = tl.dot(p, chunk_v, acc=o_acc_0)
            else:
                o_acc_1 = tl.dot(p, chunk_v, acc=o_acc_1)
                
        m_old = m_curr
        l_old = l_curr
        tile_idx = 1 - tile_idx
        
    # Finalizing calculations normalizing aggregated outputs to exact distribution.
    o_acc_0 = (o_acc_0 / l_old[:, None]).to(tl.bfloat16)
    o_acc_1 = (o_acc_1 / l_old[:, None]).to(tl.bfloat16)
    
    # Direct, highly-optimized native 4D stores avoiding slow runtime indexing evaluation overhead.
    row_offsets = rows * 128
    base_offset = bh * S_len * 128 + step * 128 * 128
    
    ptr_0 = out_o + base_offset + row_offsets[:, None] + tl.arange(0, 64)[None, :]
    ptr_1 = out_o + base_offset + row_offsets[:, None] + (64 + tl.arange(0, 64))[None, :]
    
    row_mask = (rows < (S_len - step * 128))[:, None]
    tl.store(ptr_0, o_acc_0, mask=row_mask)
    tl.store(ptr_1, o_acc_1, mask=row_mask)
    
    base_offset_lse = bh * S_len + step * 128
    ptr_lse = out_lse + base_offset_lse + rows
    row_mask_lse = (rows < (S_len - step * 128))
    tl.store(ptr_lse, m_old + tl.log(l_old), mask=row_mask_lse)


@triton.jit
def allocate_shared_array(shape, dtype):
    return tl.empty(shape, dtype=dtype, memory_space="shared")

@triton.jit
def assert_aligned(ptr):
    addr = ptr.to(tl.int64)
    assert (addr % 128 == 0), "Shared memory pointer must be 128-byte aligned for TMA"


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention forward."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    assert HEAD_DIM == D
    assert Q.shape == K.shape == V.shape
    
    sqrt_D = 1.0 / math.sqrt(D)
    
    # Utilize generic rank-agnostic TensorDescriptors mapped cleanly over original tensors
    q_desc = TensorDescriptor.from_tensor(Q, [128, 128], padding_option="zero")
    k_desc = TensorDescriptor.from_tensor(K, [128, 128], padding_option="zero")
    v_desc = TensorDescriptor.from_tensor(V, [128, 128], padding_option="zero")
    
    grid = (triton.cdiv(S, 128), B * H)
    
    flash_attention_causal[grid](
        q_desc, k_desc, v_desc,
        O, LSE,
        S, sqrt_D,
        HEAD_DIM=D,
        num_warps=4,
        num_stages=4,
    )