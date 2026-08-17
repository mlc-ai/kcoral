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
    o_desc,
    out_lse,
    S_len,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Optimized FlashAttention-Style Causal Forward Pass
    
    Grid map: Dim 0 encodes the Query Tile Step, Dim 1 encodes the flattened Batch and Head index (B*H).
    """
    step = tl.program_id(0)
    bh = tl.program_id(1)
    
    # Allocate shared memory space for the double buffered pipeline 
    shared_q = allocate_shared_array((BLOCK_M, BLOCK_N), dtype=tl.bfloat16)
    shared_k = allocate_shared_array((2, BLOCK_M, BLOCK_N), dtype=tl.bfloat16)
    shared_v = allocate_shared_array((2, BLOCK_M, BLOCK_N), dtype=tl.bfloat16)
    
    # Mandatory 128-byte alignment assertions for robust TMA execution.
    shared_q = align_shared_ptr(shared_q)
    shared_k = align_shared_ptr(shared_k)
    shared_v = align_shared_ptr(shared_v)
    
    # Prime the TMA Hardware Pipeline loops asynchronously.
    barrier_0 = tl.enq_tma_barrier()
    q_desc.load_async(shared_q, [bh * S_len + step * BLOCK_M, 0])
    k_desc.load_async(shared_k[0], [bh * S_len + 0, 0])
    v_desc.load_async(shared_v[0], [bh * S_len + 0, 0])
    
    # Fundamental running states initialized for numerically stable Online Softmax calculation.
    m_old = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    l_old = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    o_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    
    # Calculate the precise maximum number of tiles dictated by causal constraint masking properties.
    max_k_tiles = min(step + 1, triton.cdiv(S_len, BLOCK_M))
    
    tile_idx = 0
    for k_step in range(max_k_tiles):
        # Prefetch remaining pipeline stages while processing the current step.
        if k_step + 1 < max_k_tiles:
            next_tile = 1 - tile_idx
            next_k = k_step * BLOCK_M + BLOCK_M
            barrier_1 = tl.enq_tma_barrier()
            k_desc.load_async(shared_k[next_tile], [bh * S_len + next_k, 0])
            v_desc.load_async(shared_v[next_tile], [bh * S_len + next_k, 0])
            
        barrier_0.wait()
        
        # Extracted block tensors mapped directly from optimized shared memory layouts.
        q = tl.load(shared_q) 
        k = tl.load(shared_k[tile_idx])
        v = tl.load(shared_v[tile_idx])
        
        # Inner reduction accumulating over the complete block.
        zero = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        s = tl.dot(q, k.T, acc=zero)
        s *= scale
        
        # Hard boundary enforcing causal logic constraints.
        mask_q = step * BLOCK_M + rows
        mask_k = k_step * BLOCK_M + cols
        causal_mask = (mask_k[None, :] <= mask_q[:, None])
        
        s = tl.where(causal_mask, s, -1e20)
        
        m_curr = tl.maximum(m_old, tl.max(s, axis=1))
        p = tl.exp(s - m_curr[:, None])
        l_curr = l_old * tl.exp(m_old - m_curr) + tl.sum(p, axis=1)
        
        # Scale prior accumulator dynamically compensating for exponential shifts.
        exp_m_diff = tl.exp(m_old - m_curr)
        o_acc = o_acc * exp_m_diff[:, None]
        
        # Value weightings mapped to output chunks mapped perfectly over to D space.
        o_acc = tl.dot(p, v, acc=o_acc)
        
        m_old = m_curr
        l_old = l_curr
        tile_idx = 1 - tile_idx
        if k_step + 1 < max_k_tiles:
            barrier_0 = barrier_1
            
    # Finalizing calculations normalizing aggregated outputs to exact distribution.
    o_acc = (o_acc / l_old[:, None]).to(tl.bfloat16)
    
    # Direct, highly-optimized native 4D stores avoiding slow runtime indexing evaluation overhead.
    o_desc.store_async(o_acc, [bh * S_len + step * BLOCK_M, 0])
    tl.commit_store()
    tl.armed_commit_store().wait()
    
    base_offset_lse = bh * S_len + step * BLOCK_M
    ptr_lse = out_lse + base_offset_lse + rows
    row_mask_lse = (step * BLOCK_M + rows < S_len)
    
    tl.store(ptr_lse, m_old + tl.log(l_old), mask=row_mask_lse)


@triton.jit
def allocate_shared_array(shape, dtype):
    size = 1
    for dim in shape:
        size *= dim
    return tl.empty((size,), dtype=dtype, memory_space="shared")

@triton.jit
def align_shared_ptr(ptr):
    addr = ptr.to(tl.int64)
    pad = (128 - (addr % 128)) % 128
    return ptr + pad.to(tl.int32)


def run(Q, K, V, O, LSE):
    """Compute causal multi-head attention forward."""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    assert Q.shape == K.shape == V.shape
    
    scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Utilize generic rank-agnostic TensorDescriptors mapped cleanly over original tensors utilizing optimized [128, 128] blocking
    q_desc = TensorDescriptor.from_tensor(Q, [BLOCK_M, BLOCK_N])
    k_desc = TensorDescriptor.from_tensor(K, [BLOCK_M, BLOCK_N])
    v_desc = TensorDescriptor.from_tensor(V, [BLOCK_M, BLOCK_N])
    o_desc = TensorDescriptor.from_tensor(O, [BLOCK_M, BLOCK_N])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    flash_attention_causal[grid](
        q_desc, k_desc, v_desc, o_desc,
        LSE,
        S, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )