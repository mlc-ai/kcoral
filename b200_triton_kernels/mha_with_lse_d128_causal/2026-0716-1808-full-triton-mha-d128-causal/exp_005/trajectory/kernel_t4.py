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
    H,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Optimized FlashAttention-Style Causal Forward Pass
    
    Grid map: Dim 0 encodes the Query Tile Step, Dim 1 encodes the flattened Batch and Head index (B*H).
    """
    step = tl.program_id(0)
    b_h = tl.program_id(1)
    
    # Explicit deconstruction from flattened index to navigate natively in 4D space.
    b = b_h // H
    h = b_h % H
    
    # Allocate shared memory space for the 6-stage double buffered pipeline 
    size_Q = BLOCK_M * HEAD_DIM * 2 + 128
    size_KV = 2 * BLOCK_M * HEAD_DIM * 2 + 128
    
    s_Q = allocate_shared_array(size_Q, dtype=tl.int8)
    s_K = allocate_shared_array(size_KV, dtype=tl.int8)
    s_V = allocate_shared_array(size_KV, dtype=tl.int8)
    
    # Mandatory 128-byte alignment assertions for robust TMA execution.
    s_Q = align_shared_ptr(s_Q)
    s_K = align_shared_ptr(s_K)
    s_V = align_shared_ptr(s_V)
    
    # Initialize 6-stage pipeline barriers. Expect 2 arrivals (K and V loads) per stage.
    mbarriers = [create_mbarrier(2, space="shared") for _ in range(6)]
    q_barrier = create_mbarrier(1, space="shared")
    
    # Calculate the precise maximum number of tiles dictated by causal constraint masking properties.
    num_kv_steps = min(step + 1, tl.cdiv(S_len, BLOCK_M))
    
    # Prime the TMA Hardware Pipeline loops asynchronously.
    q_desc.load_async(s_Q, [b, h, step * BLOCK_M, 0], barrier=q_barrier)
    
    # Proactively issue the first min(6, num_kv_steps) pipeline stages.
    for s in range(min(6, num_kv_steps)):
        k_desc.load_async(s_K + (s % 2) * (BLOCK_M * HEAD_DIM * 2), [b, h, s * BLOCK_M, 0], barrier=mbarriers[s])
        v_desc.load_async(s_V + (s % 2) * (BLOCK_M * HEAD_DIM * 2), [b, h, s * BLOCK_M, 0], barrier=mbarriers[s])
        
    mbarrier_wait(q_barrier)
    
    # Fundamental running states initialized for numerically stable Online Softmax calculation.
    m_old = tl.full((BLOCK_M,), -1e20, dtype=tl.float32)
    l_old = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    o_acc_0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    o_acc_1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    rows = tl.arange(0, BLOCK_M)
    
    for i in range(num_kv_steps):
        # Prefetch step i + 6 asynchronously to maximize pipeline depth.
        if i + 6 < num_kv_steps:
            mbarrier_expect_counter(mbarriers[i % 6], 2)
            k_desc.load_async(s_K + ((i + 6) % 2) * (BLOCK_M * HEAD_DIM * 2), [b, h, (i + 6) * BLOCK_M, 0], barrier=mbarriers[i % 6])
            v_desc.load_async(s_V + ((i + 6) % 2) * (BLOCK_M * HEAD_DIM * 2), [b, h, (i + 6) * BLOCK_M, 0], barrier=mbarriers[i % 6])
            
        mbarrier_wait(mbarriers[i % 6])
        
        # Extracted block tensors mapped directly from optimized shared memory layouts.
        q = s_Q.view(tl.bfloat16).reshape((BLOCK_M, HEAD_DIM))
        k = (s_K + (i % 2) * (BLOCK_M * HEAD_DIM * 2)).view(tl.bfloat16).reshape((BLOCK_M, HEAD_DIM))
        v = (s_V + (i % 2) * (BLOCK_M * HEAD_DIM * 2)).view(tl.bfloat16).reshape((BLOCK_M, HEAD_DIM))
        
        # Inner reduction accumulating over the complete block.
        s = tl.dot(q, k.T)
        s *= scale
        
        # Hard boundary enforcing causal logic constraints.
        mask_q = step * BLOCK_M + rows[:, None]
        mask_k = i * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
        causal_mask = mask_k <= mask_q
        
        s = tl.where(causal_mask, s, -1e20)
        
        m_curr = tl.maximum(m_old, tl.max(s, axis=1))
        p = tl.exp(s - m_curr[:, None])
        p = tl.where(causal_mask, p, 0.0)
        l_curr = l_old * tl.exp(m_old - m_curr) + tl.sum(p, axis=1)
        
        # Scale prior accumulator dynamically compensating for exponential shifts.
        exp_m_diff = tl.exp(m_old - m_curr)
        o_acc_0 *= exp_m_diff[:, None]
        o_acc_1 *= exp_m_diff[:, None]
        
        # Value weightings mapped perfectly over to native D=128 output space.
        v0, v1 = tl.split(v)
        o_acc_0 = tl.dot(p, v0, acc=o_acc_0)
        o_acc_1 = tl.dot(p, v1, acc=o_acc_1)
        
        m_old = m_curr
        l_old = l_curr
        
    # Finalizing calculations normalizing aggregated outputs to exact distribution.
    o_acc_0 = (o_acc_0 / l_old[:, None]).to(tl.bfloat16)
    o_acc_1 = (o_acc_1 / l_old[:, None]).to(tl.bfloat16)
    
    # Direct, highly-optimized native 4D stores avoiding slow runtime indexing evaluation overhead.
    o_desc.store_async(o_acc_0, [b, h, step * BLOCK_M, 0])
    o_desc.store_async(o_acc_1, [b, h, step * BLOCK_M, 64])
    tl.commit_store()
    
    base_offset_lse = b_h * S_len + step * BLOCK_M
    ptr_lse = out_lse + base_offset_lse + rows
    row_mask_lse = (step * BLOCK_M + rows < S_len)
    
    tl.store(ptr_lse, m_old + tl.log(l_old), mask=row_mask_lse)


@triton.jit
def allocate_shared_array(size, dtype):
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
    HEAD_DIM = 128
    
    # Utilize generic rank-matched 4D TensorDescriptors mapped cleanly over original tensors utilizing optimized blocking
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, BLOCK_N])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_M, BLOCK_N])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_M, BLOCK_N])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, 64])
    
    grid = (triton.cdiv(S, BLOCK_M), B * H)
    
    flash_attention_causal[grid](
        q_desc, k_desc, v_desc, o_desc,
        LSE,
        S, scale, H,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
        num_warps=8,
        num_stages=2,
    )