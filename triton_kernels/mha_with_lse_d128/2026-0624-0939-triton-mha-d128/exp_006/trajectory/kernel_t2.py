import torch
import triton
import triton.language as tl

# Required by Triton for allocating internal storage for device-side tensor descriptors.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.jit(do_not_specialize=['S', 'H', 'LSE'])
def _mha_kernel(
    Q, K, V, O, LSE,
    S, H,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Identify the exact batch/head mapping and the query sequence block this CTA processes.
    pid_m = tl.program_id(0)
    idx = tl.program_id(1)  # Encodes `b * H + h` sequentially
    
    start_n = pid_m * BLOCK_M
    rowOffsets = start_n + tl.arange(0, BLOCK_M)
    
    # Dynamically compute base pointers for the selected attention head.
    # Strides naturally follow from contiguous tensor layouts.
    q_base_ptr = Q + idx * S * 128
    k_base_ptr = K + idx * S * 128
    v_base_ptr = V + idx * S * 128
    o_base_ptr = O + idx * S * 128
    
    # Create logical 2D views directly on the GPU. 
    # Padding resolves out-of-bounds tails for any arbitrary sequence length `S`.
    desc_Q = tl.make_tensor_descriptor(
        q_base_ptr, shape=[S, 128], strides=[128, 1],
        block_shape=[BLOCK_M, 128], padding_option="zero")
    desc_K = tl.make_tensor_descriptor(
        k_base_ptr, shape=[S, 128], strides=[128, 1],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    desc_V = tl.make_tensor_descriptor(
        v_base_ptr, shape=[S, 128], strides=[128, 1],
        block_shape=[BLOCK_N, 128], padding_option="zero")
    desc_O = tl.make_tensor_descriptor(
        o_base_ptr, shape=[S, 128], strides=[128, 1],
        block_shape=[BLOCK_M, 128], padding_option="zero")
    
    # Materialize Q onto-chip; reused invariantly for every K iteration.
    Q_0 = desc_Q.load([start_n, 0])  
    
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    
    # Scaling factor matching standard FSDPA math (divide logits by sqrt(d)).
    scale = 1.0 / (128.0 ** 0.5) 
    
    num_blocks_k = tl.cdiv(S, BLOCK_N)
    
    for j in range(num_blocks_k):
        k_idx = j * BLOCK_N
        
        # Fetch next chunk of sequence keys and values.
        K_j = desc_K.load([k_idx, 0])  
        V_j = desc_V.load([k_idx, 0])
        
        # GEMM1: Evaluate Query-Key similarity matrix [BLOCK_M, BLOCK_N]
        S_tile = tl.dot(Q_0, K_j.T) * scale 
        
        m_local = tl.max(S_tile, axis=1)
        m_new = tl.maximum(m_i, m_local)
        
        exp_old = tl.exp(m_i - m_new)
        P = tl.exp(S_tile - m_new[:, None])
        
        l_local = tl.sum(P, axis=1)
        l_i = exp_old * l_i + l_local
        
        # GEMM2: Update running Output Accumulator [BLOCK_M, 128]
        O_acc = O_acc * exp_old[:, None] + tl.dot(P, V_j)
        
        m_i = m_new
    
    safe_l_i = tl.where(l_i > 0, l_i, 1.0)
    O_acc = O_acc / safe_l_i[:, None]
    
    # Mask guarantees invalid Q rows (tails unaligned with BLOCK_M) explicitly resolve to 0.
    mask_Q = rowOffsets[:, None] < S
    O_acc = tl.where(mask_Q, O_acc, 0.0)
    
    # Materialize normalized Output back into HBM resolving layout mappings natively.
    desc_O.store([start_n, 0], O_acc.to(tl.bfloat16))
    
    # Final Log-Sum-Exp resolves purely from tracked state variables.
    lse = m_i + tl.log(safe_l_i)
    
    # Direct coalesced write for scalar Layout Dimension Output.
    lse_ptr = LSE + idx * S + rowOffsets
    mask_lse = rowOffsets < S
    tl.store(lse_ptr, lse, mask=mask_lse)


def run(Q, K, V, O, LSE):
    """Compute non-causal multi-head attention forward returning O and LSE."""
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    block_m = 64
    block_n = 64
    
    # Flatten batch and head dimensions to map cleanly onto independent CTAs.
    grid = (triton.cdiv(S, block_m), B * H)
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        S, H,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4, num_stages=3,
    )