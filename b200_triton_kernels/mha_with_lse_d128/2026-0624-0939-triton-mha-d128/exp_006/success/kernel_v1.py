import torch
import triton
import triton.language as tl

# Required by Triton for allocating internal storage for device-side tensor descriptors.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


@triton.jit
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
    q_base_ptr = Q + idx * S * 128
    k_base_ptr = K + idx * S * 128
    v_base_ptr = V + idx * S * 128
    o_base_ptr = O + idx * S * 128
    
    # Create logical 2D views directly on the GPU. 
    # Padding resolves out-of-bounds tails for any arbitrary sequence length `S`.
    desc_Q = tl.make_tensor_descriptor(
        q_base_ptr, shape=[S, 128], strides=[128, 1],
        block_shape=[BLOCK_M, 64], padding_option="zero")
    desc_K = tl.make_tensor_descriptor(
        k_base_ptr, shape=[S, 128], strides=[128, 1],
        block_shape=[BLOCK_N, 64], padding_option="zero")
    desc_V = tl.make_tensor_descriptor(
        v_base_ptr, shape=[S, 128], strides=[128, 1],
        block_shape=[BLOCK_N, 64], padding_option="zero")
    desc_O = tl.make_tensor_descriptor(
        o_base_ptr, shape=[S, 128], strides=[128, 1],
        block_shape=[BLOCK_M, 64], padding_option="zero")
    
    # Materialize Q onto-chip; reused invariantly for every K iteration.
    Q_0_0 = desc_Q.load([start_n, 0]).to(tl.float32)
    Q_0_1 = desc_Q.load([start_n, 64]).to(tl.float32)  
    
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    O_acc_0 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    O_acc_1 = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
    
    # Scaling factor matching standard FSDPA math (divide logits by sqrt(d)).
    scale = 1.0 / (128.0 ** 0.5) 
    
    num_blocks_k = tl.cdiv(S, BLOCK_N)
    
    for j in range(num_blocks_k):
        k_idx = j * BLOCK_N
        
        # Fetch next chunk of sequence keys and values.
        K_j_0 = desc_K.load([k_idx, 0])
        K_j_1 = desc_K.load([k_idx, 64])
        V_j_0 = desc_V.load([k_idx, 0])
        V_j_1 = desc_V.load([k_idx, 64])
        
        K_j_0_f32 = K_j_0.to(tl.float32)
        K_j_1_f32 = K_j_1.to(tl.float32)
        V_j_0_f32 = V_j_0.to(tl.float32)
        V_j_1_f32 = V_j_1.to(tl.float32)
        
        # GEMM1: Evaluate Query-Key similarity matrix [BLOCK_M, BLOCK_N]
        S_tile = (tl.dot(Q_0_0, K_j_0_f32.T) + tl.dot(Q_0_1, K_j_1_f32.T)) * scale 
        
        # Explicitly nullify attention scores towards out-of-bounds key positions
        k_offsets = k_idx + tl.arange(0, BLOCK_N)
        valid_k = k_offsets < S
        S_tile = tl.where(valid_k[None, :], S_tile, -float('inf'))
        
        m_local = tl.max(S_tile, axis=1)
        m_new = tl.maximum(m_i, m_local)
        
        exp_old = tl.exp(m_i - m_new)
        exp_old = tl.where(m_i == -float('inf'), 1.0, exp_old)
        
        P = tl.exp(S_tile - m_new[:, None])
        
        l_local = tl.sum(P, axis=1)
        l_i = exp_old * l_i + l_local
        
        # GEMM2: Update running Output Accumulator divided cleanly over D=128 boundaries
        O_acc_0 = O_acc_0 * exp_old[:, None] + tl.dot(P, V_j_0_f32)
        O_acc_1 = O_acc_1 * exp_old[:, None] + tl.dot(P, V_j_1_f32)
        
        m_i = m_new
    
    safe_l_i = tl.where(l_i > 0, l_i, 1.0)
    O_acc_0 = O_acc_0 / safe_l_i[:, None]
    O_acc_1 = O_acc_1 / safe_l_i[:, None]
    
    # Mask guarantees invalid Q rows (tails unaligned with BLOCK_M) explicitly resolve to 0.
    mask_Q = rowOffsets[:, None] < S
    O_acc_0 = tl.where(mask_Q, O_acc_0, 0.0)
    O_acc_1 = tl.where(mask_Q, O_acc_1, 0.0)
    
    # Materialize normalized Output back into HBM resolving layout mappings natively.
    desc_O.store([start_n, 0], O_acc_0.to(tl.bfloat16))
    desc_O.store([start_n, 64], O_acc_1.to(tl.bfloat16))
    
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
    
    block_m = 128
    block_n = 64
    
    # Flatten batch and head dimensions to map cleanly onto independent CTAs.
    grid = (triton.cdiv(S, block_m), B * H)
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        S, H,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=8, num_stages=4,
    )