import torch
import triton
import triton.language as tl


@triton.jit
def _mha_kernel(
    Q_ptr, 
    K_ptr, 
    V_ptr, 
    O_ptr, 
    LSE_ptr, 
    H, 
    S, 
    D, 
    BLOCK_Q: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    b_idx = tl.program_id(2)
    h_idx = tl.program_id(1)
    i = tl.program_id(0)
    
    offset_q = i * BLOCK_Q
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
    
    # Create on-device tensor descriptors to dynamically target the correct (b, h) slice
    base_offset_b = b_idx * H * S * D + h_idx * S * D
    q_desc = tl.make_tensor_descriptor(Q_ptr + base_offset_b, [S, D], [D, 1], [BLOCK_Q, D], "zero")
    k_desc = tl.make_tensor_descriptor(K_ptr + base_offset_b, [S, D], [D, 1], [BLOCK_K, D], "zero")
    v_desc = tl.make_tensor_descriptor(V_ptr + base_offset_b, [S, D], [D, 1], [BLOCK_K, D], "zero")
    o_desc = tl.make_tensor_descriptor(O_ptr + base_offset_b, [S, D], [D, 1], [BLOCK_Q, D])
    
    # Load Query tile [BLOCK_Q, D]
    Q_i = q_desc.load([offset_q, 0])
    
    # Initialize Sequential Softmax state
    O = tl.zeros((BLOCK_Q, D), tl.float32)
    m = tl.full((BLOCK_Q, 1), -float('inf'), tl.float32)
    l = tl.full((BLOCK_Q, 1), 0.0, tl.float32)
    
    global_q = offset_q + tl.arange(0, BLOCK_Q)
    
    # Iterate exclusively over valid lower-triangular causal KV tiles
    for j in range(i + 1):
        offset_k = j * BLOCK_K
        K_j = k_desc.load([offset_k, 0])
        V_j = v_desc.load([offset_k, 0])
        
        S_att = tl.dot(Q_i, K_j.T) * scale 
        
        global_k = offset_k + tl.arange(0, BLOCK_K)
        valid = (global_q[:, None] >= global_k[None, :]) & \
                (global_q[:, None] < S) & \
                (global_k[None, :] < S)
        S_att = tl.where(valid, S_att, -float('inf'))
        
        m_prev = m
        m = tl.maximum(m, tl.max(S_att, axis=1, keepdims=True))
        P = tl.exp(S_att - m)
        
        l = l * tl.exp(m_prev - m) + tl.sum(P, axis=1, keepdims=True)
        O = O * tl.exp(m_prev - m) + tl.dot(P, V_j)
        
    # Final normalization safe guard
    valid_q = (global_q < S)[:, None]
    O = tl.where(valid_q, O / l, tl.zeros((BLOCK_Q, D), tl.float32))
    
    # Capture exact LSE magnitude mapping naturally matching sequential reduction definition metrics
    m_squeezed = m.squeeze(1)
    l_squeezed = l.squeeze(1)
    lse = m_squeezed + tl.log(l_squeezed)
    
    # Materialize updated memory regions atomically consistent with descriptor boundaries
    o_desc.store([offset_q, 0], O.to(tl.bfloat16))
    
    q_indices_abs = i * BLOCK_Q + tl.arange(0, BLOCK_Q)
    lse_ptr = LSE_ptr + b_idx * H * S + h_idx * S + q_indices_abs
    tl.store(lse_ptr, lse, mask=(q_indices_abs < S))


def run(Q, K, V, O, LSE):
    """Execute highly optimized causal multi-head attention computation returning precise outputs and accurate LSE"""
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Establish specialized contiguous allocation protocol strictly for internal TMA descriptor structures
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    num_blocks_q = triton.cdiv(S, 64)
    grid = (num_blocks_q, H, B)
    
    # Launch optimized SM90/SM90a native kernel targeting maximum theoretical throughput 
    _mha_kernel[grid](
        Q, K, V, O, LSE,
        H, S, D,
        BLOCK_Q=64, BLOCK_K=64,
        num_warps=4, num_stages=4,
    )