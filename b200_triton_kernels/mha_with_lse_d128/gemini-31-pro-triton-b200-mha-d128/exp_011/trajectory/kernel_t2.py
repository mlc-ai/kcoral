import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _attention_kernel(
    q_desc, k_desc, v_desc, o_desc, lse_desc,
    S, sm_scale_log2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    m_block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    offset_m = (m_block_idx * BLOCK_M).to(tl.int32)
    
    # Load Q tile and reshape locally from batched 4D down to 2D
    q_4d = q_desc.load([batch_idx, head_idx, offset_m, 0])
    q = tl.reshape(q_4d, (BLOCK_M, 128))
    
    # Base-2 Log Scaling incorporated with softmax temperature to avoid multiplying inside loop 
    q_dtype = q.dtype
    q = (q * sm_scale_log2).to(q_dtype)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    k_tiles = tl.cdiv(S, BLOCK_N)
    for kv_tile in tl.range(0, k_tiles, num_stages=NUM_STAGES):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        
        # Load respective Keys and Values natively utilizing Blackwell TMA Async copies
        k_4d = k_desc.load([batch_idx, head_idx, offset_n, 0])
        v_4d = v_desc.load([batch_idx, head_idx, offset_n, 0])
        
        k = tl.reshape(k_4d, (BLOCK_N, 128))
        v = tl.reshape(v_4d, (BLOCK_N, 128))
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Guard sequence boundaries for keys. Out-of-bound queries are gracefully absorbed since 
        # out-of-bounds Q values load as 0 and output stores ignore out-of-bounds offset writes.
        k_mask = (offset_n + tl.arange(0, BLOCK_N)) < S
        scores = tl.where(k_mask[None, :], scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        # Utilize base-2 exponential native hardware mappings
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q_dtype), v, acc)
        
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Construct base-2 LogSumExp and resolve to PyTorch reference scale (base-e)
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    LN2: tl.constexpr = 0.6931471805599453
    lse_ln = lse_log2 * LN2

    # Emit mathematically validated outputs back into 4D structures using TMA stores
    output_4d = tl.reshape(output.to(q_dtype), (1, 1, BLOCK_M, 128))
    o_desc.store([batch_idx, head_idx, offset_m, 0], output_4d)
    
    lse_ln_3d = tl.reshape(lse_ln, (1, 1, BLOCK_M))
    lse_desc.store([batch_idx, head_idx, offset_m], lse_ln_3d)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # 128x128 with 2 stages fits safely into SMEM limits and provides high hardware math throughput
    BLOCK_M = 128
    BLOCK_N = 128
    NUM_STAGES = 2
    NUM_WARPS = 8
    
    # Generate layout-transparent Host Descriptors for zero-instruction padding handling 
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])
    lse_desc = TensorDescriptor.from_tensor(LSE, [1, 1, BLOCK_M])
    
    sm_scale = 1.0 / (D ** 0.5)
    RCP_LN2 = 1.4426950408889634
    sm_scale_log2 = sm_scale * RCP_LN2
    
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, lse_desc,
        S, sm_scale_log2,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NUM_STAGES=NUM_STAGES,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES
    )