import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _attention_kernel(
    q_desc, k_desc, v_desc, o_desc, lse_desc,
    S, sm_scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    m_block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    offset_m = (m_block_idx * BLOCK_M).to(tl.int32)
    
    # Load Q using 4D TMA descriptor. 
    # The TMA engine resolves out-of-bounds automatically.
    q_4d = q_desc.load([batch_idx, head_idx, offset_m, 0])
    q = tl.reshape(q_4d, (BLOCK_M, 128))

    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    k_tiles = tl.cdiv(S, BLOCK_N)
    
    # Software-pipelined K and V streaming loop
    for kv_tile in tl.range(0, k_tiles, num_stages=NUM_STAGES):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        
        # Fetch directly into registers via TMA 4D mappings, bypassing pointer math
        k_4d = k_desc.load([batch_idx, head_idx, offset_n, 0])
        v_4d = v_desc.load([batch_idx, head_idx, offset_n, 0])
        
        k = tl.reshape(k_4d, (BLOCK_N, 128))
        v = tl.reshape(v_4d, (BLOCK_N, 128))
        
        # Native Blackwell FP16/BF16 tensor core mapping
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        scores = scores * sm_scale_log2
        
        # Mask out K sequence tails. Out-of-bounds Q computations are safely absorbed
        # since TMA output stores ignore coordinates outside the valid shape limit.
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        scores = tl.where(offs_n[None, :] < S, scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        # Execute base-2 exponentials using optimized vector units
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(scores - safe_m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)
        
        m_i = m_ij

    # Resolve normalized outputs
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Compute LSE tracking PyTorch math logic natively 
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    LN2: tl.constexpr = 0.6931471805599453
    lse_ln = lse_log2 * LN2

    # Emit output grids mapping via Native TMA stores which require no masking calculations
    output_4d = tl.reshape(output.to(q.dtype), (1, 1, BLOCK_M, 128))
    o_desc.store([batch_idx, head_idx, offset_m, 0], output_4d)
    
    lse_ln_3d = tl.reshape(lse_ln, (1, 1, BLOCK_M))
    lse_desc.store([batch_idx, head_idx, offset_m], lse_ln_3d)


def run(Q, K, V, O, LSE):
    """
    Computes a Multi-Head Attention forward pass returning Output and LogSumExp vectors.
    Accelerated with fully-TMA constrained Host Descriptors for zero-instruction memory mappings.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # 128x128 heavily triggers highly optimized tensor memory (TMEM) Blackwell structures
    BLOCK_M = 128
    BLOCK_N = 128
    NUM_STAGES = 3
    NUM_WARPS = 8
    
    # Host Descriptors bypass serial device-side allocations that cripple execution efficiency
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])
    lse_desc = TensorDescriptor.from_tensor(LSE, [1, 1, BLOCK_M])
    
    # Condense scaling constants entirely
    sm_scale = 1.0 / (D ** 0.5)
    RCP_LN2 = 1.4426950408889634
    sm_scale_log2 = sm_scale * RCP_LN2
    
    # Standard linear multi-dimensional mapping promotes L2 Hit-Rates natively 
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    _attention_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, lse_desc,
        S, sm_scale_log2,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, NUM_STAGES=NUM_STAGES,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES
    )