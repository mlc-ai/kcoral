import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _fwd_kernel(
    q_desc, k_desc, v_desc, o_desc, LSE,
    stride_lseb, stride_lseh, stride_lses,
    S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch = tl.program_id(1)
    head = tl.program_id(2)

    # Early exit for fully padded query blocks beyond the sequence length
    if start_m * BLOCK_M >= S:
        return

    m_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_idx < S

    # Load Query block once (TMA pads zeros natively beyond sequence limits)
    q_4d = q_desc.load([batch, head, start_m * BLOCK_M, 0])
    q = tl.reshape(q_4d, (BLOCK_M, HEAD_DIM))

    # Incorporate the natural-to-base-2 exponent ratio into the scaling factor
    RCP_LN2: tl.constexpr = 1.4426950408889634
    scale = sm_scale * RCP_LN2

    # Running softmax components tracked precisely in FP32
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    num_full_blocks = start_m
    
    # Phase 1: Blocks definitively before the causal boundary (no sequence/causal masks needed)
    for k0 in tl.range(0, num_full_blocks, num_stages=3):
        offset_n = k0 * BLOCK_N
        k_4d = k_desc.load([batch, head, offset_n, 0])
        v_4d = v_desc.load([batch, head, offset_n, 0])
        
        k = tl.reshape(k_4d, (BLOCK_N, HEAD_DIM))
        v = tl.reshape(v_4d, (BLOCK_N, HEAD_DIM))
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32) * scale
        
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        # No `-inf` escape check needed here because elements are guaranteed to be fully valid
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(qk - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        
        m_i = m_ij

    # Phase 2: The single causal diagonal block
    offset_n = num_full_blocks * BLOCK_N
    k_4d = k_desc.load([batch, head, offset_n, 0])
    v_4d = v_desc.load([batch, head, offset_n, 0])
    
    k = tl.reshape(k_4d, (BLOCK_N, HEAD_DIM))
    v = tl.reshape(v_4d, (BLOCK_N, HEAD_DIM))
    
    qk = tl.dot(q, k.T, out_dtype=tl.float32) * scale
    
    n_idx = offset_n + tl.arange(0, BLOCK_N)
    valid_score = m_mask[:, None] & (m_idx[:, None] >= n_idx[None, :])
    qk = tl.where(valid_score, qk, -float("inf"))
    
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
    safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
    alpha = tl.math.exp2(m_i - safe_m_ij)
    p = tl.math.exp2(qk - safe_m_ij[:, None])
    
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None]
    acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
    
    m_i = m_ij

    # Finalize context vectors ensuring no NaN exceptions on fully-masked sequences
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]

    # Convert softmax components sequentially back to natural logarithm domain for LSE metric output
    LN2: tl.constexpr = 0.6931471805599453
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    lse = lse_log2 * LN2

    # Issue Native TMA descriptors stores avoiding complex masked assignments completely
    out_4d = tl.reshape(out.to(tl.bfloat16), (1, 1, BLOCK_M, HEAD_DIM))
    o_desc.store([batch, head, start_m * BLOCK_M, 0], out_4d)

    # Standardly mask and issue the linear sequence pointer LSE stores
    lse_ptrs = LSE + batch * stride_lseb + head * stride_lseh + m_idx * stride_lses
    tl.store(lse_ptrs, lse, mask=m_mask)

def run(Q, K, V, O, LSE):
    """
    Standard Triton causal multi-head attention forward computing Output and Natural-Log-Sum-Exp.
    Writes strictly in-place to preallocated 'O' and 'LSE'.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)

    BLOCK_M = 128
    BLOCK_N = 128

    # Create host tensor descriptors immediately bypassing heavily expensive dynamic GPU creation calls
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])

    # Optimal Grid indexing leverages exact inner L2 Cache mapping constraints by scheduling identical Batch/Heads proximally
    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    _fwd_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=128,
        num_warps=8, num_stages=3
    )