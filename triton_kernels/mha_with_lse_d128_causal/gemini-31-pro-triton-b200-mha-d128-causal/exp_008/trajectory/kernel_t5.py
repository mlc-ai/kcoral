import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _attn_fwd_kernel(
    q_desc, k_desc, v_desc, o_desc, LSE,
    stride_lseb, stride_lseh, stride_lses,
    S, SCALE,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    pid_m = tl.program_id(0)
    offset_m = pid_m * BLOCK_M
    
    # Early sequence termination limit
    if offset_m >= S:
        return

    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Natively load multidimensional memory segments directly routed via TMA
    q_4d = q_desc.load([pid_b, pid_h, offset_m, 0])
    q = tl.reshape(q_4d, (BLOCK_M, D))

    # Pre-scale Q to avoid scaling the [M, N] output tile every loop iteration.
    # The SCALE incorporates both 1/sqrt(D) and log2(e) for base-2 exponentials.
    q = (q.to(tl.float32) * SCALE).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    offs_m = offset_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Restrict the inner loop directly to the causal limit
    limit = offset_m + BLOCK_M
    max_n = S if S < limit else limit
    num_kv_tiles = (max_n + BLOCK_N - 1) // BLOCK_N

    for kv_tile in range(0, num_kv_tiles):
        offset_n = kv_tile * BLOCK_N
        
        k_4d = k_desc.load([pid_b, pid_h, offset_n, 0])
        v_4d = v_desc.load([pid_b, pid_h, offset_n, 0])
        
        k = tl.reshape(k_4d, (BLOCK_N, D))
        v = tl.reshape(v_4d, (BLOCK_N, D))

        # Dot product inherently operates on the pre-scaled scores map
        scores_b2 = tl.dot(q, k.T)
        
        # Causal mask automatically resolves bounds masking (out of bounds N evaluates implicitly to false)
        causal_mask = offs_m[:, None] >= (offset_n + offs_n)[None, :]
        scores_b2 = tl.where(causal_mask, scores_b2, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(scores_b2, axis=1))

        # Exponential computations (no safe limits needed since causal diagonal guarantees finite inputs)
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores_b2 - m_ij[:, None])

        # Core reductions step logic
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        
        m_i = m_ij

    # Epilogue standard normalizations
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l_i[:, None]

    # Convert tracked base-2 metric space mathematically back to Log-E space (Natural log)
    LN2 = 0.6931471805599453
    lse = tl.where(l_i == 0.0, float("-inf"), (m_i + tl.math.log2(safe_l_i)) * LN2)

    out_4d = tl.reshape(out.to(tl.bfloat16), (1, 1, BLOCK_M, D))
    # Native 4D boundaries handling allows implicit bounds drop out discarding the need for masking
    o_desc.store([pid_b, pid_h, offset_m, 0], out_4d)

    # Reduced masked vector metrics logic mapping 
    lse_ptr = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptr, lse, mask=(offs_m < S))


def run(Q, K, V, O, LSE):
    """
    Destination-passing entry evaluation for Causal Multi-Head Attention standard execution logic.
    Receives mapped input formats [Q, K, V] followed directly by allocated pre-defined limits [O, LSE].
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape

    # Highly tuned maximal capability logic mapped strictly targeting standard Tensor Core mapping limits
    BLOCK_M = 128
    BLOCK_N = 128

    # Multi-dimensional setup created once entirely on CPU avoiding internal CUDA malloc routines
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    o_desc = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])

    sm_scale = 1.0 / (D ** 0.5)
    SCALE = sm_scale * 1.4426950408889634

    grid = (triton.cdiv(S, BLOCK_M), B, H)
    
    _attn_fwd_kernel[grid](
        q_desc, k_desc, v_desc, o_desc, LSE,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, SCALE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D,
        num_warps=8, num_stages=2
    )