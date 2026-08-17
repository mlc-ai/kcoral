import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Boilerplate TMEM requirement for Triton compilation stability on descriptors
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)


# Instantiates Host 4D TMA Load/Store Descriptors explicitly bypassing device-side creation bounds overhead 
def pre_hook(kwargs):
    Q = kwargs["Q"]
    K = kwargs["K"]
    V = kwargs["V"]
    O = kwargs["O"]
    BLOCK_M = kwargs["BLOCK_M"]
    BLOCK_N = kwargs["BLOCK_N"]
    D = kwargs["D"]
    
    kwargs["q_desc"] = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    kwargs["k_desc"] = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    kwargs["v_desc"] = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    kwargs["o_desc"] = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'STAGE_LOOP': 3}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'STAGE_LOOP': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'STAGE_LOOP': 3}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'STAGE_LOOP': 4}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'STAGE_LOOP': 3}, num_warps=4, num_stages=4),
    ],
    key=['S'],
    pre_hook=pre_hook
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale, O, LSE,
    q_desc, k_desc, v_desc, o_desc,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE_LOOP: tl.constexpr
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(S, BLOCK_M)
    
    # Intentionally swizzle grid assignments tracking 'M' limits fastest, massively spiking L2 cache pooling
    pid_m = pid % num_m
    pid_bh = pid // num_m
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    q_start = pid_m * BLOCK_M
    if q_start >= S:
        return
        
    q_4d = q_desc.load([pid_b, pid_h, q_start, 0])
    q = tl.reshape(q_4d, [BLOCK_M, D])
    
    # Scale query precisely by logarithm bounds strictly before matrix streamings to eliminate repetitive instructions 
    RCP_LN2 = 1.4426950408889634
    sm_scale_log2 = sm_scale * RCP_LN2
    q = (q * sm_scale_log2).to(q.dtype)
    
    m_i = tl.full((BLOCK_M,), -50000.0, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    
    # Calculate fully dense step ranges skipping casual overlaps altogether
    n_full_steps = q_start // BLOCK_N
    
    # Phase 1: Pure Mathematical Unrolled Pipeline (Zero overhead conditionals or boundary masks evaluated here)
    for k_step in tl.range(0, n_full_steps, num_stages=STAGE_LOOP):
        k_start = k_step * BLOCK_N
        
        k_4d = k_desc.load([pid_b, pid_h, k_start, 0])
        v_4d = v_desc.load([pid_b, pid_h, k_start, 0])
        
        k = tl.reshape(k_4d, [BLOCK_N, D])
        v = tl.reshape(v_4d, [BLOCK_N, D])
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc=acc, out_dtype=tl.float32)
        m_i = m_ij
        
    k_max = tl.minimum(S, q_start + BLOCK_M)
    n_total_steps = tl.cdiv(k_max, BLOCK_N)
    
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    q_idx = offs_m[:, None]
    
    # Phase 2: Causal Masking Path evaluating only necessary upper triangular offsets
    for k_step in range(n_full_steps, n_total_steps):
        k_start = k_step * BLOCK_N
        
        k_4d = k_desc.load([pid_b, pid_h, k_start, 0])
        v_4d = v_desc.load([pid_b, pid_h, k_start, 0])
        
        k = tl.reshape(k_4d, [BLOCK_N, D])
        v = tl.reshape(v_4d, [BLOCK_N, D])
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        k_idx = k_start + offs_n[None, :]
        valid_score = q_idx >= k_idx
        scores = tl.where(valid_score, scores, -50000.0)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc=acc, out_dtype=tl.float32)
        m_i = m_ij
        
    # Standard inverse reciprocal avoiding elementwise divisions bottleneck
    inv_l_i = 1.0 / l_i
    out = acc * inv_l_i[:, None]
    
    LN2 = 0.6931471805599453
    lse = (m_i + tl.math.log2(l_i)) * LN2
    
    # Store directly dropping irrelevant elements seamlessly out-of-bounds due to TMA metadata handling rules
    out_4d = tl.reshape(out.to(q.dtype), [1, 1, BLOCK_M, D])
    o_desc.store([pid_b, pid_h, q_start, 0], out_4d)
    
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    lse_mask = offs_m < S
    tl.store(lse_ptrs, lse, mask=lse_mask)


def run(Q, K, V, O, LSE):
    """
    Computes optimal native causal Multi-Head Attention forward via destination-passing.
    Q, K, V, O have shape (B, H, S, D) and type bfloat16.
    LSE has shape (B, H, S) and type float32.
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    # Swizzles flat grid cleanly ensuring shared M loops execute natively atop continuous cache footprints
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]) * B * H,)
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale, O, LSE,
        None, None, None, None, # Auto-Injected flawlessly by pre_hook interceptor
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        D=D
    )