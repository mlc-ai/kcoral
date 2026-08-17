import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Provide a fast memory allocator to back the device-side storage required 
# by the Triton TMA host descriptor objects.
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

# Host execution pre-hook used by the autotuner to dynamically mutate
# configuration arguments right before compilation/launch, replacing
# generic None objects with concrete 4D TMA tensor descriptors tuned exactly
# for the selected BLOCK_M and BLOCK_N tiles.
def desc_pre_hook(kwargs):
    Q = kwargs['Q']
    K = kwargs['K']
    V = kwargs['V']
    O = kwargs['O']
    BLOCK_M = kwargs['BLOCK_M']
    BLOCK_N = kwargs['BLOCK_N']
    BLOCK_D = kwargs['BLOCK_D']
    
    # 4-dimensional TMA descriptors seamlessly handle Batch and Head indexing directly via coordinates
    kwargs['q_desc'] = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, BLOCK_D])
    kwargs['k_desc'] = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, BLOCK_D])
    kwargs['v_desc'] = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, BLOCK_D])
    kwargs['o_desc'] = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, BLOCK_D])

def get_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['S'],
    pre_hook=desc_pre_hook
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O,  # Dummy input parameters consumed exclusively by the pre-hook
    q_desc, k_desc, v_desc, o_desc,
    LSE, sm_scale_log2,
    stride_lseb, stride_lseh, stride_lses,
    S,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    # Strict boundary exit for tiles entirely outside sequence bounds
    if start_m >= S:
        return

    off_b = tl.program_id(1)
    off_h = tl.program_id(2)

    # 4D TMA Load gracefully extracting exact block slices without manual pointer arithmetic
    q_4d = q_desc.load([off_b, off_h, start_m, 0])
    q = tl.reshape(q_4d, [BLOCK_M, BLOCK_D])
    
    # Initialize Safe FlashAttention online logic metrics
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    end_m = start_m + BLOCK_M
    seq_limit = tl.minimum(S, end_m)
    num_steps = (seq_limit + BLOCK_N - 1) // BLOCK_N
    
    # Analyze the separation bound splitting where keys are provably clear of causal constraints/padding limits
    num_steps_unmasked = start_m // BLOCK_N
    num_steps_unmasked = tl.minimum(num_steps_unmasked, num_steps)

    # Loop 1: Core Pipelined Loop leveraging maximal unrolling and Hopper WGMMA parallelism inherently
    for start_n_idx in tl.range(0, num_steps_unmasked):
        start_n = start_n_idx * BLOCK_N
        k_4d = k_desc.load([off_b, off_h, start_n, 0])
        v_4d = v_desc.load([off_b, off_h, start_n, 0])
        k = tl.reshape(k_4d, [BLOCK_N, BLOCK_D])
        v = tl.reshape(v_4d, [BLOCK_N, BLOCK_D])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        # Intrinsic faster exp2 utilized over generic exp functionality
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Loop 2: Tail Tail Processing mapping causality intersections and sequence edges
    for start_n_idx in range(num_steps_unmasked, num_steps):
        start_n = start_n_idx * BLOCK_N
        k_4d = k_desc.load([off_b, off_h, start_n, 0])
        v_4d = v_desc.load([off_b, off_h, start_n, 0])
        k = tl.reshape(k_4d, [BLOCK_N, BLOCK_D])
        v = tl.reshape(v_4d, [BLOCK_N, BLOCK_D])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale_log2
        
        # Formulate and deploy explicit strict-causality cutoff
        causal_mask = (start_n + offs_n)[None, :] <= offs_m[:, None]
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Safely enforce masking limit overriding zero-padding from TMA hardware
        if start_n + BLOCK_N > S:
            valid_mask = (start_n + offs_n)[None, :] < S
            qk = tl.where(valid_mask, qk, float("-inf"))
            
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Final Execution Epilogue Normalize output mapping values correctly back into Natural Log Scale
    inv_l_i = 1.0 / l_i
    acc = acc * inv_l_i[:, None]
    
    LN_2 = 0.6931471805599453
    lse = (m_i * LN_2) + tl.log(l_i)

    # 4D TMA Export Store implicitly dismissing elements crossing sequence limits seamlessly
    acc_out = acc.to(tl.bfloat16)
    acc_4d = tl.reshape(acc_out, [1, 1, BLOCK_M, BLOCK_D])
    o_desc.store([off_b, off_h, start_m, 0], acc_4d)

    # 1D Global LSE Store utilizing localized coordinate pointer mapping
    lse_offset = off_b * stride_lseb + off_h * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    q_valid = offs_m < S
    tl.store(lse_ptrs, lse, mask=q_valid)

def run(Q, K, V, O, LSE):
    # Enforce precise target execution mapping context cleanly
    torch.cuda.set_device(Q.device)
    # Connect explicit Host TMA device memory allocation handling mechanism
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    # Advance exponent intrinsic scale shift completely into Host calculations cleanly
    LOG2_E = 1.4426950408889634
    sm_scale_log2 = sm_scale * LOG2_E

    # Intrinsic dimension placement sequentially prioritizes underlying localized L2 coherence over batches natively
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B,
        H
    )

    _fwd_kernel[grid](
        Q, K, V, O,
        None, None, None, None,
        LSE, sm_scale_log2,
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S,
        BLOCK_D=D,
    )