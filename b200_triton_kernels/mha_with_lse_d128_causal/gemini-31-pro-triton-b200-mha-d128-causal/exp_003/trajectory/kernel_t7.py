import torch
import triton
import triton.language as tl
import math
from triton.tools.tensor_descriptor import TensorDescriptor

def desc_pre_hook(kwargs):
    """
    Hook to dynamically create host-side TMA TensorDescriptors for the currently
    tuned block configuration, avoiding slow device-side descriptor allocations.
    """
    Q = kwargs["Q"]
    K = kwargs["K"]
    V = kwargs["V"]
    O = kwargs["O"]
    BLOCK_M = kwargs["BLOCK_M"]
    BLOCK_N = kwargs["BLOCK_N"]
    D = kwargs["D"]
    
    # 4D descriptors safely cover full bounds and natively discard padded outer accesses
    kwargs["q_desc"] = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, D])
    kwargs["k_desc"] = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, D])
    kwargs["v_desc"] = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, D])
    kwargs["o_desc"] = TensorDescriptor.from_tensor(O, [1, 1, BLOCK_M, D])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=4, num_stages=4),
    ],
    key=["S"],
    pre_hook=desc_pre_hook,
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    q_desc, k_desc, v_desc, o_desc,
    stride_lseb, stride_lseh, stride_lses,
    sm_scale, lse_scale,
    B, H, S,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    m_start = pid_m * BLOCK_M
    
    # 4D TMA Load natively avoids manual masking
    q = q_desc.load([pid_b, pid_h, m_start, 0])
    q = tl.reshape(q, [BLOCK_M, D])
    
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # 1. Fully Unmasked Loop (Strictly Below Causal Diagonal Bounds)
    # Replaces runtime causal checks with full unbounded loop pipelining
    for start_n in tl.range(0, m_start, BLOCK_N, num_stages=PIPELINE_STAGES):
        k = k_desc.load([pid_b, pid_h, start_n, 0])
        k = tl.reshape(k, [BLOCK_N, D])
        
        v = v_desc.load([pid_b, pid_h, start_n, 0])
        v = tl.reshape(v, [BLOCK_N, D])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp2(m_i - m_new)
        beta = tl.exp2(qk - m_new[:, None])
        
        acc = acc * alpha[:, None]
        
        beta_bf16 = beta.to(tl.bfloat16)
        acc += tl.dot(beta_bf16, v, out_dtype=tl.float32)
        
        l_i = l_i * alpha + tl.sum(beta, 1)
        m_i = m_new

    # 2. Diagonal Block (Partially Bounded by Sequence Limits & Causal Diagonals)
    diag_end = tl.minimum(S, m_start + BLOCK_M)
    for start_n in range(m_start, diag_end, BLOCK_N):
        k = k_desc.load([pid_b, pid_h, start_n, 0])
        k = tl.reshape(k, [BLOCK_N, D])
        
        v = v_desc.load([pid_b, pid_h, start_n, 0])
        v = tl.reshape(v, [BLOCK_N, D])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        offs_m = m_start + tl.arange(0, BLOCK_M)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        causal_mask = (offs_m[:, None] >= offs_n[None, :]) & (offs_n[None, :] < S)
        
        qk = tl.where(causal_mask, qk, float("-inf"))
        
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp2(m_i - m_new)
        beta = tl.exp2(qk - m_new[:, None])
        
        acc = acc * alpha[:, None]
        
        beta_bf16 = beta.to(tl.bfloat16)
        acc += tl.dot(beta_bf16, v, out_dtype=tl.float32)
        
        l_i = l_i * alpha + tl.sum(beta, 1)
        m_i = m_new

    # Final Softmax Reduction
    out = acc / l_i[:, None]
    
    # Store through hardware TMA natively discarding arbitrary M boundaries
    out_4d = tl.reshape(out, [1, 1, BLOCK_M, D])
    o_desc.store([pid_b, pid_h, m_start, 0], out_4d.to(tl.bfloat16))
    
    LSE_ptr = LSE + pid_b * stride_lseb + pid_h * stride_lseh
    offs_m_store = m_start + tl.arange(0, BLOCK_M)
    LSE_ptrs = LSE_ptr + offs_m_store * stride_lses
    
    lse_val = (m_i + tl.log2(l_i)) * lse_scale
    tl.store(LSE_ptrs, lse_val, mask=offs_m_store < S)


def run(Q, K, V, O, LSE):
    """
    Computes causal scaled dot product attention and its log-sum-exp (LSE).
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Mathematical pre-scaling mappings offloading exponential components optimally
    sm_scale = (1.0 / math.sqrt(D)) * 1.4426950408889634
    lse_scale = 0.6931471805599453

    # Schedule: Processing batch & head linearly along Y assigns concurrent SMs sequentially along X.
    # This guarantees execution mapping aligns concurrent SMs to the identical batch + head, sharing
    # K & V TMA fetches collaboratively in the shared 126MB L2 Cache.
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        None, None, None, None,  # Descriptors correctly overwritten via runtime pre_hook
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        sm_scale, lse_scale,
        B, H, S,
        D,
    )