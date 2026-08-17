import math
import torch
import triton
import triton.language as tl

# Set up the allocator required for Triton's device-side tensor descriptors (Hopper TMA requirement)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale_log2,
    O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    start_m = pid_m * BLOCK_M
    # Early exit if the query block is fully out-of-bounds
    if start_m >= S:
        return
        
    # Base offsets for the current batch and head
    q_offset = b_idx * stride_qb + h_idx * stride_qh
    k_offset = b_idx * stride_kb + h_idx * stride_kh
    v_offset = b_idx * stride_vb + h_idx * stride_vh
    o_offset = b_idx * stride_ob + h_idx * stride_oh
    
    # Hopper Tensor Memory Accelerator (TMA) descriptors for asynchronous loading
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset, shape=[S, BLOCK_D], strides=[stride_qs, 1], block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset, shape=[S, BLOCK_D], strides=[stride_ks, 1], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset, shape=[S, BLOCK_D], strides=[stride_vs, 1], block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        O + o_offset, shape=[S, BLOCK_D], strides=[stride_os, 1], block_shape=[BLOCK_M, BLOCK_D]
    )
    
    # Load query block
    q = q_desc.load([start_m, 0])
    
    # Initialize softmax accumulators in FP32
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Causal attention split: 
    # 1. Fully unmasked blocks (keys strictly before queries)
    # 2. Partially masked blocks (keys overlap with queries or sequence boundary)
    num_unmasked = start_m // BLOCK_N
    
    # Unmasked loop: leverages full WGMMA hardware pipeline without masking overhead
    for start_n_idx in range(0, num_unmasked):
        start_n = start_n_idx * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        # WGMMA SS: [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, acc=qk)
        qk *= sm_scale_log2
        
        # Online softmax update using base-2 math for performance
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # WGMMA RS: [BLOCK_M, BLOCK_N] @ [BLOCK_N, BLOCK_D] -> [BLOCK_M, BLOCK_D]
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc)
        
        m_i = m_ij

    # Masked loop: strict causal masking & sequence boundary protection
    end_n_limit = tl.minimum(S, start_m + BLOCK_M)
    num_total_blocks = tl.cdiv(end_n_limit, BLOCK_N)
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    
    for start_n_idx in range(num_unmasked, num_total_blocks):
        start_n = start_n_idx * BLOCK_N
        k = k_desc.load([start_n, 0])
        v = v_desc.load([start_n, 0])
        
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.T, acc=qk)
        qk *= sm_scale_log2
        
        # Apply lower-triangular causal mask & handle sequence edges for keys
        key_idx = start_n + offs_n_base[None, :]
        valid_mask = (offs_m[:, None] >= key_idx) & (key_idx < S)
        qk = tl.where(valid_mask, qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        p_bf16 = p.to(tl.bfloat16)
        acc = tl.dot(p_bf16, v, acc=acc)
        
        m_i = m_ij

    # Epilogue: normalize output by LSE components
    acc = acc / l_i[:, None]
    
    # Scale max value back to natural log domain and compute finalized LSE
    lse = m_i * 0.6931471805599453 + tl.log(l_i)
    
    # Store matrix output through TMA block descriptor (filters out-of-bounds rows internally)
    o_desc.store([start_m, 0], acc.to(tl.bfloat16))
    
    # Store auxiliary LSE metrics conventionally via pointers with strict bounds checks
    lse_ptrs = LSE + b_idx * stride_lseb + h_idx * stride_lseh + offs_m * stride_lses
    lse_mask = offs_m < S
    tl.store(lse_ptrs, lse, mask=lse_mask)


def run(Q, K, V, O, LSE):
    """
    Computes causal multi-head attention forward pass optimized for NVIDIA Hopper SM90+.
    Leverages hardware features Tensor Memory Accelerator (TMA) and WGMMA Core execution paths.
    
    Inputs:
        Q, K, V: bfloat16 tensors of shape [B, H, S, D]
    Outputs:
        O: bfloat16 tensor of shape [B, H, S, D]
        LSE: float32 tensor of shape [B, H, S]
    """
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    
    # Pre-compute the log2 factor for the kernel base-2 operations (log2(e))
    sm_scale_log2 = sm_scale * 1.4426950408889634
    
    # TMA descriptors guarantee 16-byte alignment assumption limits on contiguous memory 
    assert Q.stride(-1) == 1 and K.stride(-1) == 1 and V.stride(-1) == 1 and O.stride(-1) == 1
    
    torch.cuda.set_device(Q.device)
    
    # Configure 3D execution grid: prioritize `S` (M-dim) fastest to naturally distribute cache loads
    grid = lambda META: (
        triton.cdiv(S, META["BLOCK_M"]),
        B * H,
        1
    )
    
    _attn_fwd_kernel[grid](
        Q, K, V, sm_scale_log2,
        O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S,
        BLOCK_D=128
    )