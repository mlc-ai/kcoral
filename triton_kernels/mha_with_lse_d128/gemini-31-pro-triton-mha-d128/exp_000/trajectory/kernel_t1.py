import torch
import triton
import triton.language as tl

# Set up the descriptor allocator required for device-side tensor descriptors (TMA).
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    S, H,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    
    # Identify batch and head for current program
    b = off_bh // H
    h = off_bh % H
    
    # Compute base pointers for the specific batch and head
    q_ptr = Q + b * stride_qb + h * stride_qh
    k_ptr = K + b * stride_kb + h * stride_kh
    v_ptr = V + b * stride_vb + h * stride_vh
    o_ptr = O + b * stride_ob + h * stride_oh
    
    # Device-created descriptors for native Hopper TMA loads
    q_desc = tl.make_tensor_descriptor(
        q_ptr,
        shape=[S, BLOCK_D],
        strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, BLOCK_D],
        padding_option="zero"
    )
    
    k_desc = tl.make_tensor_descriptor(
        k_ptr,
        shape=[S, BLOCK_D],
        strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    
    v_desc = tl.make_tensor_descriptor(
        v_ptr,
        shape=[S, BLOCK_D],
        strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, BLOCK_D],
        padding_option="zero"
    )
    
    # Descriptor for fast TMA store of output
    o_desc = tl.make_tensor_descriptor(
        o_ptr,
        shape=[S, BLOCK_D],
        strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, BLOCK_D]
    )
    
    # Load the Query block once, avoiding repeated memory transactions
    q = q_desc.load([start_m * BLOCK_M, 0])
    
    # Initialize running statistics (Flash Attention setup)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Standard software-pipelined K & V fetching loop
    for start_n in range(0, S, BLOCK_N):
        # 1. Load keys and calculate raw attention scores
        k = k_desc.load([start_n, 0])
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        # 2. Mask the sequence length boundary
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        qk = tl.where(mask_n[None, :], qk, float('-inf'))
        
        # 3. Softmax numerical stabilizations (keep track of max and sums)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        # 4. Multiply with Value and aggregate
        v = v_desc.load([start_n, 0])
        p_bfloat16 = p.to(tl.bfloat16)
        acc += tl.dot(p_bfloat16, v, out_dtype=tl.float32)
        
        # 5. Commit statistics sequentially
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        
    # Finalize normalization for Output
    acc = acc / l_i[:, None]
    
    # LogSumExp (LSE): maximum_score + natural_log(sum_of_exps)
    lse = m_i + tl.log(l_i)
    
    # TMA properly ignores boundary rows >= S automatically via shape=[S, BLOCK_D]
    o_desc.store([start_m * BLOCK_M, 0], acc.to(tl.bfloat16))
    
    # Store LSE linearly via masked pointer assignment
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = LSE + b * stride_lseb + h * stride_lseh + offs_m * stride_lses
    mask_m = offs_m < S
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Executes a heavily optimized, memory-efficient non-causal Flash Attention 
    forward pass using native Hopper Tensor Memory Accelerator (TMA) and WGMMA features.
    
    Inputs:
        Q, K, V: [B, H, S, 128] shaped bfloat16 tensors.
    Outputs:
        O: Preallocated [B, H, S, 128] bfloat16 tensor.
        LSE: Preallocated [B, H, S] float32 tensor.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    sm_scale = 1.0 / (D ** 0.5)
    
    # 2D Grid: M-tiles spread across dimension 0, and batch*head elements mapped directly to dimension 1.
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        S, H,
        sm_scale,
        BLOCK_D=128
    )