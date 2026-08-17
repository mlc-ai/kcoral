import torch
import triton
import triton.language as tl
import math

# Configure Triton's allocator for device-created descriptors (TMA)
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        # Configurations tailored strictly to Blackwell SM100's 228 KB shared memory envelope
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPELINE_STAGES": 2}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "PIPELINE_STAGES": 3}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, O, LSE,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_lseb, stride_lseh, stride_lses,
    B, H, S, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, 
    BLOCK_D: tl.constexpr, PIPELINE_STAGES: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Base pointers per batch & head
    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh
    o_base = O + pid_b * stride_ob + pid_h * stride_oh

    # TMA hardware descriptors ensuring global layout alignments
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, BLOCK_D], strides=[stride_qs, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, BLOCK_D], strides=[stride_ks, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, BLOCK_D], strides=[stride_vs, 1],
        block_shape=[BLOCK_N, BLOCK_D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, BLOCK_D], strides=[stride_os, 1],
        block_shape=[BLOCK_M, BLOCK_D], padding_option="zero"
    )

    offset_m = (pid_m * BLOCK_M).to(tl.int32)
    q = q_desc.load([offset_m, 0])

    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    kv_tiles = tl.cdiv(S, BLOCK_N)
    
    offs_m_v = offset_m + tl.arange(0, BLOCK_M)
    q_valid = offs_m_v < S
    
    # Standard linear loop equipped with robust internal pipelining attributes
    # overlapping TMA memory bulk copies identically mapped to computation waves
    for kv_tile in tl.range(0, kv_tiles, num_stages=PIPELINE_STAGES):
        offset_n = (kv_tile * BLOCK_N).to(tl.int32)
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])

        # Execute Tensor Core dot product dynamically retaining float32 scale internally
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        qk = qk * sm_scale
        
        offs_n_v = offset_n + tl.arange(0, BLOCK_N)
        valid_mask = q_valid[:, None] & (offs_n_v[None, :] < S)
        qk = tl.where(valid_mask, qk, -float("inf"))

        # Row-wise max 
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        
        # Base-2 Logarithmic transformation natively aligned to HW capabilities
        alpha = tl.math.exp2(m_i - safe_m_ij)
        p = tl.math.exp2(qk - safe_m_ij[:, None])

        # Accumulate sums
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        m_i = m_ij

    # Final softmax norm denominator completion
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    
    # Compute LSE explicitly in natural log mapping baseline behaviors
    lse_log2 = tl.where(l_i == 0.0, -float("inf"), m_i + tl.math.log2(safe_l_i))
    LN2 = 0.6931471805599453
    lse = lse_log2 * LN2

    # Emit output array through Native Hardware Descriptors natively bounding boundaries
    o_desc.store([offset_m, 0], output.to(tl.bfloat16))

    # Standard contiguous line store of Log-Sum-Exp statistics via mask filtering
    lse_ptrs = LSE + (pid_b * stride_lseb + pid_h * stride_lseh) + offs_m_v * stride_lses
    tl.store(lse_ptrs, lse, mask=q_valid)

def run(Q, K, V, O, LSE):
    """
    Computes streaming attention forward mapping relying solely on Blackwell standard descriptors.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Compute FA exponential scale integrating standard scaling and base e multiplier dynamically
    RCP_LN2 = 1.4426950408889634
    sm_scale = (1.0 / math.sqrt(D)) * RCP_LN2

    # Group inherently interleaves identically scheduled block regions
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        B, H, S, sm_scale,
        BLOCK_D=D
    )