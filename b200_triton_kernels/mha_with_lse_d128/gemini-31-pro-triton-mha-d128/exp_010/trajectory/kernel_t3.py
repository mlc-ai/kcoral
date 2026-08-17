import torch
import triton
import triton.language as tl
import math

def alloc_fn(size: int, alignment: int, stream):
    """
    Allocator for device-created TMA descriptors as required by standard Triton Hopper guidelines.
    """
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    ],
    key=["S"]
)
@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    sm_scale,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    b_idx = pid_bh // H
    h_idx = pid_bh % H
    
    # Calculate base pointers for the current Batch and Head
    q_base = q_ptr + b_idx * stride_qb + h_idx * stride_qh
    k_base = k_ptr + b_idx * stride_kb + h_idx * stride_kh
    v_base = v_ptr + b_idx * stride_vb + h_idx * stride_vh
    o_base = o_ptr + b_idx * stride_ob + h_idx * stride_oh
    
    # Device-created Hopper TMA descriptors (natively bounds-checked, zero-padded)
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, D], strides=[stride_qs, stride_qd],
        block_shape=[BLOCK_M, D], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, D], strides=[stride_ks, stride_kd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, D], strides=[stride_vs, stride_vd],
        block_shape=[BLOCK_N, D], padding_option="zero"
    )
    o_desc = tl.make_tensor_descriptor(
        o_base, shape=[S, D], strides=[stride_os, stride_od],
        block_shape=[BLOCK_M, D]
    )
    
    offset_m = pid_m * BLOCK_M
    
    # Load Q block once per CTA
    q = q_desc.load([offset_m, 0])
    
    # Scale Q upfront using LOG2_E to replace tl.exp with the hardware-native tl.exp2 in the inner loop
    LOG2_E = 1.4426950408889634
    q = (q * (sm_scale * LOG2_E)).to(tl.bfloat16)
    
    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    num_full_blocks = S // BLOCK_N
    
    # -------------------------------------------------------------------------
    # Manual Software Pipeline: Overlapping Softmax (MFU) with GEMM (WGMMA/TC)
    # -------------------------------------------------------------------------
    if num_full_blocks > 0:
        offset_n = 0
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        # Initial WGMMA for the very first block
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        for start_n in range(1, num_full_blocks):
            # Asynchronously issue next TMA loads
            offset_n = start_n * BLOCK_N
            k_next = k_desc.load([offset_n, 0])
            v_next = v_desc.load([offset_n, 0])
            
            # Execute Softmax on the MFU for the *current* iteration
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            alpha = tl.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            
            # Scale running output
            acc = acc * alpha[:, None]
            
            # Concurrently issue WGMMA QK dot for the *next* iteration
            # This completely hides the latency of the MFU Softmax logic
            qk_next = tl.dot(q, k_next.T, out_dtype=tl.float32)
            
            # Issue WGMMA PV dot for the *current* iteration
            acc = tl.dot(p.to(tl.bfloat16), v, acc)
            
            # Advance cyclic pipeline registers
            m_i = m_ij
            k = k_next
            v = v_next
            qk = qk_next

        # Epilogue: Softmax and PV dot for the last full block
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # -------------------------------------------------------------------------
    # Tail Block (if S is not perfectly divisible by BLOCK_N)
    # -------------------------------------------------------------------------
    if S % BLOCK_N != 0:
        offset_n = num_full_blocks * BLOCK_N
        k = k_desc.load([offset_n, 0])
        v = v_desc.load([offset_n, 0])
        
        qk = tl.dot(q, k.T, out_dtype=tl.float32)
        
        # Explicit bounds masking since TMA zero-padding affects Softmax correctness for QK
        offs_n = offset_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        qk = tl.where(mask_n[None, :], qk, float('-inf'))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_ij

    # -------------------------------------------------------------------------
    # Normalization and Storage
    # -------------------------------------------------------------------------
    acc = acc / l_i[:, None]
    
    # Convert base-2 scaled tracking values back to standard natural-log LSE
    LN_2 = 0.6931471805599453
    lse = (m_i + tl.log2(l_i)) * LN_2
    
    # TMA dynamically drops bounds-violating M rows on store
    o_desc.store([offset_m, 0], acc.to(tl.bfloat16))
    
    # Store auxiliary LSE output conventionally (masking guarantees safety for out-of-bounds M rows)
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < S
    lse_ptrs = lse_ptr + b_idx * stride_lseb + h_idx * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=mask_m)


def run(Q, K, V, O, LSE):
    """
    Computes a batched non-causal multi-head attention forward pass via optimized software pipeline overlapping.
    Inputs: Q, K, V: bfloat16 [B, H, S, D]
    Outputs: O: bfloat16 [B, H, S, D], LSE: float32 [B, H, S]
    """
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    sm_scale = 1.0 / math.sqrt(D)
    
    # The M dimension effectively acts as the inner axis maximizing L2 cache hits for contiguous K and V iterations
    grid = lambda META: (
        triton.cdiv(S, META['BLOCK_M']),
        B * H
    )
    
    _fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H=H,
        D=D,
    )