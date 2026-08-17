import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O, LSE,
    sm_scale_log2,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    H,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Setup coordinates
    start_m = tl.program_id(0)
    batch_head = tl.program_id(1)
    
    batch_idx = batch_head // H
    head_idx = batch_head % H
    
    # Fast exit if the sequence tile is fully out of sequence length bounds
    if start_m * BLOCK_M >= S:
        return

    # Setup offsets and unroll sequences natively 
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    
    # Base pointers mapped to standard layout
    q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + batch_idx * stride_kb + head_idx * stride_kh + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + batch_idx * stride_vb + head_idx * stride_vh + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Load fully stationary Q into registers/SRAM
    q_mask = offs_m < S
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    
    # Maintain accumulators in stable fp32
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    
    # Compute boundary conditions for fully unmasked fast path vs casual/sequence masked boundary
    num_full_blocks = (start_m * BLOCK_M) // BLOCK_N
    end_n_blocks = tl.cdiv(tl.minimum(S, (start_m + 1) * BLOCK_M), BLOCK_N)
    
    # ================= Phase 1: Fully unmasked unrolled pipelining =================
    # Blocks handled here are guaranteed entirely in-bounds & compliant with causality
    for start_n_idx in range(0, num_full_blocks):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale_log2
        
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i_new = l_i * alpha + tl.sum(p, axis=1)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # ================= Phase 2: Causal boundary & sequence masked blocks =================
    for start_n_idx in range(num_full_blocks, end_n_blocks):
        offs_n_curr = start_n_idx * BLOCK_N + offs_n
        k_mask = offs_n_curr < S
        
        k = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        
        qk = tl.dot(q, k.T)
        qk = qk * sm_scale_log2
        
        valid_mask = (offs_m[:, None] >= offs_n_curr[None, :]) & k_mask[None, :]
        # Keep natively out-of-bounds Q rows as unmasked padding computations (=0.0) safely preventing NaN explosions
        final_mask = valid_mask | (~q_mask[:, None])
        qk = tl.where(final_mask, qk, float("-inf"))
        
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_i_new)
        p = tl.exp2(qk - m_i_new[:, None])
        
        l_i_new = l_i * alpha + tl.sum(p, axis=1)
        
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc=acc)
        
        m_i = m_i_new
        l_i = l_i_new
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    # ================= Final Softmax Normalization =================
    acc = acc / l_i[:, None]
    
    o_ptrs = O + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=q_mask[:, None])
    
    # Reconvert LSE back to standard scale from highly optimized base-2 implementation mappings
    ln2 = 0.6931471805599453
    lse = (m_i + tl.log2(l_i)) * ln2
    lse_ptrs = LSE + batch_idx * stride_lseb + head_idx * stride_lseh + offs_m * stride_lses
    tl.store(lse_ptrs, lse, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    B, H, S, D = Q.shape
    
    # Incorporate log2_e shift strictly into the scalar multiple mapping avoiding inner-loop `tl.exp()` overhead limits
    sm_scale = 1.0 / (D ** 0.5)
    log2_e = 1.4426950408889634
    sm_scale_log2 = sm_scale * log2_e
    
    # 2D Grid grouping guarantees Hopper's L2 Cache implicitly reuses K & V accurately without explicit swizzling
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    _attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale_log2,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        H,
        D=D,
    )