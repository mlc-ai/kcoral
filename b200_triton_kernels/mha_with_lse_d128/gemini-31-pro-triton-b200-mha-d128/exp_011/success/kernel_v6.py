import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    sm_scale_log2, S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    EVEN_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m_block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # Resolve base mapping 
    q_offset = batch_idx * stride_qb + head_idx * stride_qh
    k_offset = batch_idx * stride_kb + head_idx * stride_kh
    v_offset = batch_idx * stride_vb + head_idx * stride_vh
    
    offs_m = m_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 128)

    # Establish coalesced pointers. The final dimension is contiguous.
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    k_ptrs = K + k_offset + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_ptrs = V + v_offset + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    # Load Q explicitly bypassing masking overhead if length is uniform
    if EVEN_S:
        q = tl.load(q_ptrs)
    else:
        q_mask = offs_m[:, None] < S
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Core hardware registers for FA
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    k_tiles = tl.cdiv(S, BLOCK_N)
    
    # Software-pipelined K and V loop (unrolled per num_stages automatically by Triton pointer bindings)
    for kv_tile in range(0, k_tiles):
        if EVEN_S:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k_mask = (kv_tile * BLOCK_N + offs_n)[:, None] < S
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
            v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # Matmul operation mapped directly down to native MMA blocks
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        scores = scores * sm_scale_log2

        # Filter out bounds if arbitrary sequence domains are expected
        if not EVEN_S:
            valid = (kv_tile * BLOCK_N + offs_n)[None, :] < S
            scores = tl.where(valid, scores, -float("inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        
        # Exponential and reductions computed efficiently in FP32
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q.dtype), v, acc)
        
        m_i = m_ij

        # Safely advance pointer offsets along the contiguous sequence structures
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    output = acc / l_i[:, None]
    
    # Base-2 LogSumExp normalized back structurally into Base-E per PyTorch target
    lse_log2 = m_i + tl.math.log2(l_i)
    LN2: tl.constexpr = 0.6931471805599453
    lse_ln = lse_log2 * LN2

    # Map sequential memory writeback
    o_offset = batch_idx * stride_ob + head_idx * stride_oh
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od

    lse_offset = batch_idx * stride_lseb + head_idx * stride_lseh
    lse_ptrs = LSE + lse_offset + offs_m * stride_lses

    # Output emissions
    if EVEN_S:
        tl.store(o_ptrs, output.to(q.dtype))
        tl.store(lse_ptrs, lse_ln)
    else:
        q_mask_1d = offs_m < S
        tl.store(o_ptrs, output.to(q.dtype), mask=q_mask_1d[:, None])
        tl.store(lse_ptrs, lse_ln, mask=q_mask_1d)


def run(Q, K, V, O, LSE):
    """
    Computes a Multi-Head Attention forward pass returning Output and LogSumExp vectors.
    """
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Compress standard attention scaling explicitly before the loop math block execution
    sm_scale = 1.0 / (D ** 0.5)
    RCP_LN2 = 1.4426950408889634
    sm_scale_log2 = sm_scale * RCP_LN2
    
    # Pre-calculate boolean branch statically for JIT optimizations
    EVEN_S = (S % 128 == 0)
    
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        sm_scale_log2, S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        EVEN_S=EVEN_S,
    )