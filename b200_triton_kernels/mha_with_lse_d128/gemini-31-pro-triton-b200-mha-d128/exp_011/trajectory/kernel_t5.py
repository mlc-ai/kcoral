import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "NUM_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "NUM_STAGES": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "NUM_STAGES": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "NUM_STAGES": 4}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "NUM_STAGES": 4}, num_warps=4, num_stages=4),
    ],
    key=["S"],
)
@triton.jit
def _attention_kernel(
    Q, K, V, O, LSE,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    sm_scale_log2,
    IS_S_MULTIPLE_OF_128: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    m_block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # Base pointers for this batch and head
    q_base = Q + batch_idx * stride_qb + head_idx * stride_qh
    k_base = K + batch_idx * stride_kb + head_idx * stride_kh
    v_base = V + batch_idx * stride_vb + head_idx * stride_vh

    # 2D TMA Descriptors native to TMEM paths avoiding the need for tl.reshape casts
    q_desc = tl.make_tensor_descriptor(
        q_base, shape=[S, 128], strides=[stride_qs, 1], block_shape=[BLOCK_M, 128], padding_option="zero"
    )
    k_desc = tl.make_tensor_descriptor(
        k_base, shape=[S, 128], strides=[stride_ks, 1], block_shape=[BLOCK_N, 128], padding_option="zero"
    )
    v_desc = tl.make_tensor_descriptor(
        v_base, shape=[S, 128], strides=[stride_vs, 1], block_shape=[BLOCK_N, 128], padding_option="zero"
    )

    offset_m = (m_block_idx * BLOCK_M).to(tl.int32)
    
    # Load Q tile and execute base-2 scaling once 
    q = q_desc.load([offset_m, 0])
    q = (q * sm_scale_log2).to(q.dtype)

    # Core states initialization
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    k_tiles = tl.cdiv(S, BLOCK_N)
    
    # Branched out to compile flat TMEM/MMA compatible hardware loops without masking constraints
    if IS_S_MULTIPLE_OF_128:
        for kv_tile in tl.range(0, k_tiles, num_stages=NUM_STAGES):
            offset_n = (kv_tile * BLOCK_N).to(tl.int32)
            k = k_desc.load([offset_n, 0])
            v = v_desc.load([offset_n, 0])

            scores = tl.dot(q, k.T, out_dtype=tl.float32)
            
            # Since non-causal lengths > 0 guarantee at least one valid key on first pass,
            # max(scores) is definitively > -inf, saving us safety tl.where checks 
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(scores - m_ij[:, None])
            
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(q.dtype), v, acc)
            m_i = m_ij
            
    else:
        for kv_tile in tl.range(0, k_tiles, num_stages=NUM_STAGES):
            offset_n = (kv_tile * BLOCK_N).to(tl.int32)
            k = k_desc.load([offset_n, 0])
            v = v_desc.load([offset_n, 0])

            scores = tl.dot(q, k.T, out_dtype=tl.float32)
            
            offs_n = offset_n + tl.arange(0, BLOCK_N)
            scores = tl.where(offs_n[None, :] < S, scores, -float("inf"))
            
            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.math.exp2(m_i - m_ij)
            p = tl.math.exp2(scores - m_ij[:, None])
            
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(q.dtype), v, acc)
            m_i = m_ij

    # Safe normalization since l_i strictly accumulates exponents > 0.0
    output = acc / l_i[:, None]
    
    # Mathematical reduction of LogSumExp to natural log space (PyTorch expectation)
    lse_log2 = m_i + tl.math.log2(l_i)
    LN2: tl.constexpr = 0.6931471805599453
    lse_ln = lse_log2 * LN2

    # Sequence alignment output boundary mapping using normal pointers. (Out-of-bounds Q logic
    # naturally collapses safely since it is discarded sequentially here).
    offs_m = offset_m + tl.arange(0, BLOCK_M)
    q_mask = offs_m < S
    
    o_offset = batch_idx * stride_ob + head_idx * stride_oh
    lse_offset = batch_idx * stride_lseb + head_idx * stride_lseh
    offs_d = tl.arange(0, 128)
    
    o_ptrs = O + o_offset + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, output.to(q.dtype), mask=q_mask[:, None])

    lse_ptrs = LSE + lse_offset + offs_m * stride_lses
    tl.store(lse_ptrs, lse_ln, mask=q_mask)


def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device=Q.device, dtype=torch.int8)
    triton.set_allocator(alloc_fn)
    
    B, H, S, D = Q.shape
    
    sm_scale = 1.0 / (D ** 0.5)
    RCP_LN2 = 1.4426950408889634
    sm_scale_log2 = sm_scale * RCP_LN2
    
    # Extract branch status explicitly at compile time enabling TMA loops perfectly 
    is_s_multiple_of_128 = (S % 128 == 0)
    
    # L2 Cache Swizzling via axis sequencing (Q tile iterations promote stable K/V blocks)
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B, H)
    
    _attention_kernel[grid](
        Q, K, V, O, LSE,
        S,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        sm_scale_log2,
        IS_S_MULTIPLE_OF_128=is_s_multiple_of_128,
    )