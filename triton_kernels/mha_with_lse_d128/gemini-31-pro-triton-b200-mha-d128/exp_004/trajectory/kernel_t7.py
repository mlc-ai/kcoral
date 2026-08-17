import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# A pre_hook enables dynamic block shape mutation during Triton Autotuning tests 
# generating exactly one pristine 2D host-side TMA descriptor per execution.
def tma_pre_hook(args):
    Q_flat = args["Q_flat"]
    K_flat = args["K_flat"]
    V_flat = args["V_flat"]
    BM = args["BLOCK_M"]
    BN = args["BLOCK_N"]
    args["q_desc"] = TensorDescriptor.from_tensor(Q_flat, [BM, 128])
    args["k_desc"] = TensorDescriptor.from_tensor(K_flat, [BN, 128])
    args["v_desc"] = TensorDescriptor.from_tensor(V_flat, [BN, 128])

def get_tma_configs():
    return [
        # Aggressively target maximum K and V memory multicast streaming via L2 Cache
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3, pre_hook=tma_pre_hook),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3, pre_hook=tma_pre_hook),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=4, num_stages=4, pre_hook=tma_pre_hook),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=8, num_stages=4, pre_hook=tma_pre_hook),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "PIPELINE_STAGES": 4}, num_warps=4, num_stages=4, pre_hook=tma_pre_hook),
    ]

@triton.autotune(configs=get_tma_configs(), key=["S"])
@triton.jit
def _attn_fwd_tma_2d_kernel(
    Q_flat, K_flat, V_flat,
    q_desc, k_desc, v_desc,
    O, LSE,
    S, H,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # Mathematical breakdown traversing the flattened 2D Sequence Matrix representation
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m_start = pid_m * BLOCK_M
    seq_offset = pid_bh * S

    # Perfect Native 2D TMA Loads feeding cleanly into Tensor Core mm ops
    q = q_desc.load([seq_offset + offs_m_start, 0])
    
    # Math folded scaling minimizing redundant FP32 instructions -> 1 / sqrt(128) * log2(e)
    SCALE_LN2: tl.constexpr = 0.08838834764831843 * 1.4426950408889634
    q = (q * SCALE_LN2).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    limit = (S // BLOCK_N) * BLOCK_N
    num_blocks = limit // BLOCK_N

    # Pipelined hardware loop unrolling streaming loads ahead of Tensor Core computation cycles
    for block_idx in tl.range(0, num_blocks, num_stages=PIPELINE_STAGES):
        start_n = block_idx * BLOCK_N
        k = k_desc.load([seq_offset + start_n, 0])
        v = v_desc.load([seq_offset + start_n, 0])
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        m_i = m_ij

    # Masked sequence bounds handling (Isolated and unrolled from fast pathway limits)
    if limit < S:
        k = k_desc.load([seq_offset + limit, 0])
        v = v_desc.load([seq_offset + limit, 0])
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        offs_n = limit + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S
        scores = tl.where(mask_n[None, :], scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        m_i = m_ij

    # Stable base transformations converting from Base-2 exponent forms
    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    lse = (m_i + tl.math.log2(safe_l_i)) * 0.6931471805599453

    offs_m = offs_m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, 128)
    
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    
    mask_m = offs_m < S
    tl.store(o_ptrs, output.to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(lse_ptrs, lse, mask=mask_m)


def get_ptr_configs():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PIPELINE_STAGES": 3}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "PIPELINE_STAGES": 4}, num_warps=4, num_stages=4),
    ]

@triton.autotune(configs=get_ptr_configs(), key=["S"])
@triton.jit
def _attn_fwd_ptr_kernel(
    Q, K, V, O, LSE,
    S, H,
    stride_qb, stride_qh, stride_qs,
    stride_kb, stride_kh, stride_ks,
    stride_vb, stride_vh, stride_vs,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lseb, stride_lseh, stride_lses,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m_start = pid_m * BLOCK_M
    offs_m = offs_m_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, 128)

    q_base = Q + pid_b * stride_qb + pid_h * stride_qh
    k_base = K + pid_b * stride_kb + pid_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_h * stride_vh

    q_ptrs = q_base + offs_m[:, None] * stride_qs + offs_d[None, :]
    k_ptrs = k_base + offs_n[:, None] * stride_ks + offs_d[None, :]
    v_ptrs = v_base + offs_n[:, None] * stride_vs + offs_d[None, :]

    mask_m = offs_m < S
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    SCALE_LN2: tl.constexpr = 0.08838834764831843 * 1.4426950408889634
    q = (q * SCALE_LN2).to(tl.bfloat16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 128), tl.float32)

    limit = (S // BLOCK_N) * BLOCK_N
    num_blocks = limit // BLOCK_N

    for block_idx in tl.range(0, num_blocks, num_stages=PIPELINE_STAGES):
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        m_i = m_ij
        
        k_ptrs += BLOCK_N * stride_ks
        v_ptrs += BLOCK_N * stride_vs

    if limit < S:
        offs_n_tail = limit + offs_n
        mask_n = offs_n_tail < S
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        scores = tl.dot(q, k.T, out_dtype=tl.float32)
        scores = tl.where(mask_n[None, :], scores, -float("inf"))
        
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(scores - m_ij[:, None])
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc, out_dtype=tl.float32)
        m_i = m_ij

    safe_l_i = tl.where(l_i == 0.0, 1.0, l_i)
    output = acc / safe_l_i[:, None]
    lse = (m_i + tl.math.log2(safe_l_i)) * 0.6931471805599453

    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lses
    
    tl.store(o_ptrs, output.to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(lse_ptrs, lse, mask=mask_m)

def run(Q, K, V, O, LSE):
    torch.cuda.set_device(Q.device)
    B, H, S, D = Q.shape
    
    # Broadcast identical K/V dependencies onto identically overlapping L2 clusters across Sequence dimensions
    grid = lambda META: (triton.cdiv(S, META["BLOCK_M"]), B * H)
    
    # Safely evaluate PyTorch structural forms to ensure Native 2D TMA support
    tma_supported = Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

    if tma_supported:
        # Formally flattening inputs identically to host tensor rules generating single 2D views seamlessly
        Q_flat = Q.view(-1, D)
        K_flat = K.view(-1, D)
        V_flat = V.view(-1, D)
        
        _attn_fwd_tma_2d_kernel[grid](
            Q_flat, K_flat, V_flat,
            None, None, None,
            O, LSE,
            S, H,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
        )
    else:
        _attn_fwd_ptr_kernel[grid](
            Q, K, V, O, LSE,
            S, H,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
        )